"""Pure-PyTorch checks for the layout-exact NVFP4 reference."""

import torch

from benchmarks import reference as ref

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def test_scale_factor_atom_mapping():
    rows, sf_cols = 128, 4
    dense = torch.arange(rows * sf_cols, device=DEVICE, dtype=torch.float32)
    dense = dense.remainder(200).reshape(rows, sf_cols).to(torch.float8_e4m3fn)
    blocked = ref.pack_sf_blocked(dense).view(torch.uint8)
    source = dense.view(torch.uint8)
    for row in (0, 1, 31, 32, 33, 95, 127):
        for col in range(sf_cols):
            assert blocked[0, 0, row % 32, row // 32, col] == source[row, col]


def test_scale_factor_pack_roundtrip():
    torch.manual_seed(1)
    for rows, sf_cols in ((128, 4), (256, 128), (300, 47), (1024, 48)):
        dense = (torch.randn(rows, sf_cols, device=DEVICE) * 3).to(torch.float8_e4m3fn)
        blocked = ref.pack_sf_blocked(dense)
        restored = ref.unpack_sf_blocked(blocked, rows, sf_cols)
        assert torch.equal(restored.view(torch.uint8), dense.view(torch.uint8))


def test_quantize_dequantize_accuracy():
    torch.manual_seed(2)
    x = torch.randn(2048, 1024, device=DEVICE, dtype=torch.bfloat16)
    scale = ref.per_tensor_scale_from_amax(x.abs().max())
    qdata, sf = ref.quantize_nvfp4_lastdim(x, scale)
    restored = ref.dequantize_nvfp4_lastdim(qdata, sf, scale)
    rel = (restored - x.float()).norm() / x.float().norm()
    cosine = torch.nn.functional.cosine_similarity(restored.flatten(), x.float().flatten(), dim=0)
    assert rel.item() < 0.12
    assert cosine.item() > 0.99


def test_varlen_scale_factor_offsets_do_not_overlap():
    cu = torch.tensor([0, 100, 300, 300, 812], dtype=torch.int32)
    offsets = ref.varlen_sf_tile_offsets(cu)
    assert offsets == [0, 1, 4, 5]
    for expert, offset in enumerate(offsets):
        tiles = ref.ceil_div(int(cu[expert + 1]) - int(cu[expert]), 128)
        next_offset = (
            offsets[expert + 1] if expert + 1 < len(offsets) else ref.varlen_sf_num_tiles(cu)
        )
        assert offset + tiles <= next_offset
    assert ref.varlen_sf_num_tiles(cu) == 9


def test_fused_reference_layout_consistency():
    torch.manual_seed(3)
    tokens, hidden, experts, topk = 512, 256, 4, 2
    x = torch.randn(tokens, hidden, device=DEVICE, dtype=torch.bfloat16)
    num_assignments = tokens * topk
    gather = torch.randint(0, tokens, (num_assignments,), device=DEVICE, dtype=torch.int32)
    counts = torch.tensor([num_assignments // experts] * experts, dtype=torch.int32)
    cu = torch.cat((torch.zeros(1, dtype=torch.int32), counts.cumsum(0).to(torch.int32)))
    scale = ref.per_tensor_scale_from_amax(x.abs().max())

    output = ref.fused_gather_dual_quantize_ref(x, gather, cu.to(DEVICE), scale)
    gathered = x[gather.long()]
    direct_q, direct_sf = ref.quantize_nvfp4_lastdim(gathered, scale)
    assert torch.equal(output["rowwise"]["qdata"], direct_q)

    sf_buffer = output["rowwise"]["sf"]
    offsets = output["rowwise"]["sf_tile_offsets"].tolist()
    cu_list = cu.tolist()
    for expert in range(experts):
        lo, hi = cu_list[expert], cu_list[expert + 1]
        block = sf_buffer[offsets[expert] : offsets[expert] + ref.ceil_div(hi - lo, 128)]
        restored = ref.unpack_sf_blocked(block, hi - lo, hidden // 16)
        assert torch.equal(restored.view(torch.uint8), direct_sf[lo:hi].view(torch.uint8))

    colwise = output["colwise"]
    expert = 1
    lo, hi = cu_list[expert], cu_list[expert + 1]
    offset = int(colwise["seg_offsets"][expert])
    padded = int(colwise["seg_padded_lens"][expert])
    qdata = colwise["qdata"][:, offset // 2 : (offset + padded) // 2]
    sf = colwise["sf_2d"][:, offset // 16 : (offset + padded) // 16]
    restored = ref.dequantize_nvfp4_lastdim(qdata, sf, colwise["per_tensor_scale"])[:, : hi - lo]
    target = gathered[lo:hi].t().float()
    cosine = torch.nn.functional.cosine_similarity(restored.flatten(), target.flatten(), dim=0)
    assert cosine.item() > 0.99


def test_traffic_model_bounds():
    traffic = ref.traffic_bytes(T=32768, d=2048, k=8)
    assert 2.2 <= traffic["speedup_noreuse"] <= 2.4
    assert 3.8 <= traffic["speedup_reuse"] <= 4.0
    assert max(traffic["speedup_noreuse"], traffic["speedup_reuse"]) < 4.5
