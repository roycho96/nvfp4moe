"""Capture layer-0 MoE inputs, routing and weights from published checkpoints."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import torch
from torch import nn

from q32_specs import ModelSpec, canonical_trace_key


DATASET_ID = "HuggingFaceFW/fineweb-edu"
DATASET_CONFIG = "sample-10BT"
DATASET_REVISION = "v1.0.0"


class ExpertTraceRecorder(nn.Module):
    def __init__(self):
        super().__init__()
        self.captured = None

    def forward(self, hidden_states, top_k_index, top_k_weights):
        self.captured = (
            hidden_states.detach(),
            top_k_index.detach().to(torch.int32),
            top_k_weights.detach().float(),
        )
        return torch.zeros_like(hidden_states)


def _text_config(raw_config: dict) -> dict:
    return raw_config.get("text_config", raw_config)


def _verify_config(spec: ModelSpec, raw_config: dict) -> None:
    cfg = _text_config(raw_config)
    expected = {
        "hidden_size": spec.hidden,
        "moe_intermediate_size": spec.intermediate,
        "num_experts": spec.experts,
    }
    topk_key = "top_k_experts" if "top_k_experts" in cfg else "num_experts_per_tok"
    expected[topk_key] = spec.topk
    for key, value in expected.items():
        got = cfg.get(key)
        if got != value:
            raise RuntimeError(
                f"checkpoint config drift for {spec.model_id}: {key}={got!r}, expected {value!r}"
            )


def _tokenize_fineweb(spec: ModelSpec, seq_len: int):
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(spec.model_id, use_fast=True)
    ds = load_dataset(
        DATASET_ID,
        DATASET_CONFIG,
        split="train",
        streaming=True,
        revision=DATASET_REVISION,
    )
    token_ids = []
    rows = []
    bos = tokenizer.bos_token_id
    if bos is not None:
        token_ids.append(int(bos))
    for row in ds:
        text = row.get("text") or ""
        if not text:
            continue
        ids = tokenizer(text, add_special_tokens=False).input_ids
        if not ids:
            continue
        token_ids.extend(int(x) for x in ids)
        rows.append({
            "id": row.get("id"),
            "url": row.get("url"),
            "token_count": len(ids),
        })
        if len(token_ids) >= seq_len:
            break
    if len(token_ids) < seq_len:
        raise RuntimeError(f"FineWeb stream ended at {len(token_ids)} tokens")
    ids = torch.tensor(token_ids[:seq_len], dtype=torch.long).view(1, seq_len)
    digest = hashlib.sha256(ids.numpy().tobytes()).hexdigest()
    return ids, rows, digest


def _download_selected_state(spec: ModelSpec, cache_dir: str):
    """Return raw config and tensors from embed + layer-0 shards only."""
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    cfg_path = hf_hub_download(spec.model_id, "config.json", cache_dir=cache_dir)
    raw_config = json.loads(Path(cfg_path).read_text())
    _verify_config(spec, raw_config)

    index_path = hf_hub_download(
        spec.model_id, "model.safetensors.index.json", cache_dir=cache_dir
    )
    index = json.loads(Path(index_path).read_text())
    weight_map = index["weight_map"]
    # Gemma 4 is multimodal: both the text tower and the vision tower contain
    # a ``layers.0`` with identically named norms.  Pulling both makes the
    # suffix-based one-layer loader deliberately reject those ambiguous keys.
    # Scope Gemma to the language-model tower; Qwen has only ``model.layers``.
    layer0_marker = (
        ".language_model.layers.0."
        if spec.key.startswith("gemma4")
        else ".layers.0."
    )
    selected_names = [
        name for name in weight_map
        if layer0_marker in name or name.endswith("embed_tokens.weight")
    ]
    if not selected_names:
        raise RuntimeError(f"no layer-0 tensors found in {spec.model_id} index")
    shards = sorted({weight_map[name] for name in selected_names})
    shard_paths = {
        shard: hf_hub_download(spec.model_id, shard, cache_dir=cache_dir)
        for shard in shards
    }
    wanted = set(selected_names)
    state = {}
    for shard, path in shard_paths.items():
        shard_names = [name for name in selected_names if weight_map[name] == shard]
        with safe_open(path, framework="pt", device="cpu") as handle:
            for name in shard_names:
                if name in wanted:
                    state[name] = handle.get_tensor(name)
    return raw_config, state, shards


def _find_one(state: dict[str, torch.Tensor], suffix: str) -> tuple[str, torch.Tensor]:
    found = [(name, value) for name, value in state.items() if name.endswith(suffix)]
    if len(found) != 1:
        raise RuntimeError(f"expected one tensor ending {suffix!r}, found {[x[0] for x in found]}")
    return found[0]


def _expert_weights(spec: ModelSpec, state: dict[str, torch.Tensor]):
    """Normalize grouped (Gemma 4) and per-expert (Qwen 3) checkpoints."""
    grouped_gate_up = [
        value for name, value in state.items() if name.endswith(".experts.gate_up_proj")
    ]
    grouped_down = [
        value for name, value in state.items() if name.endswith(".experts.down_proj")
    ]
    if len(grouped_gate_up) == len(grouped_down) == 1:
        return grouped_gate_up[0], grouped_down[0]

    gate_up = []
    down = []
    for expert in range(spec.experts):
        prefix = f".experts.{expert}."
        _, gate = _find_one(state, prefix + "gate_proj.weight")
        _, up = _find_one(state, prefix + "up_proj.weight")
        _, down_weight = _find_one(state, prefix + "down_proj.weight")
        gate_up.append(torch.cat((gate, up), dim=0))
        down.append(down_weight)
    return torch.stack(gate_up), torch.stack(down)


def _model_state_for_one_layer(model: nn.Module, raw_state: dict[str, torch.Tensor]):
    """Suffix-match checkpoint names to a one-layer HF text model."""
    out = {}
    for target in model.state_dict():
        matches = [value for name, value in raw_state.items() if name.endswith(target)]
        if len(matches) == 1:
            out[target] = matches[0]
    return out


def _build_capture_model(spec: ModelSpec, raw_config: dict):
    from transformers import AutoConfig, AutoModel

    cfg = AutoConfig.from_pretrained(spec.model_id)
    if spec.key.startswith("gemma4"):
        from transformers import Gemma4TextModel

        cfg = copy.deepcopy(cfg.text_config)
        cfg.num_hidden_layers = 1
        cfg.layer_types = ["sliding_attention"]
        cfg.num_kv_shared_layers = 0
        model = Gemma4TextModel(cfg)
        recorder = ExpertTraceRecorder()
        model.layers[0].experts = recorder
    else:
        cfg = copy.deepcopy(cfg)
        cfg.num_hidden_layers = 1
        model = AutoModel.from_config(cfg)
        recorder = ExpertTraceRecorder()
        model.layers[0].mlp.experts = recorder
    return model, recorder


@torch.inference_mode()
def capture_trace(spec: ModelSpec, seq_len: int, cache_dir: str, output_path: str):
    raw_config, raw_state, shards = _download_selected_state(spec, cache_dir)
    input_ids, rows, token_hash = _tokenize_fineweb(spec, seq_len)
    model, recorder = _build_capture_model(spec, raw_config)
    loadable = _model_state_for_one_layer(model, raw_state)
    missing, _ = model.load_state_dict(loadable, strict=False)
    critical_missing = [
        name for name in missing
        if ("layers.0" in name or name == "embed_tokens.weight")
        and "experts" not in name
    ]
    if critical_missing:
        raise RuntimeError(f"missing layer-0 checkpoint tensors: {critical_missing[:20]}")

    model = model.to(device="cuda", dtype=torch.bfloat16).eval()
    input_ids_cuda = input_ids.cuda()
    block_input = model.embed_tokens(input_ids_cuda).detach()
    attention_mask = torch.ones_like(input_ids_cuda)
    model(input_ids=input_ids_cuda, attention_mask=attention_mask, use_cache=False)
    torch.cuda.synchronize()
    if recorder.captured is None:
        raise RuntimeError("expert recorder was not called")
    x, topi, topv = recorder.captured

    w_gate_up, w_down = _expert_weights(spec, raw_state)
    if tuple(w_gate_up.shape) != (spec.experts, 2 * spec.intermediate, spec.hidden):
        raise RuntimeError(f"unexpected gate_up shape {tuple(w_gate_up.shape)}")
    if tuple(w_down.shape) != (spec.experts, spec.hidden, spec.intermediate):
        raise RuntimeError(f"unexpected down shape {tuple(w_down.shape)}")

    layer0_state = {}
    for name, value in raw_state.items():
        marker = ".layers.0."
        if marker in name and ".experts." not in name:
            layer0_state[name.split(marker, 1)[1]] = value.cpu()

    counts = torch.bincount(topi.reshape(-1).long(), minlength=spec.experts)
    trace = {
        "metadata": {
            "schema": 1,
            "trace_key": canonical_trace_key(spec.key),
            "model_id": spec.model_id,
            "seq_len": seq_len,
            "dataset_id": DATASET_ID,
            "dataset_config": DATASET_CONFIG,
            "dataset_revision": DATASET_REVISION,
            "token_sha256": token_hash,
            "source_rows": rows,
            "checkpoint_shards": shards,
            "geometry": {
                "hidden": spec.hidden,
                "intermediate": spec.intermediate,
                "experts": spec.experts,
                "topk": spec.topk,
                "activation": spec.activation,
            },
            "routing": {
                "min": int(counts.min()),
                "max": int(counts.max()),
                "mean": float(counts.float().mean()),
                "empty": int((counts == 0).sum()),
            },
        },
        "input_ids": input_ids.cpu(),
        "block_input": block_input.cpu(),
        "expert_input": x.cpu(),
        "topk_index": topi.cpu(),
        "topk_weight": topv.cpu(),
        "gate_up_weight": w_gate_up.cpu(),
        "down_weight": w_down.cpu(),
        "layer0_nonexpert": layer0_state,
    }
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(trace, out)
    return trace["metadata"]


__all__ = ["capture_trace"]
