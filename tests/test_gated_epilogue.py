import pytest

from nvfp4moe.kernels.epilogue import (
    GatedBackwardEpilogue,
    GatedEpilogue,
    gated_output_n,
    gated_postact_fragment,
    gated_sf_u32_word_count,
    resolve_gemm_epilogue,
    validate_gated_activation,
    validate_gated_tile_n,
)


def test_epilogue_policies_validate_activation():
    assert GatedEpilogue("swiglu").activation == "swiglu"
    assert GatedEpilogue("swiglu", save_preact=True).save_preact
    assert GatedBackwardEpilogue("geglu").activation == "geglu"
    with pytest.raises(ValueError, match="activation must be one of"):
        GatedEpilogue("gelu")


def test_clamped_swiglu_policy():
    policy = GatedEpilogue("swiglu", clamp_limit=10)
    assert policy.clamp_limit == 10.0
    with pytest.raises(ValueError, match="finite and positive"):
        GatedEpilogue("swiglu", clamp_limit=0)
    with pytest.raises(ValueError, match="only for swiglu"):
        GatedEpilogue("geglu", clamp_limit=10)
    with pytest.raises(ValueError, match="saved preactivation"):
        GatedEpilogue("swiglu", save_preact=True, clamp_limit=10)


def test_epilogue_policy_resolves_compile_mode():
    assert resolve_gemm_epilogue(GatedEpilogue("geglu"), None, None) == ("geglu", None)
    assert resolve_gemm_epilogue(GatedBackwardEpilogue("reglu"), None, None) == (None, "reglu")
    with pytest.raises(ValueError, match="cannot be combined"):
        resolve_gemm_epilogue(GatedEpilogue(), "swiglu", None)


@pytest.mark.parametrize(("accumulator_n", "output_n"), [(2, 1), (256, 128), (1536, 768)])
def test_gated_output_n(accumulator_n, output_n):
    assert gated_output_n(accumulator_n) == output_n


@pytest.mark.parametrize("accumulator_n", [-2, 0, 1, 255])
def test_gated_output_n_rejects_invalid_extent(accumulator_n):
    with pytest.raises(ValueError, match="positive even"):
        gated_output_n(accumulator_n)


@pytest.mark.parametrize("tile_n", [128, 256, 384, 512])
def test_gated_tile_n_alignment(tile_n):
    assert validate_gated_tile_n(tile_n) == tile_n


@pytest.mark.parametrize("tile_n", [-128, 0, 32, 64, 192, 257])
def test_gated_tile_n_rejects_partial_sf_atoms(tile_n):
    with pytest.raises(ValueError, match="multiple of 128"):
        validate_gated_tile_n(tile_n)


@pytest.mark.parametrize("activation", ["swiglu", "geglu", "reglu"])
def test_gated_activation_names(activation):
    assert validate_gated_activation(activation) == activation


def test_gated_activation_rejects_unknown_name():
    with pytest.raises(ValueError, match="activation must be one of"):
        validate_gated_activation("gelu")


def test_gated_fragment_helper_is_jitted():
    assert callable(gated_postact_fragment)


@pytest.mark.parametrize(
    ("tile_n", "word_count"),
    [(128, 1), (256, 2), (384, 3)],
)
def test_gated_sf_u32_word_count(tile_n, word_count):
    assert gated_sf_u32_word_count(128, tile_n, 128, 64) == word_count


@pytest.mark.parametrize(
    ("tile_m", "tile_n", "epi_m", "epi_n"),
    [
        (128, 64, 128, 64),
        (128, 192, 128, 64),
        (128, 256, 64, 64),
        (128, 256, 128, 32),
        (0, 256, 128, 64),
    ],
)
def test_gated_sf_u32_word_count_falls_back(tile_m, tile_n, epi_m, epi_n):
    assert gated_sf_u32_word_count(tile_m, tile_n, epi_m, epi_n) == 0
