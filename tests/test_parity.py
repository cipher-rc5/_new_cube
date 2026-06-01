"""MLX-vs-torch parity gate (migration plan 7.3).

This is the correctness contract for the optional Apple Silicon MLX backend:
for a fixed prompt with DETERMINISTIC (argmax, top_p=None) decoding, the MLX
GPT decode must reproduce the torch reference exactly, and the resulting mesh
must match within float tolerance.

Requirements (all genuinely needed, so this skips in CI):
* mlx           -> pytest.importorskip("mlx")
* torch + mps   -> the MlxEngine composes a torch Engine on device "mps"
* model weights -> via env CUBE_TEST_WEIGHTS_DIR (see conftest.weights_dir_or_skip)

The test is written to be ready-to-run on a real Apple Silicon machine: set
CUBE_TEST_WEIGHTS_DIR to a directory containing shape_gpt.safetensors and
shape_tokenizer.safetensors, install the mlx extra (uv sync --extra mlx), and
run ``pytest tests/test_parity.py``.

Known divergence points (see the TODO(parity) notes in
cube3d/inference/mlx_engine.py):
  1. top_p sampling: the torch top_p path hardcodes top_p=0.9 inside
     top_p_filtering regardless of the value passed, and uses torch RNG. The MLX
     ``_sample`` replicates the 0.9 quirk but uses numpy RNG, so top_p runs are
     NOT expected to match bit-for-bit. Parity is asserted only for argmax
     (top_p=None) decoding.
  2. KV cache: the MLX ``_dual_block`` recomputes the full prefix each step (no
     incremental KV cache), which is functionally equivalent to torch's
     use_kv_cache=False branch. CFG gamma is applied per-step identically. A
     future MLX KV cache must keep this parity test green.
"""

import os

import numpy as np
import pytest

# Hard requirement: the MLX backend itself.
mx = pytest.importorskip("mlx")
torch = pytest.importorskip("torch")

from tests.conftest import weights_dir_or_skip

CONFIG_REL = os.path.join("cube3d", "configs", "open_model_v0.5.yaml")
PROMPT = "A pair of noise-canceling headphones"


def _require_mps():
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is not available; MlxEngine requires Apple Silicon / MPS")


def _ckpt_paths(weights_dir: str):
    gpt = os.path.join(weights_dir, "shape_gpt.safetensors")
    shape = os.path.join(weights_dir, "shape_tokenizer.safetensors")
    for p in (gpt, shape):
        if not os.path.exists(p):
            pytest.skip(f"Missing checkpoint {p}; cannot run parity test")
    return gpt, shape


def test_mlx_run_gpt_argmax_matches_torch(repo_root, monkeypatch):
    """Deterministic (argmax) decode: MLX output_ids must EXACTLY match torch.

    This is the core parity assertion. With top_p=None both engines select
    argmax over the same CFG-mixed logits, so the integer token IDs must be
    identical element-for-element.

    The MLX decode runs in float32; pin the torch reference to float32 too
    (MPS autocast defaults to bfloat16) so the two are compared like-for-like.
    Without this, accumulated bf16 rounding flips an argmax mid-sequence and the
    decodes cascade apart.
    """
    monkeypatch.setenv("CUBE_MPS_AUTOCAST_DTYPE", "float32")
    weights_dir = weights_dir_or_skip()
    _require_mps()
    gpt_ckpt, shape_ckpt = _ckpt_paths(weights_dir)
    config_path = os.path.join(repo_root, CONFIG_REL)

    from cube3d.inference.engine import Engine
    from cube3d.inference.mlx_engine import MlxEngine

    torch_engine = Engine(
        config_path, gpt_ckpt, shape_ckpt, device=torch.device("mps")
    )
    mlx_engine = MlxEngine(config_path, gpt_ckpt, shape_ckpt)

    # Deterministic decode: top_p=None -> argmax on both sides.
    # The torch reference uses use_kv_cache=False: the MLX engine recomputes the
    # full prefix every step (no incremental KV cache), so its math mirrors
    # torch's no-cache branch exactly. torch's KV-cache path accumulates its own
    # fp differences and diverges from BOTH the no-cache path and MLX around
    # token ~12, so it is not a valid bit-exact reference here.
    torch_ids = torch_engine.run_gpt(
        [PROMPT], use_kv_cache=False, guidance_scale=3.0, top_p=None
    )
    mlx_ids = mlx_engine.run_gpt(
        [PROMPT], use_kv_cache=False, guidance_scale=3.0, top_p=None
    )

    torch_arr = torch_ids.detach().cpu().numpy()
    mlx_arr = mlx_ids.detach().cpu().numpy()

    assert torch_arr.shape == mlx_arr.shape, (
        f"output_ids shape mismatch: torch {torch_arr.shape} vs mlx {mlx_arr.shape}"
    )
    # EXACT match required for argmax decoding (handoff boundary is integer IDs).
    assert np.array_equal(torch_arr, mlx_arr), (
        "MLX argmax output_ids diverged from torch reference"
    )


def test_mlx_mesh_matches_torch_within_tolerance(repo_root, monkeypatch):
    """End-to-end mesh parity: vertex/face counts and bbox match within tol.

    Because both engines share the SAME torch shape-decode (MlxEngine delegates
    run_shape_decode to its internal torch Engine) and, per the test above,
    produce identical token IDs under argmax, the meshes must agree up to
    floating-point noise from the decode itself.

    As above, pin the torch reference to float32 to match the MLX decode.
    """
    monkeypatch.setenv("CUBE_MPS_AUTOCAST_DTYPE", "float32")
    weights_dir = weights_dir_or_skip()
    _require_mps()
    gpt_ckpt, shape_ckpt = _ckpt_paths(weights_dir)
    config_path = os.path.join(repo_root, CONFIG_REL)

    from cube3d.inference.engine import Engine
    from cube3d.inference.mlx_engine import MlxEngine

    torch_engine = Engine(
        config_path, gpt_ckpt, shape_ckpt, device=torch.device("mps")
    )
    mlx_engine = MlxEngine(config_path, gpt_ckpt, shape_ckpt)

    # use_kv_cache=False on the torch reference so its decode math matches the
    # MLX full-recompute path (see the argmax test above for why).
    torch_mesh = torch_engine.t2s(
        [PROMPT], use_kv_cache=False, resolution_base=8.0, top_p=None
    )
    mlx_mesh = mlx_engine.t2s(
        [PROMPT], use_kv_cache=False, resolution_base=8.0, top_p=None
    )

    tv, tf = torch_mesh[0][0], torch_mesh[0][1]
    mv, mf = mlx_mesh[0][0], mlx_mesh[0][1]

    tv, tf = np.asarray(tv), np.asarray(tf)
    mv, mf = np.asarray(mv), np.asarray(mf)

    # Vertex and face counts must match exactly (same token IDs -> same surface).
    assert tv.shape == mv.shape, f"vertex count mismatch: {tv.shape} vs {mv.shape}"
    assert tf.shape == mf.shape, f"face count mismatch: {tf.shape} vs {mf.shape}"

    # Bounding boxes must agree within float tolerance.
    np.testing.assert_allclose(tv.min(axis=0), mv.min(axis=0), atol=1e-3, rtol=1e-3)
    np.testing.assert_allclose(tv.max(axis=0), mv.max(axis=0), atol=1e-3, rtol=1e-3)
