"""Import-guard tests for optional native dependencies.

The migration made two heavy/native dependencies optional and import-guarded:

* ``warp-lang`` (CUDA-only marching cubes) in
  ``cube3d.model.autoencoder.grid`` via ``grid.WARP_AVAILABLE``.
* ``mlx`` (Apple Silicon GPT decode backend) in
  ``cube3d.inference.mlx_engine`` via ``mlx_engine.MLX_AVAILABLE``.

Both modules must import cleanly even when the optional dependency is absent.
These modules DO import torch, so we ``importorskip('torch')`` up front: on a
machine without torch the whole file skips rather than erroring.
"""

import pytest

# Both target modules import torch at module scope. If torch is not installed
# (as in the lightweight CI sandbox), skip this file cleanly.
torch = pytest.importorskip("torch")

from tests.conftest import have_module


def test_grid_imports_without_warp():
    """grid imports cleanly and exposes a boolean WARP_AVAILABLE flag.

    We assert the module imports regardless of whether warp-lang is installed.
    If warp-lang happens to be absent, WARP_AVAILABLE must be False; if present,
    it must be True. Either way it must be a bool.
    """
    from cube3d.model.autoencoder import grid

    assert hasattr(grid, "WARP_AVAILABLE")
    assert isinstance(grid.WARP_AVAILABLE, bool)
    # The flag must agree with the actual availability of the dependency.
    assert grid.WARP_AVAILABLE == have_module("warp")


def test_mlx_engine_imports_without_mlx():
    """mlx_engine imports cleanly and exposes a boolean MLX_AVAILABLE flag.

    The module must be importable on any platform even when mlx is not
    installed; the MLX-specific code paths are guarded behind MLX_AVAILABLE.
    """
    from cube3d.inference import mlx_engine

    assert hasattr(mlx_engine, "MLX_AVAILABLE")
    assert isinstance(mlx_engine.MLX_AVAILABLE, bool)
    assert mlx_engine.MLX_AVAILABLE == have_module("mlx")
    # MlxEngine is always defined as a class regardless of mlx availability;
    # only constructing it requires mlx.
    assert hasattr(mlx_engine, "MlxEngine")


def test_mlx_engine_construction_raises_without_mlx():
    """Constructing MlxEngine without mlx raises a clear RuntimeError.

    Only meaningful when mlx is genuinely absent; otherwise skip (constructing a
    real engine would require model weights).
    """
    from cube3d.inference import mlx_engine

    if mlx_engine.MLX_AVAILABLE:
        pytest.skip("mlx is installed; construction would require model weights")

    with pytest.raises(RuntimeError):
        mlx_engine.MlxEngine("config.yaml", "gpt.safetensors", "shape.safetensors")


def test_marching_cubes_raises_when_warp_unavailable(monkeypatch):
    """marching_cubes_with_warp raises RuntimeError when WARP_AVAILABLE is False.

    We force the flag off via monkeypatch so the test is deterministic even on a
    machine that does have warp installed, and call with dummy args. The
    RuntimeError must be raised before any warp API is touched.
    """
    from cube3d.model.autoencoder import grid

    monkeypatch.setattr(grid, "WARP_AVAILABLE", False)

    dummy = torch.zeros((2, 2, 2))
    with pytest.raises(RuntimeError):
        grid.marching_cubes_with_warp(dummy, level=0.0, device="cuda")
