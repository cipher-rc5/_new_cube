"""Shared pytest fixtures and skip helpers for the cube migration test suite.

These tests are designed to run in CI on any machine, including ones WITHOUT
torch (CUDA or otherwise), warp-lang, or mlx installed. Tests that genuinely
need a heavy/optional dependency use the helpers here (or
``pytest.importorskip``) so that the missing dependency results in a SKIP, not
an ERROR.
"""

import os

import pytest

# Repo root: tests/ lives directly under the project root.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def have_module(name: str) -> bool:
    """Return True if ``name`` can be imported, without raising on failure."""
    import importlib.util

    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


@pytest.fixture(scope="session")
def repo_root() -> str:
    """Absolute path to the repository root (parent of tests/)."""
    return REPO_ROOT


def weights_dir_or_skip() -> str:
    """Return CUBE_TEST_WEIGHTS_DIR or skip the calling test with a clear message.

    Parity / end-to-end tests need real model weights. CI does not ship them, so
    those tests skip unless the maintainer points this env var at a directory
    holding ``shape_gpt.safetensors`` and ``shape_tokenizer.safetensors``.
    """
    weights = os.environ.get("CUBE_TEST_WEIGHTS_DIR")
    if not weights:
        pytest.skip(
            "CUBE_TEST_WEIGHTS_DIR is not set; skipping weight-dependent test. "
            "Set it to a directory containing shape_gpt.safetensors and "
            "shape_tokenizer.safetensors to run this on a real machine."
        )
    return weights
