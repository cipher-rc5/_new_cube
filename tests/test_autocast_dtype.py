"""Tests for cube3d.inference.utils.select_autocast_dtype.

select_autocast_dtype is the targeted fix for MPS bf16 op gaps: it picks the
autocast / KV-cache dtype per device and lets MPS users override via the
CUBE_MPS_AUTOCAST_DTYPE env var. Contract:

* cuda  -> torch.bfloat16 (always)
* mps   -> CUBE_MPS_AUTOCAST_DTYPE: default/unset -> bfloat16,
           "float16" -> float16, "float32" -> float32,
           unknown   -> bfloat16 (with a logged warning)
* cpu / anything else -> torch.float32
"""

import pytest

torch = pytest.importorskip("torch")

from cube3d.inference.utils import select_autocast_dtype

ENV_VAR = "CUBE_MPS_AUTOCAST_DTYPE"


def test_cuda_device_object_returns_bfloat16():
    assert select_autocast_dtype(torch.device("cuda")) == torch.bfloat16


def test_cuda_string_returns_bfloat16():
    assert select_autocast_dtype("cuda") == torch.bfloat16


def test_cpu_device_object_returns_float32():
    assert select_autocast_dtype(torch.device("cpu")) == torch.float32


def test_cpu_string_returns_float32():
    assert select_autocast_dtype("cpu") == torch.float32


def test_unknown_device_returns_float32():
    # Anything that is not cuda/mps defaults to float32.
    assert select_autocast_dtype("xpu") == torch.float32


def test_mps_default_unset_is_bfloat16(monkeypatch):
    monkeypatch.delenv(ENV_VAR, raising=False)
    assert select_autocast_dtype(torch.device("mps")) == torch.bfloat16


def test_mps_string_default_unset_is_bfloat16(monkeypatch):
    monkeypatch.delenv(ENV_VAR, raising=False)
    assert select_autocast_dtype("mps") == torch.bfloat16


def test_mps_env_float16(monkeypatch):
    monkeypatch.setenv(ENV_VAR, "float16")
    assert select_autocast_dtype(torch.device("mps")) == torch.float16


def test_mps_env_float32(monkeypatch):
    monkeypatch.setenv(ENV_VAR, "float32")
    assert select_autocast_dtype(torch.device("mps")) == torch.float32


def test_mps_env_bfloat16_explicit(monkeypatch):
    monkeypatch.setenv(ENV_VAR, "bfloat16")
    assert select_autocast_dtype(torch.device("mps")) == torch.bfloat16


def test_mps_env_is_case_insensitive(monkeypatch):
    # The implementation lowercases the env value.
    monkeypatch.setenv(ENV_VAR, "Float16")
    assert select_autocast_dtype(torch.device("mps")) == torch.float16


def test_mps_env_garbage_falls_back_to_bfloat16_with_warning(monkeypatch, caplog):
    monkeypatch.setenv(ENV_VAR, "not-a-real-dtype")
    import logging

    with caplog.at_level(logging.WARNING):
        result = select_autocast_dtype(torch.device("mps"))
    assert result == torch.bfloat16
    # A warning should have been logged about the unknown value.
    assert any(
        "CUBE_MPS_AUTOCAST_DTYPE" in rec.getMessage() for rec in caplog.records
    )
