import logging
import os
from typing import Any, Optional, Tuple

import torch
from omegaconf import DictConfig, OmegaConf
from safetensors.torch import load_model

BOUNDING_BOX_MAX_SIZE = 1.925


def normalize_bbox(bounding_box_xyz: Tuple[float]):
    max_l = max(bounding_box_xyz)
    return [BOUNDING_BOX_MAX_SIZE * elem / max_l for elem in bounding_box_xyz]


def load_config(cfg_path: str) -> Any:
    """
    Load and resolve a configuration file.
    Args:
        cfg_path (str): The path to the configuration file.
    Returns:
        Any: The loaded and resolved configuration object.
    Raises:
        AssertionError: If the loaded configuration is not an instance of DictConfig.
    """

    cfg = OmegaConf.load(cfg_path)
    OmegaConf.resolve(cfg)
    assert isinstance(cfg, DictConfig)
    return cfg


def parse_structured(cfg_type: Any, cfg: DictConfig) -> Any:
    """
    Parses a configuration dictionary into a structured configuration object.
    Args:
        cfg_type (Any): The type of the structured configuration object.
        cfg (DictConfig): The configuration dictionary to be parsed.
    Returns:
        Any: The structured configuration object created from the dictionary.
    """

    cfg_dict: dict[str, Any] = {str(k): v for k, v in cfg.items()}
    scfg = OmegaConf.structured(cfg_type(**cfg_dict))
    return scfg


def load_model_weights(model: torch.nn.Module, ckpt_path: str) -> None:
    """
    Load a safetensors checkpoint into a PyTorch model.
    The model is updated in place.

    Args:
        model: PyTorch model to load weights into
        ckpt_path: Path to the safetensors checkpoint file

    Returns:
        None
    """
    assert ckpt_path.endswith(
        ".safetensors"
    ), f"Checkpoint path '{ckpt_path}' is not a safetensors file"

    load_model(model, ckpt_path)


def select_device() -> Any:
    """
    Selects the appropriate PyTorch device for tensor allocation.

    Returns:
        Any: The `torch.device` object.
    """
    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )


def select_autocast_dtype(device: Any) -> torch.dtype:
    """
    Selects the autocast (and KV-cache) dtype appropriate for the given device.

    - cuda: always torch.bfloat16.
    - mps: read env var CUBE_MPS_AUTOCAST_DTYPE (default "bfloat16"; allowed
      "float16", "float32"). An unknown value falls back to bfloat16 with a
      logged warning.
    - cpu (and anything else): torch.float32.

    Args:
        device (Any): A torch.device or device-string identifying the target.

    Returns:
        torch.dtype: The dtype to use for autocast and KV-cache allocation.
    """
    device_type = device.type if isinstance(device, torch.device) else str(device)

    if "cuda" in device_type:
        return torch.bfloat16

    if "mps" in device_type:
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        requested = os.environ.get("CUBE_MPS_AUTOCAST_DTYPE", "bfloat16").lower()
        if requested not in dtype_map:
            logging.warning(
                "Unknown CUBE_MPS_AUTOCAST_DTYPE value %r; falling back to bfloat16. "
                "Allowed values: bfloat16, float16, float32.",
                requested,
            )
            return torch.bfloat16
        return dtype_map[requested]

    return torch.float32
