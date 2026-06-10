"""Torch model loader for raw model packages."""

import os
from typing import Optional

import torch


def _instantiate(spec: dict):
    """Instantiate a simple Hydra-style ``{"_target_": ...}`` spec."""
    if "_target_" not in spec:
        return spec
    import importlib

    target = spec["_target_"]
    module_name, class_name = target.rsplit(".", 1)
    model_type = getattr(importlib.import_module(module_name), class_name)
    kwargs = {
        key: _instantiate(value) if isinstance(value, dict) else value
        for key, value in spec.items()
        if key != "_target_"
    }
    return model_type(**kwargs)


def load_torch_model(
    model_bin_path: str,
    model_class: type,
    spec: Optional[dict] = None,
) -> torch.nn.Module:
    """Load a torch model from a state-dict raw package.

    Args:
        model_bin_path: Path to the package ``model`` directory.
        model_class: ``torch.nn.Module`` class to instantiate.
        spec: Optional constructor kwargs or Hydra-style target spec.

    Returns:
        Loaded model in eval mode.
    """
    pt_files = [name for name in os.listdir(model_bin_path) if name.endswith(".pt")]
    if not pt_files:
        raise FileNotFoundError(f"No .pt file found in {model_bin_path}")

    model_file_path = os.path.join(model_bin_path, pt_files[0])
    try:
        state_dict = torch.load(model_file_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load state_dict from {model_file_path}: {exc}"
        ) from exc
    if not isinstance(state_dict, dict):
        raise TypeError(f"Expected state_dict format, got {type(state_dict)}")

    if spec and "_target_" in spec:
        model = _instantiate(spec)
    elif spec:
        model = model_class(**spec)
    else:
        model = model_class()

    model.load_state_dict(state_dict)
    model.eval()
    return model
