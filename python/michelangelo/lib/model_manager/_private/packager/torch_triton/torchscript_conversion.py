"""Utilities for preparing torchscript deployable artifacts."""

import os
from typing import Optional

import torch

from michelangelo._internal.utils.reflection_utils import get_module_attr


def is_state_dict(value: object) -> bool:
    """Return whether a loaded torch object looks like a state dict."""
    return isinstance(value, dict) and all(isinstance(key, str) for key in value)


def convert_to_torchscript(
    model_path: str,
    model_class: Optional[str] = None,
    hyperparameters: Optional[dict] = None,
) -> None:
    """Convert a torch artifact in place to torchscript when needed.

    Args:
        model_path: Local path to a ``.pt`` or ``.pth`` artifact.
        model_class: Import path for the model class when ``model_path``
            contains a state dict.
        hyperparameters: Constructor keyword arguments for ``model_class``.

    Raises:
        FileNotFoundError: If ``model_path`` does not exist.
        TypeError: If the artifact cannot be converted to torchscript.
        ValueError: If a state dict is provided without ``model_class``.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"File does not exist: {model_path}")

    try:
        torch.jit.load(model_path, map_location="cpu")
        return
    except Exception:
        pass

    try:
        loaded_model = torch.load(model_path, map_location="cpu", weights_only=False)
        if is_state_dict(loaded_model):
            if not model_class:
                raise ValueError("model_class is required for state_dict artifacts")
            model_type = get_module_attr(model_class)
            model = model_type(**(hyperparameters or {}))
            model.load_state_dict(loaded_model)
        else:
            model = loaded_model

        if not isinstance(model, torch.nn.Module):
            raise TypeError("Artifact does not contain a torch.nn.Module")

        model.eval()
        torch.jit.save(torch.jit.script(model), model_path)
    except Exception as exc:
        raise TypeError(
            f"File does not contain a torchscript-convertible model: {model_path}"
        ) from exc
