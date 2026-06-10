"""Torch raw model loader."""

import os
import sys
from typing import Optional

import yaml

from michelangelo.lib.model_manager._private.serde.loader.torch_model_loader import (
    load_torch_model,
)
from michelangelo.lib.model_manager._private.serde.model.custom_raw_model import (
    create_alternative_defs,
)


def _import_model_class(defs_path: str, model_class: str) -> type:
    """Import a model class from the environment or packaged definitions."""
    module_name, _, class_name = model_class.rpartition(".")
    if not module_name or not class_name:
        raise ValueError(
            f"Invalid model class definition {model_class}. "
            "Please specify the full import path to the model class."
        )

    try:
        module = __import__(module_name, fromlist=[class_name])
    except (ImportError, ModuleNotFoundError):
        sys.path.append(os.path.abspath(defs_path))
        try:
            module = __import__(module_name, fromlist=[class_name])
        except (ImportError, ModuleNotFoundError):
            new_defs_path, wrapper_name = create_alternative_defs(defs_path)
            sys.path.append(new_defs_path)
            module = __import__(
                f"{wrapper_name}.defs.{module_name}",
                fromlist=[class_name],
            )

    try:
        return getattr(module, class_name)
    except AttributeError as exc:
        raise AttributeError(
            f"Class {class_name} not found in module {module_name}."
        ) from exc


def load_torch_raw_model(
    model_path: str,
    submodule: Optional[str] = None,
):
    """Load a torch raw model package.

    Args:
        model_path: Path to the raw model package.
        submodule: Reserved for compatibility with existing callers. Submodule
            extraction is not supported in OSS yet.

    Returns:
        Loaded ``torch.nn.Module`` instance.
    """
    if submodule is not None:
        raise NotImplementedError("submodule loading is not supported in OSS yet")

    model_bin_path = os.path.join(model_path, "model")
    defs_path = os.path.join(model_path, "defs")
    model_class_path = os.path.join(defs_path, "model_class.txt")
    if not os.path.exists(model_class_path):
        raise ValueError("Missing defs/model_class.txt in the model package.")

    with open(model_class_path) as f:
        model_class = f.read().strip()
    if not model_class:
        raise ValueError("defs/model_class.txt is empty in the model package.")

    spec = None
    hyperparameters_path = os.path.join(model_path, "metadata", "hyperparameters.yaml")
    if os.path.exists(hyperparameters_path):
        with open(hyperparameters_path) as f:
            spec = yaml.safe_load(f) or {}
        if spec and "_target_" in spec:
            model_class = spec["_target_"]

    model_type = _import_model_class(defs_path, model_class)
    return load_torch_model(model_bin_path, model_type, spec=spec)
