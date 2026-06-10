"""Utilities for preparing ONNX deployable artifacts."""

from __future__ import annotations

import os
import shutil
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from michelangelo._internal.utils.reflection_utils import get_module_attr

from .torchscript_conversion import is_state_dict

if TYPE_CHECKING:
    from michelangelo.lib.model_manager.schema import ModelSchema

OPSET_VERSION = 14


def _load_torch_model(
    source_model_path: str,
    model_class: str | None,
    hyperparameters: dict | None,
) -> torch.nn.Module:
    """Load a torch module from a local artifact."""
    loaded_model = torch.load(source_model_path, map_location="cpu", weights_only=False)
    if is_state_dict(loaded_model):
        if not model_class:
            raise ValueError("model_class is required when exporting a state_dict")
        model_type = get_module_attr(model_class)
        model = model_type(**(hyperparameters or {}))
        model.load_state_dict(loaded_model)
    else:
        model = loaded_model

    if not isinstance(model, torch.nn.Module):
        raise TypeError(f"Artifact is not a torch.nn.Module: {source_model_path}")

    model.eval()
    return model


def _prepare_sample_inputs(
    input_names: list[str],
    sample_data: dict[str, Any],
) -> tuple[torch.Tensor, ...]:
    """Convert sample data into tracing inputs in schema order."""
    sample_inputs: list[torch.Tensor] = []
    for name in input_names:
        if name not in sample_data:
            raise ValueError(f"sample_data missing required input {name!r}")
        value = sample_data[name]
        if isinstance(value, torch.Tensor):
            tensor = value
        elif isinstance(value, np.ndarray):
            tensor = torch.from_numpy(value)
        else:
            raise TypeError(
                f"Sample data for {name!r} must be torch.Tensor or numpy.ndarray"
            )
        if tensor.ndim == len(tensor.shape):
            tensor = tensor.unsqueeze(0)
        sample_inputs.append(tensor)
    return tuple(sample_inputs)


def convert_to_onnx(
    source_model_path: str,
    dest_onnx_path: str,
    model_schema: ModelSchema,
    sample_data: dict[str, Any] | None = None,
    model_class: str | None = None,
    hyperparameters: dict | None = None,
    enable_dynamic_batching: bool = True,
) -> None:
    """Copy or export a torch artifact into ONNX format.

    Args:
        source_model_path: Local source artifact path.
        dest_onnx_path: Destination ``model.onnx`` path.
        model_schema: Model schema used for input and output names.
        sample_data: Sample input batch used for torch-to-ONNX tracing.
        model_class: Import path for state-dict artifacts.
        hyperparameters: Constructor keyword arguments for ``model_class``.
        enable_dynamic_batching: Whether to mark batch dimension as dynamic.
    """
    if source_model_path.endswith(".onnx"):
        shutil.copy2(source_model_path, dest_onnx_path)
        return

    if not sample_data:
        raise ValueError("sample_data is required to export a torch artifact to ONNX")

    model = _load_torch_model(source_model_path, model_class, hyperparameters)
    input_names = [item.name for item in model_schema.input_schema]
    output_names = [item.name for item in model_schema.output_schema]
    sample_inputs = _prepare_sample_inputs(input_names, sample_data)
    dynamic_axes = (
        {name: {0: "batch"} for name in [*input_names, *output_names]}
        if enable_dynamic_batching
        else None
    )
    os.makedirs(os.path.dirname(dest_onnx_path), exist_ok=True)
    torch.onnx.export(
        model,
        sample_inputs,
        dest_onnx_path,
        input_names=input_names,
        output_names=output_names,
        opset_version=OPSET_VERSION,
        dynamic_axes=dynamic_axes,
    )
