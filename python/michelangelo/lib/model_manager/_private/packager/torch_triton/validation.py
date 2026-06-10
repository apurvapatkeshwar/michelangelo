"""Validation utilities for torch Triton packages."""

from __future__ import annotations

import inspect
import os
from typing import TYPE_CHECKING, Any

import torch
from numpy import ndarray

from michelangelo._internal.utils.reflection_utils import get_module_attr
from michelangelo.lib.model_manager._private.utils.data_utils import (
    validate_output_data,
    validate_output_data_with_model_schema,
)
from michelangelo.lib.model_manager.serde.model import load_raw_model

if TYPE_CHECKING:
    from michelangelo.lib.model_manager.schema import ModelSchema, ModelSchemaItem


def _validate_file_basics(
    file_path: str,
    allowed_extensions: tuple[str, ...] = (".pt", ".pth"),
) -> Exception | None:
    """Validate common torch artifact file properties."""
    if not os.path.exists(file_path):
        return FileNotFoundError(f"Torch file not found: {file_path}")
    if os.path.isdir(file_path):
        return ValueError(f"Path is not a file: {file_path}")
    if not file_path.endswith(allowed_extensions):
        return ValueError(f"File must have {allowed_extensions} extension: {file_path}")
    if os.path.getsize(file_path) == 0:
        return ValueError(f"File is empty: {file_path}")
    return None


def validate_state_dict_file(model_path: str) -> tuple[bool, Exception | None]:
    """Validate that a torch artifact contains a state dict."""
    try:
        if error := _validate_file_basics(model_path):
            return False, error
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        if isinstance(state_dict, dict):
            return True, None
        return False, ValueError(f"File does not contain a state dict: {model_path}")
    except Exception as exc:
        return False, RuntimeError(f"Cannot load file as state dict: {exc}")


def validate_torchscript_file(model_path: str) -> tuple[bool, Exception | None]:
    """Validate that a torch artifact is loadable as torchscript."""
    try:
        if error := _validate_file_basics(model_path):
            return False, error
        torch.jit.load(model_path, map_location="cpu")
    except Exception as exc:
        return False, RuntimeError(f"File is not valid torchscript: {exc}")
    return True, None


def validate_pytorch_model_file(model_path: str) -> tuple[bool, Exception | None]:
    """Validate that a torch artifact contains a ``torch.nn.Module``."""
    try:
        if error := _validate_file_basics(model_path):
            return False, error
        model = torch.load(model_path, map_location="cpu", weights_only=False)
        if not isinstance(model, torch.nn.Module) and not isinstance(model, dict):
            return False, ValueError("File must contain a torch module or state dict")
    except Exception as exc:
        return False, RuntimeError(f"Cannot load as a torch model: {exc}")
    return True, None


def validate_raw_model_file(model_path: str) -> tuple[bool, Exception | None]:
    """Validate a torch file for raw-model packaging."""
    is_state_dict, _ = validate_state_dict_file(model_path)
    if is_state_dict:
        return True, None
    return validate_pytorch_model_file(model_path)


def validate_model_class(model_class: str) -> tuple[bool, Exception | None]:
    """Validate that ``model_class`` is an importable ``torch.nn.Module`` type."""
    try:
        model_type = get_module_attr(model_class)
    except (AttributeError, ImportError, ValueError) as exc:
        return False, exc
    if not inspect.isclass(model_type) or not issubclass(model_type, torch.nn.Module):
        return False, TypeError(
            f"Model class {model_class} must subclass torch.nn.Module"
        )
    return True, None


def _add_batch_dimension(value: Any) -> Any:
    """Add a batch dimension to ndarray/tensor sample values."""
    if isinstance(value, ndarray):
        return torch.from_numpy(value).unsqueeze(0)
    if isinstance(value, torch.Tensor):
        return value.unsqueeze(0)
    raise TypeError(
        f"Torch sample values must be numpy.ndarray or torch.Tensor, got {type(value)}"
    )


def _remove_batch_dimension(
    output: torch.Tensor,
    output_schema: ModelSchemaItem | None,
) -> ndarray:
    """Convert batched tensor output to an ndarray matching schema rank."""
    output_array = output.squeeze(0).detach().cpu().numpy()
    expected_ndim = len(output_schema.shape) if output_schema else 1
    if output_array.ndim == 0 and expected_ndim > 0:
        output_array = output_array.reshape(-1)
    return output_array


def _normalize_output(output: Any, model_schema: ModelSchema) -> dict[str, Any]:
    """Normalize torch outputs into a schema-keyed dictionary."""
    if isinstance(output, dict):
        return {
            key: value.detach().cpu().numpy()
            if isinstance(value, torch.Tensor)
            else value
            for key, value in output.items()
        }
    if isinstance(output, torch.Tensor):
        schema_item = (
            model_schema.output_schema[0] if model_schema.output_schema else None
        )
        output_name = schema_item.name if schema_item else "output"
        return {output_name: _remove_batch_dimension(output, schema_item)}
    if isinstance(output, (tuple, list)):
        result = {}
        for index, value in enumerate(output):
            schema_item = (
                model_schema.output_schema[index]
                if index < len(model_schema.output_schema)
                else None
            )
            output_name = schema_item.name if schema_item else f"output_{index}"
            result[output_name] = (
                _remove_batch_dimension(value, schema_item)
                if isinstance(value, torch.Tensor)
                else value
            )
        return result
    raise TypeError(f"Unsupported torch model output type: {type(output)}")


def _invoke_model(model: torch.nn.Module, batch_dict: dict[str, Any]) -> Any:
    """Invoke a torch model using keyword inputs or a single dict input."""
    try:
        return model(**batch_dict)
    except TypeError:
        return model(batch_dict)


def validate_raw_model_package(
    package_path: str,
    sample_data: list[dict[str, ndarray]] | dict[str, ndarray] | None = None,
    model_schema: ModelSchema | None = None,
) -> None:
    """Validate a generated torch raw model package.

    Args:
        package_path: Path to the generated package directory.
        sample_data: Optional sample data used to run a forward pass.
        model_schema: Optional schema used to validate forward-pass output.
    """
    model_dir = os.path.join(package_path, "model")
    defs_path = os.path.join(package_path, "defs")
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Required directory missing: {model_dir}")
    if not os.path.exists(os.path.join(defs_path, "model_class.txt")):
        raise FileNotFoundError("Missing defs/model_class.txt in torch package")

    pt_files = [name for name in os.listdir(model_dir) if name.endswith(".pt")]
    if not pt_files:
        raise FileNotFoundError(f"No .pt file found in {model_dir}")
    is_valid, error = validate_raw_model_file(os.path.join(model_dir, pt_files[0]))
    if not is_valid:
        raise RuntimeError(f"Invalid raw model file: {error}") from error

    if sample_data and model_schema:
        model = load_raw_model(package_path)
        model.eval()
        data = sample_data[0] if isinstance(sample_data, list) else sample_data
        batch_dict = {key: _add_batch_dimension(value) for key, value in data.items()}
        with torch.no_grad():
            output = _normalize_output(_invoke_model(model, batch_dict), model_schema)

        is_valid, error = validate_output_data(output)
        if not is_valid:
            raise error
        is_valid, error = validate_output_data_with_model_schema(output, model_schema)
        if not is_valid:
            raise error
