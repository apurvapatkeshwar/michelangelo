"""Generate deployable Triton packages for torch models."""

from __future__ import annotations

import os
import tempfile
from typing import TYPE_CHECKING, Any

import yaml

from michelangelo.lib.model_manager._private.constants import TritonBackendType
from michelangelo.lib.model_manager._private.packager.custom_triton.model_class import (
    serialize_model_class,
)
from michelangelo.lib.model_manager._private.packager.custom_triton.model_py import (
    generate_model_py_content,
)
from michelangelo.lib.model_manager._private.packager.torch_triton.config_pbtxt import (
    generate_config_pbtxt_content,
)
from michelangelo.lib.model_manager._private.packager.torch_triton.constants import (
    DEPLOYABLE_CONFIG_FILE_NAME,
    DEPLOYABLE_MODEL_ONNX_FILE_NAME,
    DEPLOYABLE_MODEL_PY_FILE_NAME,
    DEPLOYABLE_SKELETON_FILE_NAME,
    DEPLOYABLE_USER_MODEL_PY_FILE_NAME,
    MODEL_CLASS_FILE_NAME,
    MODEL_PT_FILE_NAME,
)
from michelangelo.lib.model_manager._private.utils.asset_utils import download_assets
from michelangelo.lib.model_manager.constants import StorageType

from .onnx_conversion import convert_to_onnx
from .raw_model_package import convert_to_state_dict
from .torchscript_conversion import convert_to_torchscript
from .user_model_py import generate_torch_python_user_model_content
from .validation import validate_raw_model_file

if TYPE_CHECKING:
    from michelangelo.lib.artifact_manager import StorageBackend
    from michelangelo.lib.model_manager._private.packager.template_renderer import (
        TritonTemplateRenderer,
    )
    from michelangelo.lib.model_manager.schema import ModelSchema


def _download_model_file(
    model_path: str,
    target_model_path: str,
    model_path_source_type: str,
    storage_backend: StorageBackend | None,
) -> None:
    """Download a model artifact into a target local file path."""
    download_assets(
        model_path,
        target_model_path,
        model_path_source_type,
        storage_backend=storage_backend,
    )


def _serialize_model_definition(
    model_class: str,
    model_version_dir: str,
    hyperparameters: dict | None,
    include_import_prefixes: list[str] | None,
) -> dict:
    """Serialize model class metadata for Python-backend deployable packages."""
    serialize_model_class(
        model_class,
        model_version_dir,
        MODEL_CLASS_FILE_NAME,
        include_import_prefixes=include_import_prefixes,
        serialize_interface=False,
    )
    skeleton = (
        hyperparameters
        if hyperparameters and "_target_" in hyperparameters
        else {"_target_": model_class, **(hyperparameters or {})}
    )
    skeleton_path = os.path.join(model_version_dir, DEPLOYABLE_SKELETON_FILE_NAME)
    with open(skeleton_path, "w") as f:
        yaml.safe_dump(skeleton, f, default_flow_style=False, sort_keys=False)
    return {
        MODEL_CLASS_FILE_NAME: (
            f"file://{os.path.join(model_version_dir, MODEL_CLASS_FILE_NAME)}"
        ),
        DEPLOYABLE_SKELETON_FILE_NAME: f"file://{skeleton_path}",
    }


def _generate_python_backend_wrappers(
    gen: TritonTemplateRenderer,
    model_schema: ModelSchema,
) -> dict:
    """Generate Triton Python backend wrapper files."""
    output_names = [item.name for item in model_schema.output_schema]
    return {
        DEPLOYABLE_MODEL_PY_FILE_NAME: generate_model_py_content(gen),
        DEPLOYABLE_USER_MODEL_PY_FILE_NAME: generate_torch_python_user_model_content(
            gen,
            output_names,
        ),
    }


def generate_model_package_content(
    gen: TritonTemplateRenderer,
    model_path: str,
    model_name: str,
    model_revision: str,
    model_schema: ModelSchema,
    model_path_source_type: str | None = StorageType.LOCAL,
    root_path: str | None = None,
    enable_dynamic_batching: bool = True,
    model_class: str | None = None,
    hyperparameters: dict | None = None,
    backend: str | None = None,
    include_import_prefixes: list[str] | None = None,
    sample_data: list[dict[str, Any]] | None = None,
    triton_parameters: dict[str, Any] | None = None,
    storage_backend: StorageBackend | None = None,
) -> dict:
    """Generate deployable Triton package content for a torch model.

    Args:
        gen: Triton template renderer.
        model_path: Path or storage URI for a torch or ONNX artifact.
        model_name: Model name to place in ``config.pbtxt``.
        model_revision: Optional revision suffix.
        model_schema: Model schema for Triton config generation.
        model_path_source_type: Source type for local artifacts.
        root_path: Root directory used to stage package files.
        enable_dynamic_batching: Whether to enable Triton dynamic batching.
        model_class: Required for Python backend and state-dict conversion.
        hyperparameters: Constructor keyword arguments for ``model_class``.
        backend: Triton backend. Defaults to ``pytorch``.
        include_import_prefixes: Import prefixes to include for Python backend.
        sample_data: Optional sample input for ONNX export.
        triton_parameters: Optional Triton config template overrides.
        storage_backend: Optional backend used to download remote artifacts.

    Returns:
        Folder content dictionary consumable by ``generate_folder``.
    """
    if not root_path:
        root_path = tempfile.mkdtemp()
    if backend is None:
        backend = TritonBackendType.TORCH

    config_pbtxt = generate_config_pbtxt_content(
        gen,
        model_name,
        model_revision,
        model_schema,
        backend=backend,
        enable_dynamic_batching=enable_dynamic_batching,
        triton_parameters=triton_parameters,
    )

    model_version_dir = os.path.join(root_path, "0")
    os.makedirs(model_version_dir, exist_ok=True)
    content: dict[str, Any] = {DEPLOYABLE_CONFIG_FILE_NAME: config_pbtxt, "0": {}}

    if backend == TritonBackendType.PYTHON:
        target_model_path = os.path.join(model_version_dir, MODEL_PT_FILE_NAME)
        _download_model_file(
            model_path,
            target_model_path,
            model_path_source_type,
            storage_backend,
        )
        is_valid, error = validate_raw_model_file(target_model_path)
        if not is_valid:
            raise error
        convert_to_state_dict(target_model_path)

        model_subdir = os.path.join(model_version_dir, "model")
        os.makedirs(model_subdir, exist_ok=True)
        final_model_path = os.path.join(model_subdir, MODEL_PT_FILE_NAME)
        os.replace(target_model_path, final_model_path)

        content["0"].update(
            _serialize_model_definition(
                model_class,
                model_version_dir,
                hyperparameters,
                include_import_prefixes,
            )
        )
        content["0"].update(_generate_python_backend_wrappers(gen, model_schema))
        content["0"]["model"] = {MODEL_PT_FILE_NAME: f"file://{final_model_path}"}
    elif backend == TritonBackendType.ONNX:
        with tempfile.TemporaryDirectory() as staging_dir:
            staging_path = os.path.join(staging_dir, "model")
            _download_model_file(
                model_path,
                staging_path,
                model_path_source_type,
                storage_backend,
            )
            final_onnx_path = os.path.join(
                model_version_dir,
                DEPLOYABLE_MODEL_ONNX_FILE_NAME,
            )
            convert_to_onnx(
                staging_path,
                final_onnx_path,
                model_schema,
                sample_data=sample_data[0] if sample_data else None,
                model_class=model_class,
                hyperparameters=hyperparameters,
                enable_dynamic_batching=enable_dynamic_batching,
            )
        content["0"][DEPLOYABLE_MODEL_ONNX_FILE_NAME] = f"file://{final_onnx_path}"
    else:
        target_model_path = os.path.join(model_version_dir, MODEL_PT_FILE_NAME)
        _download_model_file(
            model_path,
            target_model_path,
            model_path_source_type,
            storage_backend,
        )
        convert_to_torchscript(target_model_path, model_class, hyperparameters)
        content["0"][MODEL_PT_FILE_NAME] = f"file://{target_model_path}"

    return content
