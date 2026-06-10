"""Generate raw model packages for torch models."""

from __future__ import annotations

import os
import shutil
import tempfile
from typing import TYPE_CHECKING

import torch
import yaml

from michelangelo.lib.model_manager._private.packager.custom_triton.model_class import (
    serialize_model_class,
)
from michelangelo.lib.model_manager._private.packager.custom_triton.requirements_txt import (  # noqa: E501
    generate_requirements_txt,
)
from michelangelo.lib.model_manager._private.packager.torch_triton.constants import (
    MODEL_CLASS_FILE_NAME,
    MODEL_PT_FILE_NAME,
    RAW_HYPERPARAMETERS_FILE_NAME,
    RAW_REQUIREMENTS_FILE_NAME,
    RAW_SAMPLE_DATA_FILE_NAME,
    RAW_SCHEMA_FILE_NAME,
    RAW_TYPE_FILE_NAME,
)
from michelangelo.lib.model_manager._private.packager.torch_triton.type_yaml import (
    generate_type_yaml,
)
from michelangelo.lib.model_manager._private.packager.torch_triton.validation import (
    validate_raw_model_file,
)
from michelangelo.lib.model_manager._private.schema.common import schema_to_yaml
from michelangelo.lib.model_manager._private.serde.data import dump_model_data
from michelangelo.lib.model_manager._private.utils.asset_utils import download_assets
from michelangelo.lib.model_manager.constants import StorageType

if TYPE_CHECKING:
    from numpy import ndarray

    from michelangelo.lib.artifact_manager import StorageBackend
    from michelangelo.lib.model_manager.schema import ModelSchema


def convert_to_state_dict(model_path: str) -> None:
    """Convert a saved torch module to state-dict format in place if needed.

    Args:
        model_path: Local path to a torch ``.pt`` or ``.pth`` artifact.

    Raises:
        FileNotFoundError: If ``model_path`` does not exist.
        ValueError: If the artifact cannot be represented as a state dict.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"File does not exist: {model_path}")

    try:
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        if isinstance(state_dict, dict):
            return
    except Exception:
        pass

    try:
        model = torch.load(model_path, map_location="cpu", weights_only=False)
        if not hasattr(model, "state_dict"):
            raise ValueError(f"File does not contain a convertible model: {model_path}")
        torch.save(model.state_dict(), model_path)
    except Exception as exc:
        raise ValueError(
            f"File does not contain a convertible model: {model_path}"
        ) from exc


def _stage_model_file(
    model_path: str,
    model_path_source_type: str,
    model_dir: str,
    storage_backend: StorageBackend | None,
) -> str:
    """Download and normalize a torch artifact into ``model/model.pt``."""
    model_file_path = os.path.join(model_dir, MODEL_PT_FILE_NAME)
    with tempfile.TemporaryDirectory() as temp_dir:
        downloaded_model = os.path.join(temp_dir, "model")
        download_assets(
            model_path,
            downloaded_model,
            model_path_source_type,
            storage_backend=storage_backend,
        )
        if os.path.isdir(downloaded_model):
            pt_files = [
                name for name in os.listdir(downloaded_model) if name.endswith(".pt")
            ]
            if not pt_files:
                raise ValueError(f"No .pt files found in {downloaded_model}")
            if len(pt_files) > 1:
                raise ValueError(
                    f"Multiple .pt files found in {downloaded_model}: {pt_files}"
                )
            shutil.move(os.path.join(downloaded_model, pt_files[0]), model_file_path)
        else:
            shutil.move(downloaded_model, model_file_path)
    return model_file_path


def generate_raw_model_package_content(
    model_path: str,
    model_class: str,
    model_schema: ModelSchema,
    sample_data: list[dict[str, ndarray]],
    model_path_source_type: str | None = StorageType.LOCAL,
    requirements: list[str] | str | None = None,
    root_path: str | None = None,
    include_import_prefixes: list[str] | None = None,
    hyperparameters: dict | None = None,
    storage_backend: StorageBackend | None = None,
) -> dict:
    """Generate raw model package content for a torch model.

    Args:
        model_path: Path or storage URI for a torch artifact.
        model_class: Fully qualified ``torch.nn.Module`` class path.
        model_schema: Model schema for metadata and validation.
        sample_data: Sample inputs used by validators and downstream tooling.
        model_path_source_type: Source type for local artifacts.
        requirements: Optional requirements list or requirements file path.
        root_path: Root directory used to stage package files.
        include_import_prefixes: Import prefixes to include in ``defs``.
        hyperparameters: Constructor keyword arguments for ``model_class``.
        storage_backend: Optional backend used to download remote artifacts.

    Returns:
        Folder content dictionary consumable by ``generate_folder``.
    """
    if not root_path:
        root_path = tempfile.mkdtemp()

    model_dir = os.path.join(root_path, "model")
    os.makedirs(model_dir, exist_ok=True)
    model_file_path = _stage_model_file(
        model_path,
        model_path_source_type,
        model_dir,
        storage_backend,
    )

    is_valid, error = validate_raw_model_file(model_file_path)
    if not is_valid:
        raise error
    convert_to_state_dict(model_file_path)

    defs_path = os.path.join(root_path, "defs")
    serialize_model_class(
        model_class,
        defs_path,
        MODEL_CLASS_FILE_NAME,
        include_import_prefixes=include_import_prefixes,
        serialize_interface=False,
    )

    metadata_content = {
        RAW_TYPE_FILE_NAME: generate_type_yaml(),
        RAW_SCHEMA_FILE_NAME: schema_to_yaml(model_schema),
        RAW_SAMPLE_DATA_FILE_NAME: dump_model_data(sample_data),
    }
    if hyperparameters is not None:
        metadata_content[RAW_HYPERPARAMETERS_FILE_NAME] = yaml.safe_dump(
            hyperparameters,
            default_flow_style=False,
            sort_keys=False,
        )

    if requirements is None:
        requirements = ["torch"]
    elif isinstance(requirements, list) and "torch" not in requirements:
        requirements = [*requirements, "torch"]

    return {
        "metadata": metadata_content,
        "model": f"dir://{model_dir}",
        "defs": f"dir://{defs_path}",
        "dependencies": {
            RAW_REQUIREMENTS_FILE_NAME: generate_requirements_txt(requirements),
        },
    }
