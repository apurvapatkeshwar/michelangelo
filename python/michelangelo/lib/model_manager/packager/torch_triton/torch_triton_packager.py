"""Packager for torch Triton models."""

from __future__ import annotations

import tempfile
from typing import TYPE_CHECKING, Any

from michelangelo._internal.utils.file_utils import generate_folder
from michelangelo.lib.model_manager._private.constants import TritonBackendType
from michelangelo.lib.model_manager._private.packager.template_renderer import (
    TritonTemplateRenderer,
)
from michelangelo.lib.model_manager._private.packager.torch_triton import (
    generate_model_package_content,
    generate_raw_model_package_content,
    validate_model_class,
    validate_raw_model_file,
    validate_raw_model_package,
)
from michelangelo.lib.model_manager._private.schema.triton import validate_model_schema
from michelangelo.lib.model_manager._private.utils.data_utils import (
    validate_sample_data,
    validate_sample_data_with_model_schema,
)
from michelangelo.lib.model_manager.constants import StorageType

if TYPE_CHECKING:
    from numpy import ndarray

    from michelangelo.lib.artifact_manager import StorageBackend
    from michelangelo.lib.model_manager.schema import ModelSchema

_SUPPORTED_BACKENDS = {
    TritonBackendType.TORCH,
    TritonBackendType.PYTHON,
    TritonBackendType.ONNX,
}


class TorchTritonPackager:
    """Package PyTorch models for Triton and Michelangelo workflow artifacts.

    The packager creates local package directories. Uploading those directories
    to local, MinIO, S3-compatible, or other artifact storage is the pusher's
    responsibility through ``StorageBackend``.
    """

    def __init__(self, storage_backend: StorageBackend | None = None):
        """Create a torch Triton packager.

        Args:
            storage_backend: Optional storage backend used to download
                ``model_path`` when it is a backend URI. If omitted,
                ``model_path`` is treated as a local filesystem path.
        """
        self.gen = TritonTemplateRenderer()
        self.storage_backend = storage_backend

    def create_model_package(
        self,
        model_path: str,
        model_schema: ModelSchema,
        model_name: str | None = None,
        dest_model_path: str | None = None,
        model_revision: str | None = "0",
        model_path_source_type: str | None = StorageType.LOCAL,
        sample_data: list[dict[str, ndarray]] | None = None,
        model_class: str | None = None,
        hyperparameters: dict | None = None,
        enable_dynamic_batching: bool = True,
        backend: str | None = None,
        include_import_prefixes: list[str] | None = None,
        triton_parameters: dict[str, Any] | None = None,
    ) -> str:
        """Create a deployable Triton model package.

        Args:
            model_path: Local path or storage URI for a torch artifact. For
                the default PyTorch backend this should be a torchscript file
                or a torch module/state-dict convertible to torchscript. For
                the Python backend this should be a torch module/state-dict.
                For ONNX Runtime this may be an ``.onnx`` file or a torch
                artifact exportable with ``sample_data``.
            model_schema: Model input/output schema.
            model_name: Optional model name written to ``config.pbtxt``.
            dest_model_path: Destination package directory. A temporary
                directory is created when omitted.
            model_revision: Optional model revision suffix. Defaults to ``"0"``.
            model_path_source_type: Source type for local artifacts. Defaults
                to ``StorageType.LOCAL``.
            sample_data: Optional sample inputs. Required when exporting a
                torch artifact to ONNX.
            model_class: Required for Python backend and state-dict conversion.
            hyperparameters: Constructor keyword arguments for ``model_class``.
            enable_dynamic_batching: Whether to enable Triton dynamic batching.
            backend: Triton backend. Supports ``pytorch``, ``python``, and
                ``onnxruntime``. Defaults to ``pytorch``.
            include_import_prefixes: Module prefixes to include when bundling
                Python backend model definitions. If ``None`` or empty, all
                discovered local imports are included.
            triton_parameters: Optional Triton config template overrides.

        Returns:
            Absolute path to the generated local package directory.
        """
        if not model_path:
            raise ValueError("model_path is required")
        if not model_schema:
            raise ValueError("model_schema is required")

        backend = backend or TritonBackendType.TORCH
        if backend not in _SUPPORTED_BACKENDS:
            raise ValueError(
                f"Unsupported backend: {backend!r}. "
                f"Supported backends are: {sorted(_SUPPORTED_BACKENDS)}"
            )

        is_schema_valid, error = validate_model_schema(model_schema)
        if not is_schema_valid:
            raise error

        if backend == TritonBackendType.PYTHON and not model_class:
            raise ValueError("model_class is required for Python backend")
        if model_class:
            is_model_class_valid, error = validate_model_class(model_class)
            if not is_model_class_valid:
                raise error

        if sample_data:
            is_sample_data_valid, error = validate_sample_data(sample_data)
            if not is_sample_data_valid:
                raise error
            is_sample_data_with_schema_valid, error = (
                validate_sample_data_with_model_schema(sample_data, model_schema)
            )
            if not is_sample_data_with_schema_valid:
                raise error

        if not dest_model_path:
            dest_model_path = tempfile.mkdtemp()

        content = generate_model_package_content(
            self.gen,
            model_path,
            model_name,
            model_revision,
            model_schema,
            model_path_source_type=model_path_source_type,
            root_path=dest_model_path,
            enable_dynamic_batching=enable_dynamic_batching,
            model_class=model_class,
            hyperparameters=hyperparameters,
            backend=backend,
            include_import_prefixes=include_import_prefixes,
            sample_data=sample_data,
            triton_parameters=triton_parameters,
            storage_backend=self.storage_backend,
        )

        generate_folder(content, dest_model_path)
        return dest_model_path

    def create_raw_model_package(
        self,
        model_path: str,
        model_class: str,
        model_schema: ModelSchema,
        sample_data: list[dict[str, ndarray]],
        dest_model_path: str | None = None,
        model_path_source_type: str | None = StorageType.LOCAL,
        requirements: list[str] | str | None = None,
        include_import_prefixes: list[str] | None = None,
        hyperparameters: dict | None = None,
    ) -> str:
        """Create a raw torch model package.

        Args:
            model_path: Local path or storage URI for a torch ``.pt`` or
                ``.pth`` artifact.
            model_class: Fully qualified ``torch.nn.Module`` class path.
            model_schema: Model schema for package metadata and validation.
            sample_data: Sample inputs used for metadata and validation.
            dest_model_path: Destination package directory. A temporary
                directory is created when omitted.
            model_path_source_type: Source type for local artifacts. Defaults
                to ``StorageType.LOCAL``.
            requirements: Optional requirements list or requirements file path.
            include_import_prefixes: Module prefixes to include in ``defs``.
                If ``None`` or empty, all discovered local imports are included.
            hyperparameters: Constructor keyword arguments for ``model_class``.

        Returns:
            Absolute path to the generated local package directory.
        """
        if not model_path:
            raise ValueError("model_path is required")
        if not model_class:
            raise ValueError("model_class is required")
        if not model_schema:
            raise ValueError("model_schema is required")

        is_model_class_valid, error = validate_model_class(model_class)
        if not is_model_class_valid:
            raise error

        is_schema_valid, error = validate_model_schema(model_schema)
        if not is_schema_valid:
            raise error

        if sample_data:
            is_sample_data_valid, error = validate_sample_data(sample_data)
            if not is_sample_data_valid:
                raise error
            is_sample_data_with_schema_valid, error = (
                validate_sample_data_with_model_schema(sample_data, model_schema)
            )
            if not is_sample_data_with_schema_valid:
                raise error

        if not dest_model_path:
            dest_model_path = tempfile.mkdtemp()

        content = generate_raw_model_package_content(
            model_path,
            model_class,
            model_schema,
            sample_data,
            model_path_source_type=model_path_source_type,
            requirements=requirements,
            root_path=dest_model_path,
            include_import_prefixes=include_import_prefixes,
            hyperparameters=hyperparameters,
            storage_backend=self.storage_backend,
        )

        generate_folder(content, dest_model_path)
        validate_raw_model_package(dest_model_path, sample_data, model_schema)
        return dest_model_path

    def validate_raw_model_file(self, model_path: str) -> None:
        """Validate that a local torch artifact can be used as a raw model file.

        Args:
            model_path: Path to a local ``.pt`` or ``.pth`` artifact.
        """
        is_valid, error = validate_raw_model_file(model_path)
        if not is_valid:
            raise error
