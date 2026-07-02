"""Abstract base class for model packagers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from numpy import ndarray

    from michelangelo.lib.artifact_manager.storage_backend import StorageBackend
    from michelangelo.lib.model_manager.schema import ModelSchema

__all__ = ["PackagerBase"]


class PackagerBase(ABC):
    """Abstract base for model packagers that emit Triton-format packages.

    A packager turns a trained model artifact into a directory laid out for
    NVIDIA Triton Inference Server. Concrete packagers specialize on the model
    framework (PyTorch via ``TorchTritonPackager``, custom Python via
    ``CustomTritonPackager``) and may accept framework-specific options.

    Two package shapes are produced:

    * **Deployable package** (``create_model_package``) — the artifact served
      by Triton in Michelangelo Studio.
    * **Raw package** (``create_raw_model_package``) — a self-contained bundle
      with sample data, used for validation and incremental/baseline
      retraining.

    Infrastructure is supplied by constructor injection so packager logic
    stays free of backend-specific client setup, mirroring
    ``michelangelo.workflow.tasks.pusher.plugins.base.PusherPluginBase``.

    Every parameter after ``model_path`` on the abstract methods is
    keyword-only. Concrete packagers accept these parameters in different
    positional orders (e.g. ``model_class`` is the second positional
    parameter on ``CustomTritonPackager.create_model_package`` but not on
    ``TorchTritonPackager.create_model_package``), so calling through a
    ``PackagerBase`` reference positionally would silently misbind arguments.
    Always call shared parameters by keyword when working against this base
    class.

    Args:
        storage_backend: Backend used to download source artifacts and upload
            produced packages. ``None`` for packagers that still resolve
            paths via ``model_path_source_type``; subclasses that require a
            backend validate it in ``__init__``.

    Example::

        class MyPackager(PackagerBase):
            def create_model_package(
                self, model_path, *, model_schema, dest_model_path=None,
                model_path_source_type=None, **kwargs,
            ) -> str:
                dest = dest_model_path or tempfile.mkdtemp()
                # ...build Triton layout under dest...
                return dest

            def create_raw_model_package(
                self, model_path, *, model_schema, sample_data,
                dest_model_path=None, model_path_source_type=None, **kwargs,
            ) -> str:
                ...
    """

    def __init__(self, storage_backend: StorageBackend | None = None) -> None:
        """Initialize the packager with an optional injected storage backend.

        Args:
            storage_backend: Backend used to download source artifacts and
                upload produced packages. ``None`` for packagers that still
                resolve paths via ``model_path_source_type``; subclasses that
                require a backend should validate it here (e.g. raise if
                ``None``) rather than fail later at package-creation time.
        """
        self._storage_backend = storage_backend

    @abstractmethod
    def create_model_package(
        self,
        model_path: str,
        *,
        model_schema: ModelSchema,
        dest_model_path: str | None = None,
        model_path_source_type: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Create a deployable Triton model package.

        Args:
            model_path: Path to the trained model artifact (file or
                directory).
            model_schema: Input/output schema describing the model's I/O
                tensors.
            dest_model_path: Directory to write the package into. When
                ``None`` a temporary directory is created and its path
                returned.
            model_path_source_type: Storage backend identifier for
                ``model_path`` (a ``StorageType`` value, e.g.
                ``StorageType.LOCAL``). Ignored by packagers that resolve
                artifacts via an injected ``storage_backend``.
            **kwargs: Packager-specific options. ``TorchTritonPackager``
                accepts ``backend``, ``hyperparameters``, ``model_class``,
                ``sample_data``, ``enable_dynamic_batching``;
                ``CustomTritonPackager`` accepts ``model_class``,
                ``additional_import_prefixes``, ``triton_parameters``. See
                each concrete packager for its full signature.

        Returns:
            Absolute path to the generated deployable package directory.

        Raises:
            ValueError: If required inputs (e.g. ``model_schema``) are
                missing or fail validation.
        """

    @abstractmethod
    def create_raw_model_package(
        self,
        model_path: str,
        *,
        model_schema: ModelSchema,
        sample_data: list[dict[str, ndarray]],
        dest_model_path: str | None = None,
        model_path_source_type: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Create a self-contained raw model package with sample data.

        The raw package bundles model artifacts, implementation code, and
        sample inputs so the package can be validated offline and reused for
        incremental or baseline retraining.

        Args:
            model_path: Path to the trained model artifact (file or
                directory).
            model_schema: Input/output schema describing the model's I/O
                tensors.
            sample_data: Sample inputs for the model's predict/forward
                function; each item maps input feature names to ``numpy``
                arrays. Used to validate the package after it is written.
            dest_model_path: Directory to write the package into. When
                ``None`` a temporary directory is created and its path
                returned.
            model_path_source_type: Storage backend identifier for
                ``model_path`` (a ``StorageType`` value). Ignored by
                packagers that resolve artifacts via an injected
                ``storage_backend``.
            **kwargs: Packager-specific options (see
                ``create_model_package``).

        Returns:
            Absolute path to the generated raw package directory.

        Raises:
            ValueError: If ``model_class``, ``model_schema``, or
                ``sample_data`` are missing or fail validation.
        """
