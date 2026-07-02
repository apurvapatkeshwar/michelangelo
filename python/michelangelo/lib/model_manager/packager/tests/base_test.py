"""Tests for the PackagerBase abstract base class."""

from __future__ import annotations

import inspect
import tempfile
from typing import Any
from unittest import TestCase

from michelangelo.lib.artifact_manager.storage_backend import LocalStorageBackend
from michelangelo.lib.model_manager.packager import PackagerBase
from michelangelo.lib.model_manager.packager.custom_triton import CustomTritonPackager
from michelangelo.lib.model_manager.packager.torch_triton import TorchTritonPackager


class _CompletePackager(PackagerBase):
    """Minimal concrete subclass implementing both abstract methods."""

    def create_model_package(
        self,
        model_path: str,
        *,
        model_schema: Any = None,
        dest_model_path: str | None = None,
        model_path_source_type: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Return a temp dir standing in for a deployable package."""
        return dest_model_path or tempfile.mkdtemp()

    def create_raw_model_package(
        self,
        model_path: str,
        *,
        model_schema: Any = None,
        sample_data: Any = None,
        dest_model_path: str | None = None,
        model_path_source_type: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Return a temp dir standing in for a raw package."""
        return dest_model_path or tempfile.mkdtemp()


class _PartialPackager(PackagerBase):
    """Subclass implementing only one of the two abstract methods."""

    def create_model_package(
        self,
        model_path: str,
        *,
        model_schema: Any = None,
        dest_model_path: str | None = None,
        model_path_source_type: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Return a temp dir standing in for a deployable package."""
        return dest_model_path or tempfile.mkdtemp()


class TestPackagerBaseABC(TestCase):
    """Tests for PackagerBase's abstract contract."""

    def test_cannot_be_instantiated_directly(self):
        """It raises TypeError when instantiated without concrete methods."""
        with self.assertRaises(TypeError):
            PackagerBase()  # type: ignore[abstract]

    def test_partial_subclass_cannot_be_instantiated(self):
        """It raises TypeError when a subclass omits an abstract method."""
        with self.assertRaises(TypeError):
            _PartialPackager()  # type: ignore[abstract]

    def test_complete_subclass_can_be_instantiated(self):
        """A subclass implementing both abstract methods can be constructed."""
        packager = _CompletePackager()
        self.assertIsInstance(packager, PackagerBase)

    def test_complete_subclass_methods_return_path(self):
        """Concrete methods on a minimal subclass return the package path."""
        packager = _CompletePackager()
        dest = tempfile.mkdtemp()
        self.assertEqual(
            packager.create_model_package(
                "model.pt", model_schema=None, dest_model_path=dest
            ),
            dest,
        )
        self.assertEqual(
            packager.create_raw_model_package(
                "model.pt", model_schema=None, sample_data=None, dest_model_path=dest
            ),
            dest,
        )

    def test_constructor_stores_storage_backend(self):
        """The constructor stores the injected storage backend."""
        backend = LocalStorageBackend(base_dir=tempfile.mkdtemp())
        packager = _CompletePackager(storage_backend=backend)
        self.assertIs(packager._storage_backend, backend)

    def test_constructor_defaults_storage_backend_to_none(self):
        """The storage backend defaults to None when not provided."""
        packager = _CompletePackager()
        self.assertIsNone(packager._storage_backend)

    def test_abstract_method_params_are_keyword_only(self):
        """Every param after model_path on the abstract methods is kw-only."""
        for method_name in ("create_model_package", "create_raw_model_package"):
            sig = inspect.signature(getattr(PackagerBase, method_name))
            params = list(sig.parameters.values())
            # params[0] is 'self', params[1] is 'model_path' (positional-or-kw).
            for param in params[2:]:
                if param.kind is inspect.Parameter.VAR_KEYWORD:
                    continue
                self.assertEqual(
                    param.kind,
                    inspect.Parameter.KEYWORD_ONLY,
                    f"{method_name} param '{param.name}' must be keyword-only",
                )


class TestConcretePackagersInheritBase(TestCase):
    """Verifies the real packagers were wired up to inherit PackagerBase."""

    def test_custom_triton_packager_is_a_packager_base(self):
        """CustomTritonPackager is a PackagerBase instance."""
        self.assertIsInstance(CustomTritonPackager(), PackagerBase)

    def test_torch_triton_packager_is_a_packager_base(self):
        """TorchTritonPackager is a PackagerBase instance."""
        self.assertIsInstance(TorchTritonPackager(), PackagerBase)

    def test_custom_triton_packager_accepts_storage_backend(self):
        """CustomTritonPackager threads storage_backend through to the base."""
        backend = LocalStorageBackend(base_dir=tempfile.mkdtemp())
        packager = CustomTritonPackager(storage_backend=backend)
        self.assertIs(packager._storage_backend, backend)

    def test_torch_triton_packager_accepts_storage_backend(self):
        """TorchTritonPackager threads storage_backend through to the base."""
        backend = LocalStorageBackend(base_dir=tempfile.mkdtemp())
        packager = TorchTritonPackager(storage_backend=backend)
        self.assertIs(packager._storage_backend, backend)
