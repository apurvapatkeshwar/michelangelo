"""Shared pytest fixtures for the tabular assembler test suite.

Collected automatically by pytest for every test module under this package
(``tests/``, ``custom/tests/``, ``_private/model_class/tests/``, ...).
"""

from __future__ import annotations

import tempfile
import unittest
from typing import TYPE_CHECKING

from michelangelo.lib.artifact_manager.storage_backend import LocalStorageBackend
from michelangelo.lib.model_manager.interface.custom_model import Model

if TYPE_CHECKING:
    from numpy import ndarray

__all__ = [
    "CUSTOM_MODEL_CLASS_PATH",
    "_CustomModelFixture",
    "_LocalStorageBackendTestCase",
]

CUSTOM_MODEL_CLASS_PATH = (
    "michelangelo.workflow.tasks.tabular_assembler.conftest._CustomModelFixture"
)


class _LocalStorageBackendTestCase(unittest.TestCase):
    """Shared ``setUp`` for tests needing a fresh ``LocalStorageBackend``."""

    def setUp(self) -> None:
        """Create a fresh ``LocalStorageBackend`` rooted at a temp dir per test."""
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.storage_backend = LocalStorageBackend(self._tmp.name)


class _CustomModelFixture(Model):
    """Minimal concrete ``Model`` used only as an importable dotted path.

    Shared across the assembler test suites that need a real, loadable
    ``model_class`` value (e.g. for ``import_attribute`` / framework
    resolution) rather than a mock.
    """

    def save(self, path: str) -> None:
        pass

    @classmethod
    def load(cls, path: str) -> _CustomModelFixture:
        return cls()

    def predict(self, inputs: dict[str, ndarray]) -> dict[str, ndarray]:
        return inputs
