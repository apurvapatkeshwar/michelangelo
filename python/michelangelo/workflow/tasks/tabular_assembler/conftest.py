"""Shared pytest fixtures for the tabular assembler test suite.

Collected automatically by pytest for every test module under this package
(``tests/``, ``custom/tests/``, ``_private/model_class/tests/``, ...).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from michelangelo.lib.model_manager.interface.custom_model import Model

if TYPE_CHECKING:
    from numpy import ndarray

__all__ = ["_CustomModelFixture"]


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
