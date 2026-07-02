"""Unit tests for ``...tabular_assembler._private.model_class.resolve``."""

from __future__ import annotations

import unittest
from typing import TYPE_CHECKING

import pytorch_lightning as pl
import torch.nn as nn

from michelangelo.lib.model_manager.interface.custom_model import Model
from michelangelo.workflow.tasks.tabular_assembler._private.model_class.resolve import (
    resolve_model_class,
    resolve_training_framework,
    try_load_class,
)
from michelangelo.workflow.variables.metadata import (
    TRAINING_FRAMEWORK_CUSTOM,
    TRAINING_FRAMEWORK_LIGHTNING,
    TRAINING_FRAMEWORK_PYTORCH,
)

if TYPE_CHECKING:
    from numpy import ndarray


class _CustomModelFixture(Model):
    """Minimal concrete ``Model`` used only to exercise framework resolution."""

    def save(self, path: str) -> None:
        pass

    @classmethod
    def load(cls, path: str) -> _CustomModelFixture:
        return cls()

    def predict(self, inputs: dict[str, ndarray]) -> dict[str, ndarray]:
        return inputs


_CUSTOM_MODEL_PATH = (
    "michelangelo.workflow.tasks.tabular_assembler._private.model_class.tests."
    "resolve_test._CustomModelFixture"
)


class _MinimalTorch(nn.Module):
    def forward(self, x):  # type: ignore[override]
        return x


_TORCH_MODEL_PATH = (
    "michelangelo.workflow.tasks.tabular_assembler._private.model_class.tests."
    "resolve_test._MinimalTorch"
)


class _MinimalLightning(pl.LightningModule):
    def forward(self, x):  # type: ignore[override]
        return x


_LIGHTNING_MODEL_PATH = (
    "michelangelo.workflow.tasks.tabular_assembler._private.model_class.tests."
    "resolve_test._MinimalLightning"
)


class ResolveTest(unittest.TestCase):
    """Tests for ``try_load_class``, ``resolve_training_framework``, ``resolve_model_class``."""  # noqa: E501

    def test_try_load_class_none(self):
        """``None`` input returns ``None``."""
        self.assertIsNone(try_load_class(None))

    def test_try_load_class_empty_string(self):
        """Empty string input returns ``None``."""
        self.assertIsNone(try_load_class(""))

    def test_try_load_class_invalid_module(self):
        """An unimportable module path returns ``None``."""
        self.assertIsNone(try_load_class("nonexistent_module_xyz_abc.SomeClass"))

    def test_try_load_class_builtin_type(self):
        """A builtin type resolves to that type."""
        self.assertIs(try_load_class("builtins.str"), str)

    def test_try_load_class_non_type_attribute(self):
        """A non-class attribute (e.g. a function) returns ``None``."""
        self.assertIsNone(try_load_class("math.sqrt"))

    def test_resolve_training_framework_custom_model(self):
        """A ``Model`` subclass resolves to the custom framework."""
        self.assertEqual(
            resolve_training_framework(_CUSTOM_MODEL_PATH), TRAINING_FRAMEWORK_CUSTOM
        )

    def test_resolve_training_framework_lightning_before_torch(self):
        """A ``LightningModule`` subclass resolves to lightning, not plain torch."""
        self.assertEqual(
            resolve_training_framework(_LIGHTNING_MODEL_PATH),
            TRAINING_FRAMEWORK_LIGHTNING,
        )

    def test_resolve_training_framework_plain_torch_module(self):
        """A plain ``nn.Module`` subclass resolves to the pytorch framework."""
        self.assertEqual(
            resolve_training_framework(_TORCH_MODEL_PATH), TRAINING_FRAMEWORK_PYTORCH
        )

    def test_resolve_training_framework_none_and_invalid(self):
        """Unresolvable paths and non-framework classes return ``None``."""
        self.assertIsNone(
            resolve_training_framework("nonexistent_module_xyz_abc.Model")
        )
        # str is not a Model / LightningModule / nn.Module in the resolver sense.
        self.assertIsNone(resolve_training_framework("builtins.str"))

    def test_resolve_model_class_uses_config_when_importable(self):
        """The config-supplied class wins when it imports successfully."""
        self.assertEqual(
            resolve_model_class(_CUSTOM_MODEL_PATH, "metadata.fallback.Model"),
            _CUSTOM_MODEL_PATH,
        )

    def test_resolve_model_class_falls_back_when_config_not_importable(self):
        """Falls back to the metadata class when the config class is unimportable."""
        self.assertEqual(
            resolve_model_class(
                "nonexistent_module_xyz_abc.Model", "metadata.fallback.Model"
            ),
            "metadata.fallback.Model",
        )

    def test_resolve_model_class_without_config_uses_metadata(self):
        """With no config class at all, the metadata class is used."""
        self.assertEqual(resolve_model_class(None, "meta.Model"), "meta.Model")
