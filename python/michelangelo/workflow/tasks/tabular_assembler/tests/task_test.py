"""Unit tests for ``michelangelo.workflow.tasks.tabular_assembler.task``."""

from __future__ import annotations

import tempfile
import unittest
from unittest.mock import patch

from michelangelo.lib.artifact_manager.storage_backend import LocalStorageBackend
from michelangelo.workflow.schema.assembler import TabularAssemblerConfig
from michelangelo.workflow.tasks.tabular_assembler.task import tabular_assembler
from michelangelo.workflow.variables.metadata import (
    TRAINING_FRAMEWORK_CUSTOM,
    TRAINING_FRAMEWORK_LIGHTNING,
    TRAINING_FRAMEWORK_PYTORCH,
    ModelMetadata,
)
from michelangelo.workflow.variables.types import AssembledModel, ModelArtifact

_TASK_MODULE = "michelangelo.workflow.tasks.tabular_assembler.task"

_CUSTOM_MODEL_CLASS = (
    "michelangelo.workflow.tasks.tabular_assembler.conftest._CustomModelFixture"
)


class TabularAssemblerDispatchTest(unittest.TestCase):
    """Tests for ``tabular_assembler``'s framework dispatch."""

    def setUp(self) -> None:
        """Create a fresh ``LocalStorageBackend`` rooted at a temp dir per test."""
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.storage_backend = LocalStorageBackend(self._tmp.name)

    def _sentinel_result(self) -> AssembledModel:
        """Return a placeholder ``AssembledModel`` for mocking downstream assemblers."""
        return AssembledModel(
            raw_model=ModelArtifact(path="raw"),
            deployable_model=ModelArtifact(path="deployable"),
        )

    @patch(f"{_TASK_MODULE}.custom_assembler")
    def test_dispatches_to_custom_when_metadata_framework_custom(self, mock_custom):
        """``training_framework == custom`` routes straight to ``custom_assembler``."""
        mock_custom.return_value = self._sentinel_result()
        config = TabularAssemblerConfig()
        raw_model = ModelArtifact(
            path="p",
            metadata=ModelMetadata(training_framework=TRAINING_FRAMEWORK_CUSTOM),
        )

        result = tabular_assembler(
            config, raw_model, storage_backend=self.storage_backend
        )

        self.assertIs(result, mock_custom.return_value)
        mock_custom.assert_called_once_with(
            config, raw_model, None, storage_backend=self.storage_backend
        )

    @patch(f"{_TASK_MODULE}.custom_assembler")
    def test_dispatches_to_custom_when_config_model_class_is_custom_model(
        self, mock_custom
    ):
        """A config-supplied custom ``Model`` class forces the custom path.

        This holds even when the recorded training framework is ``lightning``.
        """
        mock_custom.return_value = self._sentinel_result()
        config = TabularAssemblerConfig(model_class=_CUSTOM_MODEL_CLASS)
        raw_model = ModelArtifact(
            path="p",
            metadata=ModelMetadata(training_framework=TRAINING_FRAMEWORK_LIGHTNING),
        )
        native_tx = ModelArtifact(path="tx")

        tabular_assembler(
            config, raw_model, native_tx, storage_backend=self.storage_backend
        )

        mock_custom.assert_called_once_with(
            config, raw_model, native_tx, storage_backend=self.storage_backend
        )

    def test_torch_dispatch_resolves_now_that_torch_assembler_exists(self):
        """``torch.assembler`` now exists.

        The lazy import in ``_torch_assembler`` resolves instead of raising
        ``NotImplementedError``.
        """
        from michelangelo.workflow.tasks.tabular_assembler.torch.assembler import (
            torch_assembler,
        )

        self.assertTrue(callable(torch_assembler))

    @patch(
        "michelangelo.workflow.tasks.tabular_assembler.torch.assembler.torch_assembler"
    )
    def test_dispatches_to_torch_assembler_for_pytorch_framework(self, mock_torch):
        """``training_framework == pytorch`` routes to the real torch assembler."""
        mock_torch.return_value = self._sentinel_result()
        config = TabularAssemblerConfig()
        raw_model = ModelArtifact(
            path="p",
            metadata=ModelMetadata(training_framework=TRAINING_FRAMEWORK_PYTORCH),
        )

        result = tabular_assembler(
            config, raw_model, storage_backend=self.storage_backend
        )

        self.assertIs(result, mock_torch.return_value)
        mock_torch.assert_called_once_with(
            config, raw_model, None, storage_backend=self.storage_backend
        )

    @patch(
        "michelangelo.workflow.tasks.tabular_assembler.torch.assembler.torch_assembler"
    )
    def test_dispatches_to_torch_assembler_for_lightning_framework(self, mock_torch):
        """``training_framework == lightning`` routes to the real torch assembler."""
        mock_torch.return_value = self._sentinel_result()
        config = TabularAssemblerConfig()
        raw_model = ModelArtifact(
            path="p",
            metadata=ModelMetadata(training_framework=TRAINING_FRAMEWORK_LIGHTNING),
        )
        native_tx = ModelArtifact(path="tx")

        result = tabular_assembler(
            config, raw_model, native_tx, storage_backend=self.storage_backend
        )

        self.assertIs(result, mock_torch.return_value)
        mock_torch.assert_called_once_with(
            config, raw_model, native_tx, storage_backend=self.storage_backend
        )

    def test_unsupported_framework_returns_empty_placeholder_pair(self):
        """An unrecognized framework yields an empty (not ``None``) artifact pair."""
        config = TabularAssemblerConfig()
        raw_model = ModelArtifact(
            path="p", metadata=ModelMetadata(training_framework="unsupported_framework")
        )

        result = tabular_assembler(
            config, raw_model, storage_backend=self.storage_backend
        )

        self.assertIsNotNone(result.raw_model)
        self.assertIsNotNone(result.deployable_model)
        self.assertEqual(result.raw_model.path, "")
        self.assertEqual(result.deployable_model.path, "")

    def test_no_framework_recorded_and_no_config_model_class_returns_empty_pair(self):
        """No recorded framework and no config model class yields the empty pair."""
        config = TabularAssemblerConfig()
        raw_model = ModelArtifact(path="p", metadata=ModelMetadata())

        result = tabular_assembler(
            config, raw_model, storage_backend=self.storage_backend
        )

        self.assertEqual(result.raw_model.path, "")
        self.assertEqual(result.deployable_model.path, "")


if __name__ == "__main__":
    unittest.main()
