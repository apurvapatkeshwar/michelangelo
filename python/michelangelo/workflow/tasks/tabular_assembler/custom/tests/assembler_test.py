"""Unit tests for ``...tabular_assembler.custom.assembler``."""

from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from michelangelo.lib.artifact_manager.storage_backend import LocalStorageBackend
from michelangelo.lib.model_manager.constants import StorageType
from michelangelo.lib.model_manager.schema import DataType, ModelSchema, ModelSchemaItem
from michelangelo.workflow.schema.assembler import (
    CustomAssemblerConfig,
    TabularAssemblerConfig,
)
from michelangelo.workflow.tasks.tabular_assembler._private.schema.fuse import (
    fuse_model_schema,
)
from michelangelo.workflow.tasks.tabular_assembler.custom.assembler import (
    custom_assembler,
)
from michelangelo.workflow.variables.metadata import (
    TRAINING_FRAMEWORK_CUSTOM,
    ModelMetadata,
)
from michelangelo.workflow.variables.types import ModelArtifact

_CUSTOM_MODEL_CLASS = (
    "michelangelo.workflow.tasks.tabular_assembler.conftest._CustomModelFixture"
)

_ASSEMBLER_MODULE = "michelangelo.workflow.tasks.tabular_assembler.custom.assembler"


def _make_schema() -> ModelSchema:
    return ModelSchema(
        input_schema=[
            ModelSchemaItem(name="input", data_type=DataType.FLOAT, shape=[2, 2]),
            ModelSchemaItem(name="label", data_type=DataType.STRING, shape=[1]),
        ],
        output_schema=[
            ModelSchemaItem(name="output", data_type=DataType.FLOAT, shape=[1])
        ],
    )


def _fake_create_package(dest_dir_name: str):
    """Return a packager-method side effect that materializes a real package dir.

    ``storage_backend.upload`` (unmocked, real ``LocalStorageBackend``) needs
    an actual file on disk to copy, so the packager stand-in must write
    something to ``dest_model_path`` rather than returning a bare string.
    """

    def _side_effect(model_path, *, dest_model_path=None, **kwargs):
        os.makedirs(dest_model_path, exist_ok=True)
        with open(os.path.join(dest_model_path, "artifact.bin"), "wb") as f:
            f.write(dest_dir_name.encode())
        return dest_model_path

    return _side_effect


class CustomAssemblerTest(unittest.TestCase):
    """Tests for ``custom_assembler``."""

    def setUp(self) -> None:
        """Create a fresh ``LocalStorageBackend`` rooted at a temp dir per test."""
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.storage_backend = LocalStorageBackend(self._tmp.name)

    def _upload_raw_model_source(self, contents: bytes = b"weights") -> str:
        """Create a local source dir and upload it, returning a backend URI."""
        src_dir = tempfile.mkdtemp(dir=self._tmp.name)
        with open(os.path.join(src_dir, "model.bin"), "wb") as f:
            f.write(contents)
        return self.storage_backend.upload(
            src_dir, f"sources/{os.path.basename(src_dir)}"
        )

    @patch(f"{_ASSEMBLER_MODULE}.CustomTritonPackager.create_model_package")
    @patch(f"{_ASSEMBLER_MODULE}.CustomTritonPackager.create_raw_model_package")
    def test_custom_assembler_basic(self, mock_create_raw, mock_create_model):
        """Deployable and raw metadata, and upload, all reflect the source model."""
        mock_create_model.side_effect = _fake_create_package("deployable")
        mock_create_raw.side_effect = _fake_create_package("raw")

        config = TabularAssemblerConfig()
        raw_model = ModelArtifact(
            path=self._upload_raw_model_source(),
            metadata=ModelMetadata(
                model_class=_CUSTOM_MODEL_CLASS,
                schema=_make_schema(),
                sample_data=[
                    {
                        "input": np.array([[1.0, 2.0], [3.0, 4.0]]),
                        "label": np.array([b"a"]),
                    }
                ],
                is_incremental_training=True,
                baseline_model_identifier="baseline-model-v1",
            ),
        )

        assembled = custom_assembler(
            config, raw_model, storage_backend=self.storage_backend
        )

        self.assertEqual(assembled.deployable_model.metadata.deployable, True)
        self.assertEqual(assembled.deployable_model.metadata.assembled, True)
        self.assertEqual(
            assembled.deployable_model.metadata.schema, raw_model.metadata.schema
        )

        self.assertEqual(assembled.raw_model.metadata.deployable, False)
        self.assertEqual(assembled.raw_model.metadata.assembled, True)
        self.assertEqual(assembled.raw_model.metadata.schema, raw_model.metadata.schema)
        self.assertEqual(assembled.raw_model.metadata.is_incremental_training, True)
        self.assertEqual(
            assembled.raw_model.metadata.baseline_model_identifier, "baseline-model-v1"
        )
        self.assertEqual(
            assembled.raw_model.metadata.training_framework, TRAINING_FRAMEWORK_CUSTOM
        )
        self.assertEqual(assembled.raw_model.metadata.model_class, _CUSTOM_MODEL_CLASS)

        # Both packaged artifacts were actually uploaded through the backend.
        self.assertTrue(os.path.exists(assembled.deployable_model.path))
        self.assertTrue(os.path.exists(assembled.raw_model.path))

        pkg_kwargs = mock_create_model.call_args.kwargs
        self.assertTrue(
            np.array_equal(
                pkg_kwargs["sample_data"][0]["input"],
                np.array([[1.0, 2.0], [3.0, 4.0]]),
            )
        )
        self.assertEqual(pkg_kwargs["model_path_source_type"], StorageType.LOCAL)
        self.assertIsNone(pkg_kwargs["additional_import_prefixes"])
        self.assertIsNone(pkg_kwargs["include_import_prefixes"])

        raw_kwargs = mock_create_raw.call_args.kwargs
        self.assertEqual(raw_kwargs["model_path_source_type"], StorageType.LOCAL)
        self.assertIsNone(raw_kwargs["additional_import_prefixes"])
        self.assertIsNone(raw_kwargs["include_import_prefixes"])

    @patch(f"{_ASSEMBLER_MODULE}.CustomTritonPackager.create_model_package")
    @patch(f"{_ASSEMBLER_MODULE}.CustomTritonPackager.create_raw_model_package")
    def test_downloads_raw_model_locally_before_packaging(
        self, mock_create_raw, mock_create_model
    ):
        """The packager only understands local paths, so the source must be local.

        The download is materialized under the assembler's own
        ``TemporaryDirectory``, which is gone by the time ``custom_assembler``
        returns — so the on-disk state must be observed from inside the
        packager side effect, not after the call.
        """
        observed: dict[str, object] = {}

        def _observe_and_package(model_path, *, dest_model_path=None, **kwargs):
            observed["is_dir"] = os.path.isdir(model_path)
            with open(os.path.join(model_path, "model.bin"), "rb") as f:
                observed["contents"] = f.read()
            observed["model_path_source_type"] = kwargs["model_path_source_type"]
            os.makedirs(dest_model_path, exist_ok=True)
            return dest_model_path

        mock_create_model.side_effect = _observe_and_package
        mock_create_raw.side_effect = _fake_create_package("raw")

        config = TabularAssemblerConfig()
        raw_model = ModelArtifact(
            path=self._upload_raw_model_source(contents=b"weights-xyz"),
            metadata=ModelMetadata(
                model_class=_CUSTOM_MODEL_CLASS, schema=_make_schema(), sample_data=[{}]
            ),
        )

        custom_assembler(config, raw_model, storage_backend=self.storage_backend)

        self.assertTrue(observed["is_dir"])
        self.assertEqual(observed["contents"], b"weights-xyz")
        self.assertEqual(observed["model_path_source_type"], StorageType.LOCAL)

    @patch(f"{_ASSEMBLER_MODULE}.CustomTritonPackager.create_model_package")
    @patch(f"{_ASSEMBLER_MODULE}.CustomTritonPackager.create_raw_model_package")
    def test_custom_batch_processing_threaded_to_packager(
        self, mock_create_raw, mock_create_model
    ):
        """``custom_batch_processing`` reaches the packager constructor."""
        mock_create_model.side_effect = _fake_create_package("deployable")
        mock_create_raw.side_effect = _fake_create_package("raw")

        config = TabularAssemblerConfig(
            custom=CustomAssemblerConfig(custom_batch_processing=True)
        )
        raw_model = ModelArtifact(
            path=self._upload_raw_model_source(),
            metadata=ModelMetadata(
                model_class=_CUSTOM_MODEL_CLASS, schema=_make_schema(), sample_data=[{}]
            ),
        )

        with patch(f"{_ASSEMBLER_MODULE}.CustomTritonPackager") as mock_packager_cls:
            mock_packager_cls.return_value.create_model_package.side_effect = (
                _fake_create_package("deployable")
            )
            mock_packager_cls.return_value.create_raw_model_package.side_effect = (
                _fake_create_package("raw")
            )
            custom_assembler(config, raw_model, storage_backend=self.storage_backend)
            mock_packager_cls.assert_called_once_with(custom_batch_processing=True)

    @patch(f"{_ASSEMBLER_MODULE}.CustomTritonPackager.create_model_package")
    @patch(f"{_ASSEMBLER_MODULE}.CustomTritonPackager.create_raw_model_package")
    def test_passes_additional_import_prefixes(
        self, mock_create_raw, mock_create_model
    ):
        """Non-empty ``additional_import_prefixes`` reach both packager calls."""
        mock_create_model.side_effect = _fake_create_package("deployable")
        mock_create_raw.side_effect = _fake_create_package("raw")

        prefixes = ["mypkg.dynamic"]
        config = TabularAssemblerConfig(
            custom=CustomAssemblerConfig(additional_import_prefixes=prefixes)
        )
        raw_model = ModelArtifact(
            path=self._upload_raw_model_source(),
            metadata=ModelMetadata(
                model_class=_CUSTOM_MODEL_CLASS, schema=_make_schema(), sample_data=[{}]
            ),
        )

        custom_assembler(config, raw_model, storage_backend=self.storage_backend)

        self.assertEqual(
            mock_create_model.call_args.kwargs["additional_import_prefixes"], prefixes
        )
        self.assertEqual(
            mock_create_raw.call_args.kwargs["additional_import_prefixes"], prefixes
        )

    @patch(f"{_ASSEMBLER_MODULE}.CustomTritonPackager.create_model_package")
    @patch(f"{_ASSEMBLER_MODULE}.CustomTritonPackager.create_raw_model_package")
    def test_passes_empty_additional_import_prefixes(
        self, mock_create_raw, mock_create_model
    ):
        """An empty (but non-``None``) prefix list is passed through as-is."""
        mock_create_model.side_effect = _fake_create_package("deployable")
        mock_create_raw.side_effect = _fake_create_package("raw")

        config = TabularAssemblerConfig(
            custom=CustomAssemblerConfig(additional_import_prefixes=[])
        )
        raw_model = ModelArtifact(
            path=self._upload_raw_model_source(),
            metadata=ModelMetadata(
                model_class=_CUSTOM_MODEL_CLASS, schema=_make_schema(), sample_data=[{}]
            ),
        )

        custom_assembler(config, raw_model, storage_backend=self.storage_backend)

        self.assertEqual(
            mock_create_model.call_args.kwargs["additional_import_prefixes"], []
        )
        self.assertEqual(
            mock_create_raw.call_args.kwargs["additional_import_prefixes"], []
        )

    @patch(f"{_ASSEMBLER_MODULE}.CustomTritonPackager.create_model_package")
    @patch(f"{_ASSEMBLER_MODULE}.CustomTritonPackager.create_raw_model_package")
    def test_passes_include_import_prefixes(self, mock_create_raw, mock_create_model):
        """Non-``None`` ``include_import_prefixes`` reach both packager calls."""
        mock_create_model.side_effect = _fake_create_package("deployable")
        mock_create_raw.side_effect = _fake_create_package("raw")

        prefixes = ["mypkg.models"]
        config = TabularAssemblerConfig(
            custom=CustomAssemblerConfig(include_import_prefixes=prefixes)
        )
        raw_model = ModelArtifact(
            path=self._upload_raw_model_source(),
            metadata=ModelMetadata(
                model_class=_CUSTOM_MODEL_CLASS, schema=_make_schema(), sample_data=[{}]
            ),
        )

        custom_assembler(config, raw_model, storage_backend=self.storage_backend)

        self.assertEqual(
            mock_create_model.call_args.kwargs["include_import_prefixes"], prefixes
        )
        self.assertEqual(
            mock_create_raw.call_args.kwargs["include_import_prefixes"], prefixes
        )

    @patch(f"{_ASSEMBLER_MODULE}.CustomTritonPackager.create_model_package")
    @patch(f"{_ASSEMBLER_MODULE}.CustomTritonPackager.create_raw_model_package")
    def test_passes_empty_include_import_prefixes(
        self, mock_create_raw, mock_create_model
    ):
        """An empty (but non-``None``) prefix list is passed through as-is."""
        mock_create_model.side_effect = _fake_create_package("deployable")
        mock_create_raw.side_effect = _fake_create_package("raw")

        config = TabularAssemblerConfig(
            custom=CustomAssemblerConfig(include_import_prefixes=[])
        )
        raw_model = ModelArtifact(
            path=self._upload_raw_model_source(),
            metadata=ModelMetadata(
                model_class=_CUSTOM_MODEL_CLASS, schema=_make_schema(), sample_data=[{}]
            ),
        )

        custom_assembler(config, raw_model, storage_backend=self.storage_backend)

        self.assertEqual(
            mock_create_model.call_args.kwargs["include_import_prefixes"], []
        )
        self.assertEqual(
            mock_create_raw.call_args.kwargs["include_import_prefixes"], []
        )

    @patch(f"{_ASSEMBLER_MODULE}.download_file_tree")
    @patch(f"{_ASSEMBLER_MODULE}.CustomTritonPackager.create_model_package")
    @patch(f"{_ASSEMBLER_MODULE}.CustomTritonPackager.create_raw_model_package")
    def test_native_transform_combined_layout(
        self, mock_create_raw, mock_create_model, mock_download_file_tree
    ):
        """Native-transform models fuse their schema/sample_data into one package."""
        mock_create_model.side_effect = _fake_create_package("deployable")
        mock_create_raw.side_effect = _fake_create_package("raw")

        tx_schema = ModelSchema(
            input_schema=[
                ModelSchemaItem(name="a", data_type=DataType.FLOAT, shape=[1])
            ],
            output_schema=[
                ModelSchemaItem(name="a_tx", data_type=DataType.FLOAT, shape=[1])
            ],
        )
        pred_schema = ModelSchema(
            input_schema=[
                ModelSchemaItem(name="a", data_type=DataType.FLOAT, shape=[1]),
                ModelSchemaItem(name="a_tx", data_type=DataType.FLOAT, shape=[1]),
            ],
            output_schema=[
                ModelSchemaItem(name="out", data_type=DataType.FLOAT, shape=[1])
            ],
        )
        native_tx = ModelArtifact(
            path="mem://tx/path",
            metadata=ModelMetadata(
                schema=tx_schema,
                sample_data=[{"a": np.array([0.0], dtype=np.float32)}],
            ),
        )
        config = TabularAssemblerConfig()
        raw_model = ModelArtifact(
            path="mem://pred/path",
            metadata=ModelMetadata(
                model_class=_CUSTOM_MODEL_CLASS, schema=pred_schema, sample_data=[{}]
            ),
        )

        custom_assembler(
            config,
            raw_model,
            native_transform_model=native_tx,
            storage_backend=self.storage_backend,
        )

        self.assertEqual(mock_download_file_tree.call_count, 2)
        pkg_kw = mock_create_model.call_args.kwargs
        raw_kw = mock_create_raw.call_args.kwargs
        model_path = mock_create_model.call_args.args[0]
        self.assertTrue(model_path.replace("\\", "/").endswith("combined_model"))
        self.assertEqual(model_path, mock_create_raw.call_args.args[0])
        self.assertEqual(pkg_kw["model_path_source_type"], StorageType.LOCAL)
        self.assertEqual(raw_kw["model_path_source_type"], StorageType.LOCAL)
        fused = fuse_model_schema(tx_schema, pred_schema)
        self.assertEqual(pkg_kw["model_schema"].input_schema, fused.input_schema)
        self.assertEqual(pkg_kw["model_schema"].output_schema, fused.output_schema)
        self.assertEqual(raw_kw["sample_data"][0]["a"].tolist(), [0.0])

    @patch(f"{_ASSEMBLER_MODULE}.CustomTritonPackager.create_model_package")
    @patch(f"{_ASSEMBLER_MODULE}.CustomTritonPackager.create_raw_model_package")
    def test_prefers_config_model_class_when_importable(
        self, mock_create_raw, mock_create_model
    ):
        """An importable ``config.model_class`` overrides the metadata class."""
        mock_create_model.side_effect = _fake_create_package("deployable")
        mock_create_raw.side_effect = _fake_create_package("raw")

        config = TabularAssemblerConfig(model_class=_CUSTOM_MODEL_CLASS)
        raw_model = ModelArtifact(
            path=self._upload_raw_model_source(),
            metadata=ModelMetadata(
                model_class="metadata.should.NotBeUsed",
                schema=_make_schema(),
                sample_data=[{}],
            ),
        )

        custom_assembler(config, raw_model, storage_backend=self.storage_backend)

        self.assertEqual(
            mock_create_model.call_args.kwargs["model_class"], _CUSTOM_MODEL_CLASS
        )

    @patch(f"{_ASSEMBLER_MODULE}.CustomTritonPackager.create_model_package")
    @patch(f"{_ASSEMBLER_MODULE}.CustomTritonPackager.create_raw_model_package")
    def test_falls_back_to_metadata_model_class_when_config_invalid(
        self, mock_create_raw, mock_create_model
    ):
        """An unimportable ``config.model_class`` falls back to the metadata class."""
        mock_create_model.side_effect = _fake_create_package("deployable")
        mock_create_raw.side_effect = _fake_create_package("raw")

        config = TabularAssemblerConfig(model_class="nonexistent_xyz.module.Class")
        raw_model = ModelArtifact(
            path=self._upload_raw_model_source(),
            metadata=ModelMetadata(
                model_class=_CUSTOM_MODEL_CLASS, schema=_make_schema(), sample_data=[{}]
            ),
        )

        custom_assembler(config, raw_model, storage_backend=self.storage_backend)

        self.assertEqual(
            mock_create_model.call_args.kwargs["model_class"], _CUSTOM_MODEL_CLASS
        )


if __name__ == "__main__":
    unittest.main()
