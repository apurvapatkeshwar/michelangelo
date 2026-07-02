"""Unit tests for ``...tabular_assembler.torch.assembler``."""

from __future__ import annotations

import os
import tempfile
import unittest
import uuid
from unittest.mock import MagicMock, patch

import numpy as np
import torch
import torch.nn as nn

from michelangelo.lib.artifact_manager.storage_backend import LocalStorageBackend
from michelangelo.lib.model_manager.constants import StorageType
from michelangelo.lib.model_manager.schema import DataType, ModelSchema, ModelSchemaItem
from michelangelo.workflow.schema.assembler import (
    TabularAssemblerConfig,
    TorchAssemblerConfig,
)
from michelangelo.workflow.tasks.tabular_assembler.torch.assembler import (
    _reorder_output_schema,
    torch_assembler,
)
from michelangelo.workflow.variables.metadata import ModelMetadata
from michelangelo.workflow.variables.types import ModelArtifact

_ASSEMBLER_MODULE = "michelangelo.workflow.tasks.tabular_assembler.torch.assembler"


class _E2EPredictor(nn.Module):
    """Tiny real predictor for an end-to-end (real files, no mocks) assembler test."""

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Sum the last dimension, matching ``_make_schema``'s ``input``/``output``."""
        return input.sum(dim=-1, keepdim=True)


class _E2ETxModule(nn.Module):
    """Tiny real native-transform module for an end-to-end assembler test."""

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Double ``tx_in``, matching ``_native_tx_schema``'s input/output names."""
        return {"pred_in": inputs["tx_in"] * 2.0}


def _make_schema() -> ModelSchema:
    return ModelSchema(
        input_schema=[
            ModelSchemaItem(name="input", data_type=DataType.FLOAT, shape=[2]),
        ],
        output_schema=[
            ModelSchemaItem(name="output", data_type=DataType.FLOAT, shape=[1]),
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


class _LocalBackendTestCase(unittest.TestCase):
    """Shared ``LocalStorageBackend``-per-test fixture for the tests below."""

    def setUp(self) -> None:
        """Create a fresh ``LocalStorageBackend`` rooted at a temp dir per test."""
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.storage_backend = LocalStorageBackend(self._tmp.name)

    def _upload_raw_model_source(self, contents: bytes = b"weights") -> str:
        """Create a local source file and upload it, returning a backend URI."""
        src = os.path.join(tempfile.mkdtemp(dir=self._tmp.name), "model.pt")
        with open(src, "wb") as f:
            f.write(contents)
        return self.storage_backend.upload(src, f"sources/{os.path.basename(src)}")


class TorchAssemblerTest(_LocalBackendTestCase):
    """Tests for ``torch_assembler``'s plain (no native-transform) path."""

    def _upload_real_module_source(
        self, module: nn.Module, *, as_state_dict: bool = True
    ) -> str:
        """Save a real module (state_dict or full module) and upload it.

        Args:
            module: The module to persist.
            as_state_dict: Save ``module.state_dict()`` (the usual predictor
                format) when ``True``; save the full module object (the
                format ``fuse_models_to_python`` expects for the
                native-transform side) when ``False``.
        """
        src = os.path.join(tempfile.mkdtemp(dir=self._tmp.name), "model.pt")
        torch.save(module.state_dict() if as_state_dict else module, src)
        return self.storage_backend.upload(src, f"sources/{uuid.uuid4().hex}.pt")

    @patch(f"{_ASSEMBLER_MODULE}.TorchTritonPackager.create_model_package")
    @patch(f"{_ASSEMBLER_MODULE}.TorchTritonPackager.create_raw_model_package")
    def test_torch_assembler_basic(self, mock_create_raw, mock_create_model):
        """Deployable/raw metadata reflect the source model."""
        mock_create_model.side_effect = _fake_create_package("deployable")
        mock_create_raw.side_effect = _fake_create_package("raw")

        config = TabularAssemblerConfig()
        sample_data = [{"input": np.array([[1.0, 2.0]])}]
        raw_model = ModelArtifact(
            path=self._upload_raw_model_source(),
            metadata=ModelMetadata(
                model_class="test.SimpleTorchModel",
                hyperparameters={"input_dim": 2, "output_dim": 1},
                schema=_make_schema(),
                sample_data=sample_data,
                is_incremental_training=True,
                baseline_model_identifier="baseline-model-v1",
            ),
        )

        assembled = torch_assembler(
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

        self.assertTrue(os.path.exists(assembled.deployable_model.path))
        self.assertTrue(os.path.exists(assembled.raw_model.path))

        mock_create_model.assert_called_once()
        kwargs = mock_create_model.call_args.kwargs
        self.assertIsNone(kwargs["backend"])
        self.assertEqual(kwargs["model_path_source_type"], StorageType.LOCAL)

        raw_kwargs = mock_create_raw.call_args.kwargs
        self.assertEqual(raw_kwargs["model_class"], "test.SimpleTorchModel")
        self.assertEqual(
            raw_kwargs["hyperparameters"], {"input_dim": 2, "output_dim": 1}
        )
        self.assertEqual(raw_kwargs["model_path_source_type"], StorageType.LOCAL)
        self.assertIsNone(raw_kwargs["transform_spec"])
        self.assertIsNone(raw_kwargs["transform_feature_stats"])

    @patch(f"{_ASSEMBLER_MODULE}.TorchTritonPackager.create_model_package")
    @patch(f"{_ASSEMBLER_MODULE}.TorchTritonPackager.create_raw_model_package")
    def test_downloads_raw_model_locally_before_packaging(
        self, mock_create_raw, mock_create_model
    ):
        """The packager only understands local paths, so the source must be local."""
        observed: dict[str, object] = {}

        def _observe_and_package(model_path, *, dest_model_path=None, **kwargs):
            with open(model_path, "rb") as f:
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
                model_class="test.SimpleTorchModel", schema=_make_schema()
            ),
        )

        torch_assembler(config, raw_model, storage_backend=self.storage_backend)

        self.assertEqual(observed["contents"], b"weights-xyz")
        self.assertEqual(observed["model_path_source_type"], StorageType.LOCAL)

    @patch(f"{_ASSEMBLER_MODULE}.TorchTritonPackager.create_model_package")
    @patch(f"{_ASSEMBLER_MODULE}.TorchTritonPackager.create_raw_model_package")
    def test_missing_hyperparameters_default_to_empty_dict(
        self, mock_create_raw, mock_create_model
    ):
        """``hyperparameters=None`` on the metadata is normalized to ``{}``."""
        mock_create_model.side_effect = _fake_create_package("deployable")
        mock_create_raw.side_effect = _fake_create_package("raw")

        config = TabularAssemblerConfig()
        raw_model = ModelArtifact(
            path=self._upload_raw_model_source(),
            metadata=ModelMetadata(
                model_class="test.SimpleTorchModel",
                schema=ModelSchema(),
                sample_data=[],
            ),
        )

        torch_assembler(config, raw_model, storage_backend=self.storage_backend)

        self.assertEqual(mock_create_raw.call_args.kwargs["hyperparameters"], {})

    @patch(f"{_ASSEMBLER_MODULE}.TorchTritonPackager.create_model_package")
    @patch(f"{_ASSEMBLER_MODULE}.TorchTritonPackager.create_raw_model_package")
    def test_backend_threaded_to_packager(self, mock_create_raw, mock_create_model):
        """``config.torch.backend`` reaches ``create_model_package``."""
        mock_create_model.side_effect = _fake_create_package("deployable")
        mock_create_raw.side_effect = _fake_create_package("raw")

        config = TabularAssemblerConfig(torch=TorchAssemblerConfig(backend="python"))
        raw_model = ModelArtifact(
            path=self._upload_raw_model_source(),
            metadata=ModelMetadata(
                model_class="test.SimpleTorchModel",
                schema=_make_schema(),
                sample_data=[],
            ),
        )

        torch_assembler(config, raw_model, storage_backend=self.storage_backend)

        self.assertEqual(mock_create_model.call_args.kwargs["backend"], "python")

    def test_native_transform_raises_not_implemented_pending_native_transform(self):
        """Real fusion now runs (model_fuser landed); native_transform is still needed.

        ``torch/assembler.py``'s native-transform branch resolves its
        ``model_fuser.fuse`` import (bucket E landed it), so this no longer
        fails at the import stub in ``_model_fuser_functions``. It reaches
        real fusion code and still raises ``NotImplementedError`` -- now from
        ``fuse_models_to_python``'s ``_build_tx_hydra_spec`` call, which is
        gated on the native-transform package (migration bucket "PR F"),
        which hasn't landed yet. Real (non-garbage) model files are used here
        so the ``NotImplementedError`` is unambiguously coming from that gate
        and not an incidental file-format error.
        """
        config = TabularAssemblerConfig()
        raw_model = ModelArtifact(
            path=self._upload_real_module_source(_E2EPredictor()),
            metadata=ModelMetadata(
                model_class=f"{__name__}._E2EPredictor",
                schema=_make_schema(),
                sample_data=[{"input": np.array([1.0, 2.0])}],
            ),
        )
        native_tx = ModelArtifact(
            path=self._upload_real_module_source(_E2ETxModule(), as_state_dict=False),
            metadata=ModelMetadata(
                model_class=f"{__name__}._E2ETxModule",
                schema=_native_tx_schema(),
                sample_data=[{"tx_in": np.array([1.0])}],
            ),
        )

        with self.assertRaises(NotImplementedError) as ctx:
            torch_assembler(
                config,
                raw_model,
                native_transform_model=native_tx,
                storage_backend=self.storage_backend,
            )
        self.assertIn("native-transform", str(ctx.exception))


def _native_tx_schema() -> ModelSchema:
    return ModelSchema(
        input_schema=[
            ModelSchemaItem(name="tx_in", data_type=DataType.FLOAT, shape=[1]),
        ],
        output_schema=[
            ModelSchemaItem(name="pred_in", data_type=DataType.FLOAT, shape=[1]),
        ],
    )


class NativeTransformFusionTest(_LocalBackendTestCase):
    """Tests for ``torch_assembler``'s native-transform fusion branch.

    ``_model_fuser_functions`` is patched directly (rather than relying on
    ``torch.model_fuser.fuse``, which lands in a follow-up change) so this
    branch's control flow — backend selection, output reordering, fused
    schema/sample-data construction, and transform metadata propagation —
    is exercised now.
    """

    def _make_artifacts(self):
        raw_model = ModelArtifact(
            path=self._upload_raw_model_source(),
            metadata=ModelMetadata(
                model_class="test.SimpleTorchModel",
                hyperparameters={"input_dim": 1},
                schema=ModelSchema(
                    input_schema=[
                        ModelSchemaItem(
                            name="pred_in", data_type=DataType.FLOAT, shape=[1]
                        ),
                    ],
                    output_schema=[
                        ModelSchemaItem(
                            name="a_out", data_type=DataType.FLOAT, shape=[1]
                        ),
                        ModelSchemaItem(
                            name="b_out", data_type=DataType.FLOAT, shape=[1]
                        ),
                    ],
                ),
                sample_data=[{"pred_in": np.array([1.0])}],
            ),
        )
        native_tx = ModelArtifact(
            path=self._upload_raw_model_source(contents=b"tx-weights"),
            metadata=ModelMetadata(
                model_class="test.TxModel",
                schema=_native_tx_schema(),
                sample_data=[{"tx_in": np.array([1.0])}],
                transform_spec={"transform_specs": [{"name": "Scale"}]},
                feature_stats={"tx_in": {"mean": 0.5}},
            ),
        )
        return raw_model, native_tx

    def _patch_fuser(
        self,
        fused_model_class="test.FusedModel",
        fused_hyperparameters=None,
        field_order=None,
    ):
        mock_fuse_python = MagicMock(
            return_value=(
                os.path.join(self._tmp.name, "fused_model.pt"),
                fused_model_class,
                fused_hyperparameters or {},
            )
        )
        mock_fuse_onnx = MagicMock()
        mock_fuse_torchscript = MagicMock()
        mock_field_order = MagicMock(return_value=field_order)
        mock_sample_data = MagicMock(return_value=[{"tx_in": np.array([1.0])}])
        return patch(
            f"{_ASSEMBLER_MODULE}._model_fuser_functions",
            return_value=(
                mock_fuse_onnx,
                mock_fuse_python,
                mock_fuse_torchscript,
                mock_field_order,
                mock_sample_data,
            ),
        ), (mock_fuse_onnx, mock_fuse_python, mock_fuse_torchscript, mock_field_order)

    @patch(f"{_ASSEMBLER_MODULE}.TorchTritonPackager.create_model_package")
    @patch(f"{_ASSEMBLER_MODULE}.TorchTritonPackager.create_raw_model_package")
    def test_python_backend_fuses_and_reuses_fused_pt_as_deployable(
        self, mock_create_raw, mock_create_model
    ):
        """Python backend: deployable reuses the fused ``.pt``; no separate export."""
        mock_create_model.side_effect = _fake_create_package("deployable")
        mock_create_raw.side_effect = _fake_create_package("raw")
        raw_model, native_tx = self._make_artifacts()
        patcher, (mock_onnx, mock_python, mock_ts, _mock_order) = self._patch_fuser(
            fused_model_class="test.FusedModel",
            fused_hyperparameters={"predictor_module": {}, "transform_module": {}},
        )

        with patcher:
            config = TabularAssemblerConfig(
                torch=TorchAssemblerConfig(backend="python")
            )
            assembled = torch_assembler(
                config,
                raw_model,
                native_transform_model=native_tx,
                storage_backend=self.storage_backend,
            )

        mock_python.assert_called_once()
        mock_onnx.assert_not_called()
        mock_ts.assert_not_called()

        pkg_kwargs = mock_create_model.call_args.kwargs
        self.assertEqual(pkg_kwargs["backend"], "python")
        self.assertEqual(pkg_kwargs["model_class"], "test.FusedModel")
        self.assertEqual(
            pkg_kwargs["hyperparameters"],
            {"predictor_module": {}, "transform_module": {}},
        )

        fused_schema = pkg_kwargs["model_schema"]
        input_names = [item.name for item in fused_schema.input_schema]
        self.assertIn("tx_in", input_names)
        self.assertNotIn("pred_in", input_names, "tx output must not be an input")
        self.assertEqual(assembled.deployable_model.metadata.schema, fused_schema)
        self.assertEqual(assembled.raw_model.metadata.schema, fused_schema)

        raw_kwargs = mock_create_raw.call_args.kwargs
        self.assertEqual(
            raw_kwargs["transform_spec"], native_tx.metadata.transform_spec
        )
        self.assertEqual(
            raw_kwargs["transform_feature_stats"], native_tx.metadata.feature_stats
        )

    @patch(f"{_ASSEMBLER_MODULE}.TorchTritonPackager.create_model_package")
    @patch(f"{_ASSEMBLER_MODULE}.TorchTritonPackager.create_raw_model_package")
    def test_onnx_backend_uses_fuse_onnx(self, mock_create_raw, mock_create_model):
        """``onnxruntime`` backend exports via ``fuse_models_to_onnx``."""
        mock_create_model.side_effect = _fake_create_package("deployable")
        mock_create_raw.side_effect = _fake_create_package("raw")
        raw_model, native_tx = self._make_artifacts()
        patcher, (mock_onnx, mock_python, mock_ts, _mock_order) = self._patch_fuser()

        with patcher:
            config = TabularAssemblerConfig(
                torch=TorchAssemblerConfig(backend="onnxruntime")
            )
            torch_assembler(
                config,
                raw_model,
                native_transform_model=native_tx,
                storage_backend=self.storage_backend,
            )

        mock_python.assert_called_once()
        mock_onnx.assert_called_once()
        mock_ts.assert_not_called()
        kwargs = mock_create_model.call_args.kwargs
        self.assertEqual(kwargs["backend"], "onnxruntime")
        model_path = mock_create_model.call_args.args[0]
        self.assertIn("fused_model.onnx", model_path)

    @patch(f"{_ASSEMBLER_MODULE}.TorchTritonPackager.create_model_package")
    @patch(f"{_ASSEMBLER_MODULE}.TorchTritonPackager.create_raw_model_package")
    def test_default_backend_uses_fuse_torchscript(
        self, mock_create_raw, mock_create_model
    ):
        """No (or ``pytorch``) backend exports the fused model via TorchScript."""
        mock_create_model.side_effect = _fake_create_package("deployable")
        mock_create_raw.side_effect = _fake_create_package("raw")
        raw_model, native_tx = self._make_artifacts()
        patcher, (mock_onnx, mock_python, mock_ts, _mock_order) = self._patch_fuser()

        with patcher:
            torch_assembler(
                TabularAssemblerConfig(),
                raw_model,
                native_transform_model=native_tx,
                storage_backend=self.storage_backend,
            )

        mock_python.assert_called_once()
        mock_ts.assert_called_once()
        mock_onnx.assert_not_called()

    @patch(f"{_ASSEMBLER_MODULE}.TorchTritonPackager.create_model_package")
    @patch(f"{_ASSEMBLER_MODULE}.TorchTritonPackager.create_raw_model_package")
    def test_fused_output_schema_reordered_by_predictor_field_order(
        self, mock_create_raw, mock_create_model
    ):
        """Output schema passed to the packager follows the predictor's field order."""
        mock_create_model.side_effect = _fake_create_package("deployable")
        mock_create_raw.side_effect = _fake_create_package("raw")
        raw_model, native_tx = self._make_artifacts()
        patcher, _mocks = self._patch_fuser(field_order=["b_out", "a_out"])

        with patcher:
            torch_assembler(
                TabularAssemblerConfig(),
                raw_model,
                native_transform_model=native_tx,
                storage_backend=self.storage_backend,
            )

        fused_schema = mock_create_model.call_args.kwargs["model_schema"]
        self.assertEqual(
            [item.name for item in fused_schema.output_schema], ["b_out", "a_out"]
        )

    @patch(f"{_ASSEMBLER_MODULE}.TorchTritonPackager.create_model_package")
    @patch(f"{_ASSEMBLER_MODULE}.TorchTritonPackager.create_raw_model_package")
    def test_fused_output_schema_keeps_order_when_field_order_unavailable(
        self, mock_create_raw, mock_create_model
    ):
        """A ``None`` field order leaves the predictor's original output order."""
        mock_create_model.side_effect = _fake_create_package("deployable")
        mock_create_raw.side_effect = _fake_create_package("raw")
        raw_model, native_tx = self._make_artifacts()
        patcher, _mocks = self._patch_fuser(field_order=None)

        with patcher:
            torch_assembler(
                TabularAssemblerConfig(),
                raw_model,
                native_transform_model=native_tx,
                storage_backend=self.storage_backend,
            )

        fused_schema = mock_create_model.call_args.kwargs["model_schema"]
        self.assertEqual(
            [item.name for item in fused_schema.output_schema], ["a_out", "b_out"]
        )


class ReorderOutputSchemaTest(unittest.TestCase):
    """Tests for ``_reorder_output_schema``."""

    def _schema(self) -> ModelSchema:
        return ModelSchema(
            input_schema=[
                ModelSchemaItem(name="in", data_type=DataType.FLOAT, shape=[1])
            ],
            output_schema=[
                ModelSchemaItem(name="a_out", data_type=DataType.FLOAT, shape=[1]),
                ModelSchemaItem(name="b_out", data_type=DataType.FLOAT, shape=[1]),
            ],
        )

    def test_none_field_order_returns_schema_unchanged(self):
        """``field_order=None`` returns the exact same schema instance."""
        schema = self._schema()
        self.assertIs(_reorder_output_schema(schema, None), schema)

    def test_reorders_to_match_field_order(self):
        """Output fields are reordered to match ``field_order``; input is untouched."""
        schema = self._schema()
        reordered = _reorder_output_schema(schema, ["b_out", "a_out"])
        self.assertEqual(
            [item.name for item in reordered.output_schema], ["b_out", "a_out"]
        )
        self.assertEqual(reordered.input_schema, schema.input_schema)

    def test_fields_not_in_order_are_appended(self):
        """Output fields absent from ``field_order`` are appended at the end."""
        schema = self._schema()
        reordered = _reorder_output_schema(schema, ["b_out"])
        self.assertEqual(
            [item.name for item in reordered.output_schema], ["b_out", "a_out"]
        )

    def test_unknown_field_order_entries_are_ignored(self):
        """Names in ``field_order`` that aren't in the schema are silently skipped."""
        schema = self._schema()
        reordered = _reorder_output_schema(schema, ["nonexistent", "b_out"])
        self.assertEqual(
            [item.name for item in reordered.output_schema], ["b_out", "a_out"]
        )

    def test_empty_field_order_appends_all_original_fields(self):
        """An empty (but non-``None``) ``field_order`` keeps the original order."""
        schema = self._schema()
        reordered = _reorder_output_schema(schema, [])
        self.assertEqual(
            [item.name for item in reordered.output_schema], ["a_out", "b_out"]
        )


if __name__ == "__main__":
    unittest.main()
