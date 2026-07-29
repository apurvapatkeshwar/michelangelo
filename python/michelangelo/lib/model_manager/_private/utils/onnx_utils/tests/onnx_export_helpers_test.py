"""Tests for the private ONNX export helper functions.

Moved out of ``model_fuser/tests/fuse_test.py`` when the ONNX export
mechanics were consolidated into this shared module (mirroring internal's
``ea07350f69c`` "Decouple ONNX conversion from model fuser into shared
torch_onnx util"), so both the model fuser and the non-fused Triton packager
share one tested implementation instead of two independently-drifting ones.
``model_fuser``'s ``FuseModelsToOnnxTest`` and the packager's
``onnx_conversion_test.py`` keep their own integration tests exercising
these helpers through their respective public entry points.
"""

from __future__ import annotations

import os
import tempfile
from types import SimpleNamespace
from typing import NamedTuple
from unittest import TestCase, mock

import numpy as np
import onnx
import torch
import torch.nn as nn

from michelangelo.lib.model_manager._private.utils.onnx_utils import (
    onnx_export_helpers as helpers_module,
)
from michelangelo.lib.model_manager._private.utils.onnx_utils.onnx_export_helpers import (  # noqa: E501
    OnnxDynamoTupleWrapper,
    OnnxTupleWrapper,
    disable_transformer_encoder_fastpath_for_onnx,
    expand_batch_for_onnx_export,
    force_onnx_io_shapes_from_schema,
    onnx_dynamo_dynamic_shapes_for_tuple_arg,
    onnx_dynamo_export_error_should_retry_legacy,
    onnx_dynamo_exporter_dependencies_available,
    onnx_export_attach_inputs_to_output,
    run_export_with_retry,
)
from michelangelo.lib.model_manager.schema import DataType, ModelSchema, ModelSchemaItem


class _DictModule(nn.Module):
    """Predictor accepting a dict, used to exercise the tuple wrappers."""

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        return inputs["a"] + inputs["b"]


class ExpandBatchForOnnxExportTest(TestCase):
    """Tests for ``expand_batch_for_onnx_export``."""

    def test_repeats_batch_one_to_two(self):
        """Repeats batch one to two."""
        a = torch.zeros(1, 3)
        b = torch.ones(1, 1)
        out = expand_batch_for_onnx_export((a, b))
        self.assertEqual(out[0].shape, (2, 3))
        self.assertEqual(out[1].shape, (2, 1))

    def test_leaves_batch_greater_than_one_unchanged(self):
        """Leaves batch greater than one unchanged."""
        a = torch.randn(3, 2)
        b = torch.randn(1, 2)
        out = expand_batch_for_onnx_export((a, b))
        self.assertIs(out[0], a)
        self.assertEqual(out[1].shape[0], 2)


class DisableTransformerEncoderFastpathForOnnxTest(TestCase):
    """Tests for ``disable_transformer_encoder_fastpath_for_onnx``."""

    def test_noop_when_mha_backend_missing(self):
        """Noop when mha backend missing."""
        root = nn.Linear(1, 1)
        root.eval()
        with mock.patch.object(helpers_module.torch, "backends", SimpleNamespace()):
            with disable_transformer_encoder_fastpath_for_onnx(root):
                self.assertFalse(root.training)
            self.assertFalse(root.training)

    def test_noop_preserves_train_when_mha_backend_missing(self):
        """Noop preserves train when mha backend missing."""
        root = nn.Linear(1, 1)
        root.train()
        with mock.patch.object(helpers_module.torch, "backends", SimpleNamespace()):
            with disable_transformer_encoder_fastpath_for_onnx(root):
                self.assertTrue(root.training)
            self.assertTrue(root.training)

    def test_mha_setter_without_getter_defaults_prev_to_true(self):
        """Mha setter without getter defaults prev to true."""
        calls: list[bool] = []

        class _Mha:
            def set_fastpath_enabled(self, enabled: bool) -> None:
                calls.append(enabled)

        with mock.patch.object(
            helpers_module.torch.backends, "mha", _Mha(), create=True
        ):
            root = nn.Linear(1, 1)
            with disable_transformer_encoder_fastpath_for_onnx(root):
                self.assertEqual(calls, [False])
            self.assertEqual(calls, [False, True])


class OnnxDynamoExporterDepsTest(TestCase):
    """Tests for ``onnx_dynamo_exporter_dependencies_available``."""

    def test_true_when_onnxscript_importable(self):
        """True when onnxscript importable."""
        with mock.patch.object(
            helpers_module.importlib.util, "find_spec", return_value=object()
        ):
            self.assertTrue(onnx_dynamo_exporter_dependencies_available())

    def test_false_when_onnxscript_missing(self):
        """False when onnxscript missing."""
        with mock.patch.object(
            helpers_module.importlib.util, "find_spec", return_value=None
        ):
            self.assertFalse(onnx_dynamo_exporter_dependencies_available())


class OnnxDynamoDynamicShapesForTupleArgTest(TestCase):
    """Tests for ``onnx_dynamo_dynamic_shapes_for_tuple_arg``."""

    def test_returns_none_when_torch_export_dim_unavailable(self):
        """Returns none when torch export dim unavailable."""
        with mock.patch.object(helpers_module, "_TorchExportDim", None):
            self.assertIsNone(
                onnx_dynamo_dynamic_shapes_for_tuple_arg((torch.zeros(1),))
            )

    def test_returns_one_dim_dict_per_tensor(self):
        """Returns one dim dict per tensor."""
        result = onnx_dynamo_dynamic_shapes_for_tuple_arg(
            (torch.zeros(1, 2), torch.zeros(1, 3))
        )
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)
        per_tensor_dims = result[0]
        self.assertEqual(len(per_tensor_dims), 2)
        for dims in per_tensor_dims:
            self.assertEqual(set(dims.keys()), {0})


class OnnxDynamoExportErrorRetryPolicyTest(TestCase):
    """Tests for ``onnx_dynamo_export_error_should_retry_legacy``."""

    def test_import_errors_always_retry(self):
        """Import errors always retry."""
        self.assertTrue(onnx_dynamo_export_error_should_retry_legacy(ImportError("x")))
        self.assertTrue(
            onnx_dynamo_export_error_should_retry_legacy(ModuleNotFoundError("x"))
        )

    def test_known_dynamo_failure_signatures_retry(self):
        """Known dynamo failure signatures retry."""
        messages = [
            "onnxscript is required",
            "ConvertVersionPass failed",
            "version conversion pass error",
            "model contains functions that dynamo cannot trace",
            "torch.onnx._internal.exporter._errors.PassError",
            "failed to convert 'dynamic_axes' to dynamic_shapes",
            "TreeSpec.unflatten mismatch",
            "torch.onnx._internal.exporter._errors.TorchExportError: failed",
            "Unexpected dimension #0 in input tensor shape",
        ]
        for msg in messages:
            with self.subTest(msg=msg):
                self.assertTrue(
                    onnx_dynamo_export_error_should_retry_legacy(RuntimeError(msg))
                )

    def test_unrelated_error_does_not_retry(self):
        """Unrelated error does not retry."""
        self.assertFalse(
            onnx_dynamo_export_error_should_retry_legacy(RuntimeError("shape mismatch"))
        )


class RunExportWithRetryTest(TestCase):
    """Tests for ``run_export_with_retry``'s dynamo-failure re-raise branch."""

    def test_non_recoverable_dynamo_error_is_reraised_without_legacy_fallback(self):
        """A dynamo-export error outside the retry signatures propagates as-is.

        Proves the ``if not onnx_dynamo_export_error_should_retry_legacy(e): raise``
        branch actually re-raises instead of silently falling through to the
        legacy exporter or swallowing the error.
        """
        model = nn.Linear(1, 1)
        sample_args = (torch.zeros(1, 1),)

        with (
            tempfile.TemporaryDirectory() as d,
            mock.patch.object(
                torch.onnx, "export", side_effect=RuntimeError("shape mismatch")
            ) as mock_export,
        ):
            with self.assertRaisesRegex(RuntimeError, "shape mismatch"):
                run_export_with_retry(
                    export_args=(model, sample_args, os.path.join(d, "out.onnx")),
                    export_kwargs={"dynamic_shapes": {}},
                    legacy_export_kwargs={},
                    use_dynamo=True,
                    use_tuple_wrapper=True,
                    model=model,
                    input_key_order=["x"],
                )
            # Only the dynamo attempt ran; no legacy fallback export was attempted.
            self.assertEqual(mock_export.call_count, 1)


class OnnxExportAttachInputsToOutputTest(TestCase):
    """Tests for ``onnx_export_attach_inputs_to_output``."""

    def test_empty_inputs_returns_output_unchanged(self):
        """Empty inputs returns output unchanged."""
        out = torch.tensor([1.0])
        self.assertIs(onnx_export_attach_inputs_to_output(out, ()), out)

    def test_tensor_output_gains_zero_valued_dependency(self):
        """Tensor output gains zero valued dependency."""
        out = torch.tensor([5.0])
        inputs = (torch.tensor([1.0]), torch.tensor([2.0]))
        result = onnx_export_attach_inputs_to_output(out, inputs)
        torch.testing.assert_close(result, out)
        self.assertIsNot(result, out)

    def test_named_tuple_output_first_tensor_field_adjusted(self):
        """Named tuple output first tensor field adjusted."""

        class Out(NamedTuple):
            a: torch.Tensor
            b: torch.Tensor

        out = Out(a=torch.tensor([1.0]), b=torch.tensor([2.0]))
        result = onnx_export_attach_inputs_to_output(out, (torch.tensor([3.0]),))
        self.assertIsInstance(result, Out)
        torch.testing.assert_close(result.a, out.a)
        torch.testing.assert_close(result.b, out.b)

    def test_dict_output_first_tensor_value_adjusted(self):
        """Dict output first tensor value adjusted."""
        out = {"a": torch.tensor([1.0]), "b": torch.tensor([2.0])}
        result = onnx_export_attach_inputs_to_output(out, (torch.tensor([3.0]),))
        self.assertIsInstance(result, dict)
        torch.testing.assert_close(result["a"], out["a"])

    def test_tuple_output_first_tensor_element_adjusted(self):
        """Tuple output first tensor element adjusted."""
        out = (torch.tensor([1.0]), torch.tensor([2.0]))
        result = onnx_export_attach_inputs_to_output(out, (torch.tensor([3.0]),))
        self.assertIsInstance(result, tuple)
        torch.testing.assert_close(result[0], out[0])

    def test_unrecognized_output_type_returned_unchanged(self):
        """Unrecognized output type returned unchanged."""
        out = "not a tensor container"
        self.assertEqual(
            onnx_export_attach_inputs_to_output(out, (torch.tensor([1.0]),)), out
        )


class OnnxTupleWrapperTest(TestCase):
    """Tests for ``OnnxTupleWrapper``."""

    def test_forward_zips_keys_and_runs_inner(self):
        """Forward zips keys and runs inner."""
        wrapper = OnnxTupleWrapper(_DictModule(), ["a", "b"])
        with torch.no_grad():
            y = wrapper(torch.ones(1, 2), torch.ones(1, 2))
        self.assertTrue(torch.equal(y, torch.full((1, 2), 2.0)))


class OnnxDynamoTupleWrapperTest(TestCase):
    """Tests for ``OnnxDynamoTupleWrapper``."""

    def test_forward_takes_single_tuple_argument(self):
        """Forward takes single tuple argument."""
        wrapper = OnnxDynamoTupleWrapper(_DictModule(), ["a", "b"])
        with torch.no_grad():
            y = wrapper((torch.ones(1, 2), torch.ones(1, 2)))
        self.assertTrue(torch.equal(y, torch.full((1, 2), 2.0)))


class ForceOnnxIoShapesFromSchemaTest(TestCase):
    """Tests for ``force_onnx_io_shapes_from_schema``."""

    def _export_simple_onnx(self, dest_path: str) -> None:
        class M(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x * 2

        torch.onnx.export(
            M(),
            (torch.zeros(2, 32),),
            dest_path,
            input_names=["x"],
            output_names=["y"],
            dynamic_axes={"x": {0: "b"}, "y": {0: "b"}},
        )

    def test_overrides_dims_to_match_schema(self):
        """Overrides dims to match schema."""
        with tempfile.TemporaryDirectory() as d:
            dest_path = os.path.join(d, "m.onnx")
            self._export_simple_onnx(dest_path)
            schema = ModelSchema(
                input_schema=[
                    ModelSchemaItem(name="x", data_type=DataType.FLOAT, shape=[32])
                ],
                output_schema=[
                    ModelSchemaItem(name="y", data_type=DataType.FLOAT, shape=[32])
                ],
            )
            force_onnx_io_shapes_from_schema(dest_path, [schema])
            model_proto = onnx.load(dest_path)
            for value_info in list(model_proto.graph.input) + list(
                model_proto.graph.output
            ):
                dims = value_info.type.tensor_type.shape.dim
                self.assertEqual(dims[0].dim_param, "b")
                self.assertEqual(dims[1].dim_value, 32)

    def test_no_matching_schema_names_is_a_noop(self):
        """No matching schema names is a noop."""
        with tempfile.TemporaryDirectory() as d:
            dest_path = os.path.join(d, "m.onnx")
            self._export_simple_onnx(dest_path)
            before = onnx.load(dest_path).SerializeToString()
            force_onnx_io_shapes_from_schema(dest_path, [None])
            after = onnx.load(dest_path).SerializeToString()
            self.assertEqual(before, after)

    def test_rank_mismatch_is_skipped_with_warning(self):
        """Rank mismatch is skipped with warning."""
        with tempfile.TemporaryDirectory() as d:
            dest_path = os.path.join(d, "m.onnx")
            self._export_simple_onnx(dest_path)
            # Schema declares a 2D non-batch shape, but the exported graph is
            # rank-2 overall (batch + 1 dim) -- a rank mismatch that should be
            # skipped, not raise.
            schema = ModelSchema(
                input_schema=[
                    ModelSchemaItem(name="x", data_type=DataType.FLOAT, shape=[32, 32])
                ],
            )
            with self.assertLogs(helpers_module._logger, level="WARNING") as cm:
                force_onnx_io_shapes_from_schema(dest_path, [schema])
            self.assertTrue(
                any("Skipping ONNX shape override" in line for line in cm.output)
            )

    def test_external_data_model_shape_override_is_written_back(self):
        """Shape override survives on a model with an EXTERNAL-flagged initializer.

        Loading with ``onnx.load()`` (external data inlined) and writing back
        with ``onnx.save()``/``onnx.save_model()`` re-invokes the
        external_data_helper, which corrupts (or raises ``IsADirectoryError``
        on) a proto whose initializers already carry ``EXTERNAL`` references
        to sidecar files -- the exact failure mode a >2GB model hits. This
        proves the fix: loading with ``load_external_data=False`` and
        serializing the proto verbatim preserves the ``EXTERNAL`` reference
        while still writing back the shape override.
        """
        with tempfile.TemporaryDirectory() as d:
            dest_path = os.path.join(d, "m.onnx")
            self._export_simple_onnx(dest_path)

            model_proto = onnx.load(dest_path)
            weight = onnx.numpy_helper.from_array(
                np.zeros((2, 2), dtype="float32"), name="ext_w"
            )
            model_proto.graph.initializer.append(weight)
            onnx.save_model(model_proto, dest_path, save_as_external_data=False)

            # Flag the initializer as EXTERNAL without actually splitting it
            # into a sidecar file -- enough to reproduce the code path that
            # `force_onnx_io_shapes_from_schema` must handle safely.
            reloaded = onnx.load(dest_path, load_external_data=False)
            for initializer in reloaded.graph.initializer:
                if initializer.name == "ext_w":
                    initializer.data_location = onnx.TensorProto.EXTERNAL
            with open(dest_path, "wb") as f:
                f.write(reloaded.SerializeToString())

            schema = ModelSchema(
                input_schema=[
                    ModelSchemaItem(name="x", data_type=DataType.FLOAT, shape=[32])
                ],
                output_schema=[
                    ModelSchemaItem(name="y", data_type=DataType.FLOAT, shape=[32])
                ],
            )
            force_onnx_io_shapes_from_schema(dest_path, [schema])

            written_back = onnx.load(dest_path, load_external_data=False)
            for value_info in written_back.graph.input:
                if value_info.name == "x":
                    dims = value_info.type.tensor_type.shape.dim
                    self.assertEqual(dims[0].dim_param, "b")
                    self.assertEqual(dims[1].dim_value, 32)
            external = [
                initializer
                for initializer in written_back.graph.initializer
                if initializer.HasField("data_location")
                and initializer.data_location == onnx.TensorProto.EXTERNAL
            ]
            self.assertEqual(len(external), 1)
            self.assertEqual(external[0].name, "ext_w")
