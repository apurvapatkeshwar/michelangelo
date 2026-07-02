"""Tests for ``model_fuser.fuse``.

Landing order follows the migration plan's §3.7 strategy: non-native-transform,
non-ONNX tests first; ONNX tests next; native-transform-gated tests last
(present but skipped until native-transform support lands — see PR F).

Two internal test classes are intentionally **not** re-ported here:
``FuseModelSchemaTest``/``FuseInputSchemaTest`` (they exercise
``fuse_input_schema``/``fuse_model_schema``, which already have full coverage
in ``_private/schema/tests/fuse_test.py`` from bucket C) and
``FusedModelPythonBackendTest`` (exercises ``FusedModel.forward`` directly,
already covered by ``torch/model_fuser/tests/fused_model_test.py`` from
bucket D). Re-porting them here would just duplicate coverage of code this
bucket doesn't own.
"""

from __future__ import annotations

import inspect
import os
import tempfile
import unittest
from types import SimpleNamespace
from typing import NamedTuple
from unittest import mock

import onnx
import torch
import torch.nn as nn

from michelangelo.lib.model_manager.schema import DataType, ModelSchema, ModelSchemaItem
from michelangelo.workflow.tasks.tabular_assembler.torch.model_fuser import (
    fuse as fuse_module,
)
from michelangelo.workflow.tasks.tabular_assembler.torch.model_fuser import (
    fused_model,
)
from michelangelo.workflow.tasks.tabular_assembler.torch.model_fuser.fuse import (
    _build_fused_sample_input,
    _disable_transformer_encoder_fused_fastpath_for_onnx,
    _FusedOnnxDynamoTupleWrapper,
    _FusedOnnxTupleWrapper,
    _is_state_dict,
    _load_module_from_path,
    _onnx_dynamo_dynamic_shapes_for_tuple_arg,
    _onnx_dynamo_export_error_should_retry_legacy,
    _onnx_dynamo_exporter_dependencies_available,
    _onnx_export_attach_inputs_to_output,
    _schema_input_keys,
    _schema_output_keys,
    build_fused_sample_data,
    compute_python_fuse_metadata,
    fuse_models_to_onnx,
    fuse_models_to_python,
    fuse_models_to_torchscript,
    get_predictor_output_field_order,
)

FusedModel = fused_model.FusedModel

_FUSE_MODULE = "michelangelo.workflow.tasks.tabular_assembler.torch.model_fuser.fuse"


# ---------------------------------------------------------------------------
# Shared model fixtures
# ---------------------------------------------------------------------------


class _DictTransform(nn.Module):
    """Transform that accepts a dict and returns a dict."""

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {"out": next(iter(inputs.values())) + 1.0}


class _TensorPredictor(nn.Module):
    """Predictor accepting a single tensor named to match the transform's output key."""

    def forward(self, out: torch.Tensor) -> torch.Tensor:
        return out.sum(dim=-1, keepdim=True)


class _DictPredictor(nn.Module):
    """Predictor accepting a dict."""

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        return next(iter(inputs.values())).sum(dim=-1, keepdim=True)


class _MultiTensorPredictor(nn.Module):
    """Predictor accepting three tensors, in forward-order ``b, a, c``."""

    def forward(
        self, b: torch.Tensor, a: torch.Tensor, c: torch.Tensor
    ) -> torch.Tensor:
        return a + b + c


class _TwoParamPredictor(nn.Module):
    """Predictor accepting only ``a, b`` (for schema-not-subset tests)."""

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a + b


class _TransformerDictTransform(nn.Module):
    """Transform producing a ``[batch, seq, d_model]`` embedding for a predictor."""

    def __init__(self, d_model: int = 64, seq_len: int = 10) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.proj = nn.Linear(1, d_model * seq_len)

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        x = next(iter(inputs.values()))
        batch = x.size(0)
        out = self.proj(x).view(batch, self.seq_len, self.d_model)
        return {"emb": out}


class _TransformerPredictor(nn.Module):
    """Predictor using ``nn.TransformerEncoder``, accepting a dict with an ``emb`` key."""  # noqa: E501

    def __init__(
        self, d_model: int = 64, n_heads: int = 4, n_layers: int = 2, d_ff: int = 128
    ) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False,
        )
        self.head = nn.Linear(d_model, 1)

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        x = inputs["emb"]
        out = self.transformer(x)
        return self.head(out.mean(dim=1))


class _NamedTupleTensorPredictor(nn.Module):
    """Tensor-arg predictor with a NamedTuple output (fields ``b_out``, ``a_out``)."""

    class Output(NamedTuple):
        """Output fields, deliberately declared out of alphabetical order."""

        b_out: torch.Tensor
        a_out: torch.Tensor

    def forward(
        self, a: torch.Tensor, b: torch.Tensor
    ) -> _NamedTupleTensorPredictor.Output:
        return _NamedTupleTensorPredictor.Output(
            b_out=b.sum(-1, keepdim=True), a_out=a.sum(-1, keepdim=True)
        )


class _NamedTupleDictPredictor(nn.Module):
    """Dict-arg predictor with a NamedTuple output (fields ``z_out``, ``x_out``)."""

    class Output(NamedTuple):
        """Output fields, deliberately declared out of alphabetical order."""

        z_out: torch.Tensor
        x_out: torch.Tensor

    def forward(
        self, inputs: dict[str, torch.Tensor]
    ) -> _NamedTupleDictPredictor.Output:
        x = inputs["x"]
        z = inputs["z"]
        return _NamedTupleDictPredictor.Output(
            z_out=z.sum(-1, keepdim=True), x_out=x.sum(-1, keepdim=True)
        )


class _NoAnnotNTPredictor(nn.Module):
    """Predictor with no return annotation but a NamedTuple output."""

    class Out(NamedTuple):
        """Output fields ``c_out``, ``d_out``."""

        c_out: torch.Tensor
        d_out: torch.Tensor

    def forward(self, x):
        return _NoAnnotNTPredictor.Out(c_out=x, d_out=x * 2)


class _SingleFieldNTPredictor(nn.Module):
    """Predictor with no return annotation and a single-field NamedTuple output."""

    class Out(NamedTuple):
        """Output field ``e_out``."""

        e_out: torch.Tensor

    def forward(self, x):
        return _SingleFieldNTPredictor.Out(e_out=x)


class _Strategy1NT(NamedTuple):
    """NamedTuple used as a concrete ``forward()`` return-type annotation."""

    second: torch.Tensor
    first: torch.Tensor


class _Strategy1AnnotatedPredictor(nn.Module):
    """Predictor whose ``forward`` return annotation is a concrete NamedTuple type."""

    def forward(self, x: torch.Tensor) -> _Strategy1NT:
        return _Strategy1NT(second=x * 2, first=x)


class _PairTransform(nn.Module):
    """Transform producing predictor inputs ``a`` and ``b`` from a single feature."""

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        x = inputs["in"]
        return {"a": x, "b": x + 1.0}


def _class_path(cls: type) -> str:
    """Return the dotted import path for a test-fixture class defined in this module."""
    return f"{__name__}.{cls.__qualname__}"


# ===========================================================================
# Tranche 1: non-native-transform, non-ONNX
# ===========================================================================


class SchemaHelpersTest(unittest.TestCase):
    """Tests for ``_schema_input_keys`` and ``_schema_output_keys``."""

    def test_schema_input_keys_empty(self):
        """Schema input keys empty."""
        self.assertEqual(_schema_input_keys(None), [])
        self.assertEqual(_schema_input_keys(ModelSchema()), [])
        self.assertEqual(_schema_input_keys(ModelSchema(input_schema=[])), [])

    def test_schema_input_keys_order(self):
        """Schema input keys order."""
        schema = ModelSchema(
            input_schema=[
                ModelSchemaItem(name="a", data_type=DataType.FLOAT),
                ModelSchemaItem(name="b", data_type=DataType.INT),
            ]
        )
        self.assertEqual(_schema_input_keys(schema), ["a", "b"])

    def test_schema_output_keys_empty(self):
        """Schema output keys empty."""
        self.assertEqual(_schema_output_keys(None), [])
        self.assertEqual(_schema_output_keys(ModelSchema()), [])
        self.assertEqual(_schema_output_keys(ModelSchema(output_schema=[])), [])

    def test_schema_output_keys_order(self):
        """Schema output keys order."""
        schema = ModelSchema(
            output_schema=[
                ModelSchemaItem(name="x", data_type=DataType.FLOAT),
                ModelSchemaItem(name="y", data_type=DataType.INT),
            ]
        )
        self.assertEqual(_schema_output_keys(schema), ["x", "y"])


class ForwardAcceptsDictTest(unittest.TestCase):
    """Tests for ``_forward_accepts_dict``."""

    def test_accepts_dict_true(self):
        """Accepts dict true.

        Also covers the stringized-annotation form: this module has
        ``from __future__ import annotations``, so every annotation here
        (including this one) is already stringized.
        """

        class M(nn.Module):
            def forward(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
                return next(iter(x.values()))

        self.assertTrue(fuse_module._forward_accepts_dict(M()))

    def test_accepts_dict_false_single_tensor(self):
        """Accepts dict false single tensor."""

        class M(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x

        self.assertFalse(fuse_module._forward_accepts_dict(M()))

    def test_accepts_dict_false_no_annotation(self):
        """Accepts dict false no annotation."""

        class M(nn.Module):
            def forward(self, x):
                return x

        self.assertFalse(fuse_module._forward_accepts_dict(M()))

    def test_accepts_dict_false_forward_no_params(self):
        """Accepts dict false forward no params."""

        class M(nn.Module):
            def forward(self) -> torch.Tensor:
                return torch.tensor(0.0)

        self.assertFalse(fuse_module._forward_accepts_dict(M()))

    def test_accepts_dict_false_when_signature_raises(self):
        """Accepts dict false when signature raises."""
        m = _TensorPredictor()
        with mock.patch(f"{_FUSE_MODULE}.inspect.signature") as sig:
            sig.side_effect = ValueError("cannot inspect")
            self.assertFalse(fuse_module._forward_accepts_dict(m))
        with mock.patch(f"{_FUSE_MODULE}.inspect.signature") as sig:
            sig.side_effect = TypeError("cannot inspect")
            self.assertFalse(fuse_module._forward_accepts_dict(m))

    def test_accepts_dict_false_non_dict_annotation_after_self(self):
        """Accepts dict false non dict annotation after self."""

        class M(nn.Module):
            def forward(self, x: int) -> torch.Tensor:
                return torch.tensor(float(x))

        self.assertFalse(fuse_module._forward_accepts_dict(M()))

    def test_accepts_dict_continue_when_param_is_self(self):
        """A signature that still lists ``self`` exercises the ``continue`` branch."""
        m = _TensorPredictor()
        self_param = inspect.Parameter(
            "self",
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            annotation=inspect.Parameter.empty,
        )
        x_param = inspect.Parameter(
            "x", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=dict
        )
        fake_sig = inspect.Signature(parameters=[self_param, x_param])
        with mock.patch(f"{_FUSE_MODULE}.inspect.signature", return_value=fake_sig):
            self.assertTrue(fuse_module._forward_accepts_dict(m))


class ForwardParamOrderTest(unittest.TestCase):
    """Tests for ``_forward_param_order``."""

    def test_returns_param_names_excluding_self(self):
        """Returns param names excluding self."""
        self.assertEqual(fuse_module._forward_param_order(_TensorPredictor()), ["out"])
        self.assertEqual(fuse_module._forward_param_order(_DictPredictor()), ["inputs"])

    def test_returns_multiple_params_in_signature_order(self):
        """Returns multiple params in signature order."""

        class M(nn.Module):
            def forward(
                self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
            ) -> torch.Tensor:
                return a + b + c

        self.assertEqual(fuse_module._forward_param_order(M()), ["a", "b", "c"])

    def test_returns_empty_when_no_params_after_self(self):
        """Returns empty when no params after self."""

        class M(nn.Module):
            def forward(self) -> torch.Tensor:
                return torch.tensor(0.0)

        self.assertEqual(fuse_module._forward_param_order(M()), [])

    def test_returns_empty_when_signature_raises(self):
        """Returns empty when signature raises."""
        m = _TensorPredictor()
        with mock.patch(f"{_FUSE_MODULE}.inspect.signature") as sig:
            sig.side_effect = ValueError("cannot inspect")
            self.assertEqual(fuse_module._forward_param_order(m), [])
        with mock.patch(f"{_FUSE_MODULE}.inspect.signature") as sig:
            sig.side_effect = TypeError("cannot inspect")
            self.assertEqual(fuse_module._forward_param_order(m), [])


class BuildFusedSampleInputTest(unittest.TestCase):
    """Tests for ``_build_fused_sample_input``."""

    def test_empty_schemas_returns_empty_dict(self):
        """Empty schemas returns empty dict."""
        self.assertEqual(_build_fused_sample_input(None, None), {})
        self.assertEqual(_build_fused_sample_input(ModelSchema(), ModelSchema()), {})

    def test_returns_tensors_with_correct_shape_and_dtype(self):
        """Returns tensors with correct shape and dtype."""
        schema = ModelSchema(
            input_schema=[
                ModelSchemaItem(name="a", data_type=DataType.FLOAT, shape=[2]),
                ModelSchemaItem(name="b", data_type=DataType.LONG, shape=[1]),
            ]
        )
        out = _build_fused_sample_input(schema, None)
        self.assertEqual(set(out.keys()), {"a", "b"})
        self.assertEqual(out["a"].shape, (1, 2))
        self.assertEqual(out["a"].dtype, torch.float32)
        self.assertEqual(out["b"].shape, (1, 1))
        self.assertEqual(out["b"].dtype, torch.int64)

    def test_batch_size_parameter(self):
        """Batch size parameter."""
        schema = ModelSchema(
            input_schema=[
                ModelSchemaItem(name="x", data_type=DataType.FLOAT, shape=[1])
            ]
        )
        out = _build_fused_sample_input(schema, None, batch_size=4)
        self.assertEqual(out["x"].shape, (4, 1))

    def test_sample_tensors_device_matches_cuda_availability(self):
        """Sample tensors device matches cuda availability."""
        schema = ModelSchema(
            input_schema=[
                ModelSchemaItem(name="x", data_type=DataType.FLOAT, shape=[1])
            ]
        )
        out = _build_fused_sample_input(schema, None)
        expected_type = "cuda" if torch.cuda.is_available() else "cpu"
        self.assertTrue(all(t.device.type == expected_type for t in out.values()))

    def test_zero_or_empty_shape_uses_minimum_one(self):
        """Zero or empty shape uses minimum one."""
        schema = ModelSchema(
            input_schema=[
                ModelSchemaItem(name="a", data_type=DataType.FLOAT, shape=[0])
            ]
        )
        out = _build_fused_sample_input(schema, None)
        self.assertEqual(out["a"].shape, (1, 1))

    def test_keys_match_fused_input_schema(self):
        """Keys match fused input schema."""
        tx = ModelSchema(
            input_schema=[
                ModelSchemaItem(name="feat", data_type=DataType.FLOAT, shape=[1])
            ],
            output_schema=[
                ModelSchemaItem(name="emb", data_type=DataType.FLOAT, shape=[10, 64])
            ],
        )
        pred = ModelSchema(
            input_schema=[
                ModelSchemaItem(name="emb", data_type=DataType.FLOAT, shape=[10, 64])
            ],
            output_schema=[
                ModelSchemaItem(name="out", data_type=DataType.FLOAT, shape=[1])
            ],
        )
        sample = _build_fused_sample_input(tx, pred)
        self.assertEqual(set(sample.keys()), {"feat"})

    def test_unmappable_data_type_raises(self):
        """Unmappable data type raises."""
        schema = ModelSchema(
            input_schema=[ModelSchemaItem(name="a", data_type=None, shape=[3])]
        )
        with self.assertRaises(ValueError):
            _build_fused_sample_input(schema, None)


class BuildFusedSampleDataTest(unittest.TestCase):
    """Tests for ``build_fused_sample_data``."""

    def test_merges_and_filters_to_fused_inputs(self):
        """Merges and filters to fused inputs."""
        result = build_fused_sample_data(
            tx_sample_data=[{"raw_a": 1.0}, {"raw_b": 2}],
            predictor_sample_data=[{"raw_b": 99, "tx_only": 3.0}],
            fused_input_cols={"raw_a", "raw_b"},
        )
        self.assertEqual(result, [{"raw_a": 1.0, "raw_b": 2}])

    def test_none_samples_return_single_empty_batch(self):
        """None samples return single empty batch."""
        self.assertEqual(build_fused_sample_data(None, None, {"unused"}), [{}])


class IsStateDictTest(unittest.TestCase):
    """Tests for ``_is_state_dict``."""

    def test_true_for_dict_of_tensors(self):
        """True for dict of tensors."""
        self.assertTrue(_is_state_dict({"a": torch.zeros(1), "b": torch.ones(2)}))

    def test_empty_dict_considered_state_dict(self):
        """Empty dict considered state dict."""
        self.assertTrue(_is_state_dict({}))

    def test_false_for_non_dict(self):
        """False for non dict."""
        self.assertFalse(_is_state_dict(torch.zeros(1)))
        self.assertFalse(_is_state_dict(None))

    def test_false_for_dict_with_non_tensor_value(self):
        """False for dict with non tensor value."""
        self.assertFalse(_is_state_dict({"a": torch.zeros(1), "b": 1}))


class LoadModuleFromPathTest(unittest.TestCase):
    """Tests for ``_load_module_from_path``."""

    def test_missing_file_raises(self):
        """Missing file raises."""
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "nonexistent.pt")
            with self.assertRaises(FileNotFoundError) as ctx:
                _load_module_from_path(path, "some.Class", {})
            self.assertIn("Model file not found", str(ctx.exception))

    def test_load_full_module(self):
        """Load full module."""
        model = _TensorPredictor()
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            torch.save(model, path)
            loaded = _load_module_from_path(path, "ignored", {})
            self.assertIsInstance(loaded, nn.Module)
            self.assertFalse(loaded.training)
            x = torch.randn(2, 3)
            torch.testing.assert_close(loaded(x), model(x))
        finally:
            os.unlink(path)

    def test_load_state_dict_with_class(self):
        """Load state dict with class."""
        model = _TensorPredictor()
        state = model.state_dict()
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            torch.save(state, path)
            loaded = _load_module_from_path(path, _class_path(_TensorPredictor), {})
            self.assertIsInstance(loaded, _TensorPredictor)
            loaded.eval()
        finally:
            os.unlink(path)

    def test_load_state_dict_with_none_hyperparameters(self):
        """Load state dict with none hyperparameters."""
        model = _TensorPredictor()
        state = model.state_dict()
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            torch.save(state, path)
            loaded = _load_module_from_path(path, _class_path(_TensorPredictor), None)
            self.assertIsInstance(loaded, _TensorPredictor)
        finally:
            os.unlink(path)

    def test_load_non_module_non_state_dict_raises_type_error(self):
        """Load non module non state dict raises type error."""
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            torch.save([1, 2, 3], path)
            with self.assertRaises(TypeError) as ctx:
                _load_module_from_path(path, "ignored", {})
            self.assertIn(
                "did not contain a state_dict or nn.Module", str(ctx.exception)
            )
        finally:
            os.unlink(path)


class FuseModelsToTorchscriptTest(unittest.TestCase):
    """Integration tests for ``fuse_models_to_torchscript``."""

    def test_predictor_input_keys_dict_predictor_uses_schema_order(self):
        """Predictor input keys dict predictor uses schema order."""
        schema = ModelSchema(
            input_schema=[
                ModelSchemaItem(name="c", data_type=DataType.FLOAT, shape=[1]),
                ModelSchemaItem(name="a", data_type=DataType.FLOAT, shape=[1]),
                ModelSchemaItem(name="b", data_type=DataType.FLOAT, shape=[1]),
            ],
            output_schema=[
                ModelSchemaItem(name="out", data_type=DataType.FLOAT, shape=[1])
            ],
        )
        with tempfile.TemporaryDirectory() as d:
            tx_path = os.path.join(d, "tx.pt")
            pred_path = os.path.join(d, "pred.pt")
            dest_path = os.path.join(d, "fused.pt")
            torch.save(_DictTransform().state_dict(), tx_path)
            torch.save(_DictPredictor().state_dict(), pred_path)
            tx_schema = ModelSchema(
                input_schema=[
                    ModelSchemaItem(name="in", data_type=DataType.FLOAT, shape=[1])
                ],
                output_schema=[
                    ModelSchemaItem(name="c", data_type=DataType.FLOAT, shape=[1])
                ],
            )
            with (
                mock.patch(f"{_FUSE_MODULE}.FusedModel") as mock_fused,
                mock.patch("torch.jit.trace", return_value=mock.MagicMock()),
                mock.patch("torch.jit.save"),
            ):
                fuse_models_to_torchscript(
                    torch_model_path=pred_path,
                    tx_model_path=tx_path,
                    model_class=_class_path(_DictPredictor),
                    hyperparameters={},
                    tx_model_class=_class_path(_DictTransform),
                    tx_hyperparameters={},
                    dest_path=dest_path,
                    tx_model_schema=tx_schema,
                    model_schema=schema,
                )
                call_kwargs = mock_fused.call_args[1]
                self.assertTrue(call_kwargs["predictor_takes_dict"])
                self.assertEqual(call_kwargs["predictor_input_keys"], ["c", "a", "b"])

    def test_predictor_input_keys_tensor_predictor_uses_forward_order(self):
        """Predictor input keys tensor predictor uses forward order."""
        schema = ModelSchema(
            input_schema=[
                ModelSchemaItem(name="a", data_type=DataType.FLOAT, shape=[1]),
                ModelSchemaItem(name="b", data_type=DataType.FLOAT, shape=[1]),
                ModelSchemaItem(name="c", data_type=DataType.FLOAT, shape=[1]),
            ],
            output_schema=[
                ModelSchemaItem(name="out", data_type=DataType.FLOAT, shape=[1])
            ],
        )
        with tempfile.TemporaryDirectory() as d:
            tx_path = os.path.join(d, "tx.pt")
            pred_path = os.path.join(d, "pred.pt")
            dest_path = os.path.join(d, "fused.pt")
            torch.save(_DictTransform().state_dict(), tx_path)
            torch.save(_MultiTensorPredictor().state_dict(), pred_path)
            tx_schema = ModelSchema(
                input_schema=[
                    ModelSchemaItem(name="in", data_type=DataType.FLOAT, shape=[1])
                ],
                output_schema=[
                    ModelSchemaItem(name="a", data_type=DataType.FLOAT, shape=[1])
                ],
            )
            with (
                mock.patch(f"{_FUSE_MODULE}.FusedModel") as mock_fused,
                mock.patch("torch.jit.trace", return_value=mock.MagicMock()),
                mock.patch("torch.jit.save"),
            ):
                fuse_models_to_torchscript(
                    torch_model_path=pred_path,
                    tx_model_path=tx_path,
                    model_class=_class_path(_MultiTensorPredictor),
                    hyperparameters={},
                    tx_model_class=_class_path(_DictTransform),
                    tx_hyperparameters={},
                    dest_path=dest_path,
                    tx_model_schema=tx_schema,
                    model_schema=schema,
                )
                call_kwargs = mock_fused.call_args[1]
                self.assertFalse(call_kwargs["predictor_takes_dict"])
                # forward order is (b, a, c); schema order is (a, b, c).
                self.assertEqual(call_kwargs["predictor_input_keys"], ["b", "a", "c"])

    def test_predictor_input_keys_raises_when_schema_not_subset_of_forward(self):
        """Predictor input keys raises when schema not subset of forward."""
        schema = ModelSchema(
            input_schema=[
                ModelSchemaItem(name="a", data_type=DataType.FLOAT, shape=[1]),
                ModelSchemaItem(name="b", data_type=DataType.FLOAT, shape=[1]),
                ModelSchemaItem(name="c", data_type=DataType.FLOAT, shape=[1]),
            ],
            output_schema=[
                ModelSchemaItem(name="out", data_type=DataType.FLOAT, shape=[1])
            ],
        )
        with tempfile.TemporaryDirectory() as d:
            tx_path = os.path.join(d, "tx.pt")
            pred_path = os.path.join(d, "pred.pt")
            dest_path = os.path.join(d, "fused.pt")
            torch.save(_DictTransform().state_dict(), tx_path)
            torch.save(_TwoParamPredictor().state_dict(), pred_path)
            tx_schema = ModelSchema(
                input_schema=[
                    ModelSchemaItem(name="in", data_type=DataType.FLOAT, shape=[1])
                ],
                output_schema=[
                    ModelSchemaItem(name="a", data_type=DataType.FLOAT, shape=[1])
                ],
            )
            with self.assertRaises(ValueError) as ctx:
                fuse_models_to_torchscript(
                    torch_model_path=pred_path,
                    tx_model_path=tx_path,
                    model_class=_class_path(_TwoParamPredictor),
                    hyperparameters={},
                    tx_model_class=_class_path(_DictTransform),
                    tx_hyperparameters={},
                    dest_path=dest_path,
                    tx_model_schema=tx_schema,
                    model_schema=schema,
                )
            self.assertIn("c", str(ctx.exception))
            self.assertIn("forward()", str(ctx.exception))

    def test_predictor_input_keys_empty_forward_uses_schema_order(self):
        """Predictor input keys empty forward uses schema order."""
        schema = ModelSchema(
            input_schema=[
                ModelSchemaItem(name="x", data_type=DataType.FLOAT, shape=[1])
            ],
            output_schema=[
                ModelSchemaItem(name="out", data_type=DataType.FLOAT, shape=[1])
            ],
        )
        with tempfile.TemporaryDirectory() as d:
            tx_path = os.path.join(d, "tx.pt")
            pred_path = os.path.join(d, "pred.pt")
            dest_path = os.path.join(d, "fused.pt")
            torch.save(_DictTransform().state_dict(), tx_path)
            torch.save(_TensorPredictor().state_dict(), pred_path)
            tx_schema = ModelSchema(
                input_schema=[
                    ModelSchemaItem(name="in", data_type=DataType.FLOAT, shape=[1])
                ],
                output_schema=[
                    ModelSchemaItem(name="x", data_type=DataType.FLOAT, shape=[1])
                ],
            )
            with (
                mock.patch(f"{_FUSE_MODULE}.FusedModel") as mock_fused,
                mock.patch(f"{_FUSE_MODULE}._forward_param_order", return_value=[]),
                mock.patch("torch.jit.trace", return_value=mock.MagicMock()),
                mock.patch("torch.jit.save"),
            ):
                fuse_models_to_torchscript(
                    torch_model_path=pred_path,
                    tx_model_path=tx_path,
                    model_class=_class_path(_TensorPredictor),
                    hyperparameters={},
                    tx_model_class=_class_path(_DictTransform),
                    tx_hyperparameters={},
                    dest_path=dest_path,
                    tx_model_schema=tx_schema,
                    model_schema=schema,
                )
                call_kwargs = mock_fused.call_args[1]
                self.assertFalse(call_kwargs["predictor_takes_dict"])
                self.assertEqual(call_kwargs["predictor_input_keys"], ["x"])

    def test_dest_filename_only_uses_dot_makedirs(self):
        """Dest filename only uses dot makedirs."""
        with tempfile.TemporaryDirectory() as d:
            tx_path = os.path.join(d, "tx.pt")
            pred_path = os.path.join(d, "pred.pt")
            torch.save(_DictTransform().state_dict(), tx_path)
            torch.save(_TensorPredictor().state_dict(), pred_path)
            tx_schema = ModelSchema(
                input_schema=[
                    ModelSchemaItem(name="in", data_type=DataType.FLOAT, shape=[1])
                ],
                output_schema=[
                    ModelSchemaItem(name="out", data_type=DataType.FLOAT, shape=[1])
                ],
            )
            pred_schema = ModelSchema(
                input_schema=[
                    ModelSchemaItem(name="out", data_type=DataType.FLOAT, shape=[1])
                ],
                output_schema=[
                    ModelSchemaItem(name="output", data_type=DataType.FLOAT, shape=[1])
                ],
            )
            old = os.getcwd()
            try:
                os.chdir(d)
                fuse_models_to_torchscript(
                    torch_model_path=pred_path,
                    tx_model_path=tx_path,
                    model_class=_class_path(_TensorPredictor),
                    hyperparameters={},
                    tx_model_class=_class_path(_DictTransform),
                    tx_hyperparameters={},
                    dest_path="bare_name.pt",
                    tx_model_schema=tx_schema,
                    model_schema=pred_schema,
                )
                self.assertTrue(os.path.isfile(os.path.join(d, "bare_name.pt")))
            finally:
                os.chdir(old)

    def test_saves_torchscript_and_returns_dest_path(self):
        """Saves torchscript and returns dest path."""
        with tempfile.TemporaryDirectory() as d:
            tx_path = os.path.join(d, "tx.pt")
            pred_path = os.path.join(d, "pred.pt")
            dest_path = os.path.join(d, "fused.pt")
            torch.save(_DictTransform().state_dict(), tx_path)
            torch.save(_TensorPredictor().state_dict(), pred_path)
            tx_schema = ModelSchema(
                input_schema=[
                    ModelSchemaItem(name="in", data_type=DataType.FLOAT, shape=[1])
                ],
                output_schema=[
                    ModelSchemaItem(name="out", data_type=DataType.FLOAT, shape=[1])
                ],
            )
            pred_schema = ModelSchema(
                input_schema=[
                    ModelSchemaItem(name="out", data_type=DataType.FLOAT, shape=[1])
                ],
                output_schema=[
                    ModelSchemaItem(name="output", data_type=DataType.FLOAT, shape=[1])
                ],
            )
            result = fuse_models_to_torchscript(
                torch_model_path=pred_path,
                tx_model_path=tx_path,
                model_class=_class_path(_TensorPredictor),
                hyperparameters={},
                tx_model_class=_class_path(_DictTransform),
                tx_hyperparameters={},
                dest_path=dest_path,
                tx_model_schema=tx_schema,
                model_schema=pred_schema,
            )
            self.assertEqual(result, dest_path)
            self.assertTrue(os.path.isfile(dest_path))

            loaded = torch.jit.load(dest_path)
            out = loaded({"in": torch.tensor([[1.0]], dtype=torch.float32)})
            self.assertIsInstance(out, torch.Tensor)
            self.assertEqual(out.shape, (1, 1))

    def test_raises_when_fused_input_schema_empty(self):
        """Raises when fused input schema empty."""
        with tempfile.TemporaryDirectory() as d:
            tx_path = os.path.join(d, "tx.pt")
            pred_path = os.path.join(d, "pred.pt")
            dest_path = os.path.join(d, "fused.pt")
            torch.save(_DictTransform().state_dict(), tx_path)
            torch.save(_TensorPredictor().state_dict(), pred_path)
            with self.assertRaises(ValueError) as ctx:
                fuse_models_to_torchscript(
                    torch_model_path=pred_path,
                    tx_model_path=tx_path,
                    model_class=_class_path(_TensorPredictor),
                    hyperparameters={},
                    tx_model_class=_class_path(_DictTransform),
                    tx_hyperparameters={},
                    dest_path=dest_path,
                    tx_model_schema=None,
                    model_schema=None,
                )
            self.assertIn("Cannot build sample input", str(ctx.exception))

    def test_fuse_with_dict_predictor(self):
        """Fuse with dict predictor."""
        with tempfile.TemporaryDirectory() as d:
            tx_path = os.path.join(d, "tx.pt")
            pred_path = os.path.join(d, "pred.pt")
            dest_path = os.path.join(d, "fused.pt")
            torch.save(_DictTransform().state_dict(), tx_path)
            torch.save(_DictPredictor().state_dict(), pred_path)
            tx_schema = ModelSchema(
                input_schema=[
                    ModelSchemaItem(name="in", data_type=DataType.FLOAT, shape=[1])
                ],
                output_schema=[
                    ModelSchemaItem(name="out", data_type=DataType.FLOAT, shape=[1])
                ],
            )
            pred_schema = ModelSchema(
                input_schema=[
                    ModelSchemaItem(name="out", data_type=DataType.FLOAT, shape=[1])
                ],
                output_schema=[
                    ModelSchemaItem(name="output", data_type=DataType.FLOAT, shape=[1])
                ],
            )
            result = fuse_models_to_torchscript(
                torch_model_path=pred_path,
                tx_model_path=tx_path,
                model_class=_class_path(_DictPredictor),
                hyperparameters={},
                tx_model_class=_class_path(_DictTransform),
                tx_hyperparameters={},
                dest_path=dest_path,
                tx_model_schema=tx_schema,
                model_schema=pred_schema,
            )
            self.assertEqual(result, dest_path)
            self.assertTrue(os.path.isfile(dest_path))

    def test_fuse_with_transformer_predictor(self):
        """Fuse with transformer predictor."""
        with tempfile.TemporaryDirectory() as d:
            tx_path = os.path.join(d, "tx.pt")
            pred_path = os.path.join(d, "pred.pt")
            dest_path = os.path.join(d, "fused.pt")
            torch.save(
                _TransformerDictTransform(d_model=64, seq_len=10).state_dict(), tx_path
            )
            torch.save(
                _TransformerPredictor(
                    d_model=64, n_heads=4, n_layers=2, d_ff=128
                ).state_dict(),
                pred_path,
            )
            tx_schema = ModelSchema(
                input_schema=[
                    ModelSchemaItem(name="feat", data_type=DataType.FLOAT, shape=[1])
                ],
                output_schema=[
                    ModelSchemaItem(
                        name="emb", data_type=DataType.FLOAT, shape=[10, 64]
                    )
                ],
            )
            pred_schema = ModelSchema(
                input_schema=[
                    ModelSchemaItem(
                        name="emb", data_type=DataType.FLOAT, shape=[10, 64]
                    )
                ],
                output_schema=[
                    ModelSchemaItem(name="output", data_type=DataType.FLOAT, shape=[1])
                ],
            )
            result = fuse_models_to_torchscript(
                torch_model_path=pred_path,
                tx_model_path=tx_path,
                model_class=_class_path(_TransformerPredictor),
                hyperparameters={
                    "d_model": 64,
                    "n_heads": 4,
                    "n_layers": 2,
                    "d_ff": 128,
                },
                tx_model_class=_class_path(_TransformerDictTransform),
                tx_hyperparameters={"d_model": 64, "seq_len": 10},
                dest_path=dest_path,
                tx_model_schema=tx_schema,
                model_schema=pred_schema,
            )
            self.assertEqual(result, dest_path)
            self.assertTrue(os.path.isfile(dest_path))

            loaded = torch.jit.load(dest_path)
            out = loaded({"feat": torch.randn(2, 1, dtype=torch.float32)})
            self.assertIsInstance(out, torch.Tensor)
            self.assertEqual(out.shape, (2, 1))


class GetPredictorOutputFieldOrderTest(unittest.TestCase):
    """Tests for ``get_predictor_output_field_order``."""

    def _make_schema(
        self, input_names: list[str], output_names: list[str]
    ) -> ModelSchema:
        return ModelSchema(
            input_schema=[
                ModelSchemaItem(name=n, data_type=DataType.FLOAT, shape=[1])
                for n in input_names
            ],
            output_schema=[
                ModelSchemaItem(name=n, data_type=DataType.FLOAT, shape=[1])
                for n in output_names
            ],
        )

    def test_tensor_predictor_named_tuple_returns_definition_order(self):
        """Tensor predictor named tuple returns definition order."""
        model = _NamedTupleTensorPredictor()
        schema = self._make_schema(["a", "b"], ["a_out", "b_out"])
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            torch.save(model, path)
            result = get_predictor_output_field_order(path, "", {}, schema)
        finally:
            os.unlink(path)
        # _fields order is b_out, a_out (definition order), not schema order.
        self.assertEqual(result, ["b_out", "a_out"])

    def test_dict_predictor_named_tuple_returns_definition_order(self):
        """Dict predictor named tuple returns definition order."""
        model = _NamedTupleDictPredictor()
        schema = self._make_schema(["x", "z"], ["x_out", "z_out"])
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            torch.save(model, path)
            result = get_predictor_output_field_order(path, "", {}, schema)
        finally:
            os.unlink(path)
        self.assertEqual(result, ["z_out", "x_out"])

    def test_plain_tensor_output_returns_none(self):
        """Plain tensor output returns none."""
        model = _TensorPredictor()
        schema = self._make_schema(["out"], ["result"])
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            torch.save(model, path)
            result = get_predictor_output_field_order(path, "", {}, schema)
        finally:
            os.unlink(path)
        self.assertIsNone(result)

    def test_missing_file_returns_none_with_warning(self):
        """Missing file returns none with warning."""
        schema = self._make_schema(["x"], ["y"])
        with tempfile.TemporaryDirectory() as d:
            bad_path = os.path.join(d, "nonexistent.pt")
            with self.assertLogs(level="WARNING") as cm:
                result = get_predictor_output_field_order(bad_path, "", {}, schema)
        self.assertIsNone(result)
        self.assertTrue(any("Output schema unchanged" in line for line in cm.output))

    def test_strategy2_no_annotation_with_named_tuple_output(self):
        """Strategy2 no annotation with named tuple output."""
        model = _NoAnnotNTPredictor()
        schema = self._make_schema(["x"], ["c_out", "d_out"])
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            torch.save(model, path)
            result = get_predictor_output_field_order(path, "", {}, schema)
        finally:
            os.unlink(path)
        self.assertEqual(result, ["c_out", "d_out"])

    def test_strategy1_named_tuple_from_return_annotation_state_dict(self):
        """Strategy1 named tuple from return annotation state dict."""
        model = _Strategy1AnnotatedPredictor()
        schema = self._make_schema(["x"], ["second", "first"])
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            torch.save(model.state_dict(), path)
            result = get_predictor_output_field_order(
                path, _class_path(_Strategy1AnnotatedPredictor), {}, schema
            )
        finally:
            os.unlink(path)
        self.assertEqual(result, ["second", "first"])

    def test_strategy1_signature_failure_falls_through_to_strategy2(self):
        """When Strategy 1's signature inspection raises, Strategy 2 still runs."""
        model = _NoAnnotNTPredictor()
        schema = self._make_schema(["x"], ["c_out", "d_out"])
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            torch.save(model, path)
            calls = {"n": 0}
            real_sig = inspect.signature

            def _sig(obj):
                calls["n"] += 1
                if calls["n"] == 1:
                    raise ValueError("simulated Strategy 1 failure")
                return real_sig(obj)

            with mock.patch(f"{_FUSE_MODULE}.inspect.signature", side_effect=_sig):
                result = get_predictor_output_field_order(path, "", {}, schema)
        finally:
            os.unlink(path)
        self.assertEqual(result, ["c_out", "d_out"])


class ComputePythonFuseMetadataTest(unittest.TestCase):
    """Tests for ``compute_python_fuse_metadata``."""

    def _run(self, predictor, pred_class, tx_schema=None, pred_schema=None):
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            torch.save(predictor.state_dict(), path)
            return compute_python_fuse_metadata(
                torch_model_path=path,
                model_class=pred_class,
                hyperparameters={},
                tx_model_schema=tx_schema,
                model_schema=pred_schema,
            )
        finally:
            os.unlink(path)

    def test_dict_predictor_returns_schema_order_keys(self):
        """Dict predictor returns schema order keys."""
        tx_schema = ModelSchema(
            input_schema=[
                ModelSchemaItem(name="in_a", data_type=DataType.FLOAT, shape=[1]),
                ModelSchemaItem(name="in_b", data_type=DataType.FLOAT, shape=[1]),
            ]
        )
        pred_schema = ModelSchema(
            input_schema=[
                ModelSchemaItem(name="out", data_type=DataType.FLOAT, shape=[1]),
                ModelSchemaItem(name="extra", data_type=DataType.FLOAT, shape=[1]),
            ]
        )
        tx_keys, pred_keys, takes_dict = self._run(
            _DictPredictor(), _class_path(_DictPredictor), tx_schema, pred_schema
        )
        self.assertEqual(tx_keys, ["in_a", "in_b"])
        self.assertEqual(pred_keys, ["out", "extra"])
        self.assertTrue(takes_dict)

    def test_tensor_predictor_reorders_keys_to_forward_order(self):
        """Tensor predictor reorders keys to forward order."""
        tx_schema = ModelSchema(
            input_schema=[
                ModelSchemaItem(name="tx_in", data_type=DataType.FLOAT, shape=[1])
            ]
        )
        pred_schema = ModelSchema(
            input_schema=[
                ModelSchemaItem(name="a", data_type=DataType.FLOAT, shape=[1]),
                ModelSchemaItem(name="b", data_type=DataType.FLOAT, shape=[1]),
                ModelSchemaItem(name="c", data_type=DataType.FLOAT, shape=[1]),
            ]
        )
        tx_keys, pred_keys, takes_dict = self._run(
            _MultiTensorPredictor(),
            _class_path(_MultiTensorPredictor),
            tx_schema,
            pred_schema,
        )
        self.assertEqual(tx_keys, ["tx_in"])
        self.assertEqual(pred_keys, ["b", "a", "c"])
        self.assertFalse(takes_dict)

    def test_raises_when_schema_not_subset_of_forward(self):
        """Raises when schema not subset of forward."""
        bad_schema = ModelSchema(
            input_schema=[
                ModelSchemaItem(name="a", data_type=DataType.FLOAT, shape=[1]),
                ModelSchemaItem(name="b", data_type=DataType.FLOAT, shape=[1]),
                ModelSchemaItem(name="c", data_type=DataType.FLOAT, shape=[1]),
            ]
        )
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            torch.save(_TwoParamPredictor().state_dict(), path)
            with self.assertRaises(ValueError) as ctx:
                compute_python_fuse_metadata(
                    torch_model_path=path,
                    model_class=_class_path(_TwoParamPredictor),
                    hyperparameters={},
                    tx_model_schema=None,
                    model_schema=bad_schema,
                )
            self.assertIn("c", str(ctx.exception))
            self.assertIn("forward()", str(ctx.exception))
        finally:
            os.unlink(path)

    def test_none_schemas_return_empty_keys(self):
        """None schemas return empty keys."""
        tx_keys, pred_keys, takes_dict = self._run(
            _TensorPredictor(), _class_path(_TensorPredictor)
        )
        self.assertEqual(tx_keys, [])
        self.assertEqual(pred_keys, [])
        self.assertFalse(takes_dict)


# ===========================================================================
# Tranche 2: ONNX
# ===========================================================================


class ExpandBatchForOnnxExportTest(unittest.TestCase):
    """Tests for ``_expand_batch_for_onnx_export``."""

    def test_repeats_batch_one_to_two(self):
        """Repeats batch one to two."""
        a = torch.zeros(1, 3)
        b = torch.ones(1, 1)
        out = fuse_module._expand_batch_for_onnx_export((a, b))
        self.assertEqual(out[0].shape, (2, 3))
        self.assertEqual(out[1].shape, (2, 1))

    def test_leaves_batch_greater_than_one_unchanged(self):
        """Leaves batch greater than one unchanged."""
        a = torch.randn(3, 2)
        b = torch.randn(1, 2)
        out = fuse_module._expand_batch_for_onnx_export((a, b))
        self.assertIs(out[0], a)
        self.assertEqual(out[1].shape[0], 2)


class DisableTransformerEncoderFastpathForOnnxTest(unittest.TestCase):
    """Tests for ``_disable_transformer_encoder_fused_fastpath_for_onnx``."""

    def test_noop_when_mha_backend_missing(self):
        """Noop when mha backend missing."""
        root = nn.Linear(1, 1)
        root.eval()
        with mock.patch.object(fuse_module.torch, "backends", SimpleNamespace()):
            with _disable_transformer_encoder_fused_fastpath_for_onnx(root):
                self.assertFalse(root.training)
            self.assertFalse(root.training)

    def test_noop_preserves_train_when_mha_backend_missing(self):
        """Noop preserves train when mha backend missing."""
        root = nn.Linear(1, 1)
        root.train()
        with mock.patch.object(fuse_module.torch, "backends", SimpleNamespace()):
            with _disable_transformer_encoder_fused_fastpath_for_onnx(root):
                self.assertTrue(root.training)
            self.assertTrue(root.training)

    def test_mha_setter_without_getter_defaults_prev_to_true(self):
        """Mha setter without getter defaults prev to true."""
        calls: list[bool] = []

        class _Mha:
            def set_fastpath_enabled(self, enabled: bool) -> None:
                calls.append(enabled)

        with mock.patch.object(fuse_module.torch.backends, "mha", _Mha(), create=True):
            root = nn.Linear(1, 1)
            with _disable_transformer_encoder_fused_fastpath_for_onnx(root):
                self.assertEqual(calls, [False])
            self.assertEqual(calls, [False, True])


class OnnxDynamoExporterDepsTest(unittest.TestCase):
    """Tests for ``_onnx_dynamo_exporter_dependencies_available``."""

    def test_true_when_onnxscript_importable(self):
        """True when onnxscript importable."""
        with mock.patch.object(
            fuse_module.importlib.util, "find_spec", return_value=object()
        ):
            self.assertTrue(_onnx_dynamo_exporter_dependencies_available())

    def test_false_when_onnxscript_missing(self):
        """False when onnxscript missing."""
        with mock.patch.object(
            fuse_module.importlib.util, "find_spec", return_value=None
        ):
            self.assertFalse(_onnx_dynamo_exporter_dependencies_available())


class OnnxDynamoDynamicShapesForTupleArgTest(unittest.TestCase):
    """Tests for ``_onnx_dynamo_dynamic_shapes_for_tuple_arg``."""

    def test_returns_none_when_torch_export_dim_unavailable(self):
        """Returns none when torch export dim unavailable."""
        with mock.patch.object(fuse_module, "_TorchExportDim", None):
            self.assertIsNone(
                _onnx_dynamo_dynamic_shapes_for_tuple_arg((torch.zeros(1),))
            )

    def test_returns_one_dim_dict_per_tensor(self):
        """Returns one dim dict per tensor."""
        result = _onnx_dynamo_dynamic_shapes_for_tuple_arg(
            (torch.zeros(1, 2), torch.zeros(1, 3))
        )
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)
        per_tensor_dims = result[0]
        self.assertEqual(len(per_tensor_dims), 2)
        for dims in per_tensor_dims:
            self.assertEqual(set(dims.keys()), {0})


class OnnxDynamoExportErrorRetryPolicyTest(unittest.TestCase):
    """Tests for ``_onnx_dynamo_export_error_should_retry_legacy``."""

    def test_import_errors_always_retry(self):
        """Import errors always retry."""
        self.assertTrue(_onnx_dynamo_export_error_should_retry_legacy(ImportError("x")))
        self.assertTrue(
            _onnx_dynamo_export_error_should_retry_legacy(ModuleNotFoundError("x"))
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
                    _onnx_dynamo_export_error_should_retry_legacy(RuntimeError(msg))
                )

    def test_unrelated_error_does_not_retry(self):
        """Unrelated error does not retry."""
        self.assertFalse(
            _onnx_dynamo_export_error_should_retry_legacy(
                RuntimeError("shape mismatch")
            )
        )


class OnnxExportAttachInputsToOutputTest(unittest.TestCase):
    """Tests for ``_onnx_export_attach_inputs_to_output``."""

    def test_empty_inputs_returns_output_unchanged(self):
        """Empty inputs returns output unchanged."""
        out = torch.tensor([1.0])
        self.assertIs(_onnx_export_attach_inputs_to_output(out, ()), out)

    def test_tensor_output_gains_zero_valued_dependency(self):
        """Tensor output gains zero valued dependency."""
        out = torch.tensor([5.0])
        inputs = (torch.tensor([1.0]), torch.tensor([2.0]))
        result = _onnx_export_attach_inputs_to_output(out, inputs)
        torch.testing.assert_close(result, out)
        self.assertIsNot(result, out)

    def test_named_tuple_output_first_tensor_field_adjusted(self):
        """Named tuple output first tensor field adjusted."""

        class Out(NamedTuple):
            a: torch.Tensor
            b: torch.Tensor

        out = Out(a=torch.tensor([1.0]), b=torch.tensor([2.0]))
        result = _onnx_export_attach_inputs_to_output(out, (torch.tensor([3.0]),))
        self.assertIsInstance(result, Out)
        torch.testing.assert_close(result.a, out.a)
        torch.testing.assert_close(result.b, out.b)

    def test_dict_output_first_tensor_value_adjusted(self):
        """Dict output first tensor value adjusted."""
        out = {"a": torch.tensor([1.0]), "b": torch.tensor([2.0])}
        result = _onnx_export_attach_inputs_to_output(out, (torch.tensor([3.0]),))
        self.assertIsInstance(result, dict)
        torch.testing.assert_close(result["a"], out["a"])

    def test_tuple_output_first_tensor_element_adjusted(self):
        """Tuple output first tensor element adjusted."""
        out = (torch.tensor([1.0]), torch.tensor([2.0]))
        result = _onnx_export_attach_inputs_to_output(out, (torch.tensor([3.0]),))
        self.assertIsInstance(result, tuple)
        torch.testing.assert_close(result[0], out[0])

    def test_unrecognized_output_type_returned_unchanged(self):
        """Unrecognized output type returned unchanged."""
        out = "not a tensor container"
        self.assertEqual(
            _onnx_export_attach_inputs_to_output(out, (torch.tensor([1.0]),)), out
        )


class FusedOnnxTupleWrapperTest(unittest.TestCase):
    """Tests for ``_FusedOnnxTupleWrapper``."""

    def test_forward_zips_keys_and_runs_fused(self):
        """Forward zips keys and runs fused."""
        fused = FusedModel(
            transform_module=_DictTransform(),
            predictor_module=_TensorPredictor(),
            transform_input_keys=["in"],
            predictor_input_keys=["out"],
            predictor_takes_dict=False,
        )
        wrapper = _FusedOnnxTupleWrapper(fused, ["in"])
        with torch.no_grad():
            y = wrapper(torch.tensor([[1.0]], dtype=torch.float32))
        self.assertEqual(y.shape, (1, 1))


class FusedOnnxDynamoTupleWrapperTest(unittest.TestCase):
    """Tests for ``_FusedOnnxDynamoTupleWrapper``."""

    def test_forward_takes_single_tuple_argument(self):
        """Forward takes single tuple argument."""
        fused = FusedModel(
            transform_module=_DictTransform(),
            predictor_module=_TensorPredictor(),
            transform_input_keys=["in"],
            predictor_input_keys=["out"],
            predictor_takes_dict=False,
        )
        wrapper = _FusedOnnxDynamoTupleWrapper(fused, ["in"])
        with torch.no_grad():
            y = wrapper((torch.tensor([[1.0]], dtype=torch.float32),))
        self.assertEqual(y.shape, (1, 1))


class ForceOnnxIoShapesFromSchemaTest(unittest.TestCase):
    """Tests for ``_force_onnx_io_shapes_from_schema``."""

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
            fuse_module._force_onnx_io_shapes_from_schema(dest_path, [schema])
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
            fuse_module._force_onnx_io_shapes_from_schema(dest_path, [None])
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
            with self.assertLogs(fuse_module._logger, level="WARNING") as cm:
                fuse_module._force_onnx_io_shapes_from_schema(dest_path, [schema])
            self.assertTrue(
                any("Skipping ONNX shape override" in line for line in cm.output)
            )


class FuseModelsToOnnxTest(unittest.TestCase):
    """Integration tests for ``fuse_models_to_onnx``."""

    def test_saves_valid_model(self):
        """Saves valid model."""
        with tempfile.TemporaryDirectory() as d:
            tx_path = os.path.join(d, "tx.pt")
            pred_path = os.path.join(d, "pred.pt")
            dest_path = os.path.join(d, "fused.onnx")
            torch.save(_DictTransform().state_dict(), tx_path)
            torch.save(_TensorPredictor().state_dict(), pred_path)
            tx_schema = ModelSchema(
                input_schema=[
                    ModelSchemaItem(name="in", data_type=DataType.FLOAT, shape=[1])
                ],
                output_schema=[
                    ModelSchemaItem(name="out", data_type=DataType.FLOAT, shape=[1])
                ],
            )
            pred_schema = ModelSchema(
                input_schema=[
                    ModelSchemaItem(name="out", data_type=DataType.FLOAT, shape=[1])
                ],
                output_schema=[
                    ModelSchemaItem(name="output", data_type=DataType.FLOAT, shape=[1])
                ],
            )
            result = fuse_models_to_onnx(
                torch_model_path=pred_path,
                tx_model_path=tx_path,
                model_class=_class_path(_TensorPredictor),
                hyperparameters={},
                tx_model_class=_class_path(_DictTransform),
                tx_hyperparameters={},
                dest_path=dest_path,
                tx_model_schema=tx_schema,
                model_schema=pred_schema,
            )
            self.assertEqual(result, dest_path)
            self.assertTrue(os.path.isfile(dest_path))
            model_proto = onnx.load(dest_path)
            self.assertGreater(len(model_proto.graph.input), 0)
            self.assertGreater(len(model_proto.graph.output), 0)

    def test_named_tuple_probe_sets_output_names(self):
        """Named tuple probe sets output names."""
        with tempfile.TemporaryDirectory() as d:
            tx_path = os.path.join(d, "tx.pt")
            pred_path = os.path.join(d, "pred.pt")
            dest_path = os.path.join(d, "fused_nt.onnx")
            torch.save(_PairTransform().state_dict(), tx_path)
            torch.save(_NamedTupleTensorPredictor().state_dict(), pred_path)
            tx_schema = ModelSchema(
                input_schema=[
                    ModelSchemaItem(name="in", data_type=DataType.FLOAT, shape=[1])
                ],
                output_schema=[
                    ModelSchemaItem(name="a", data_type=DataType.FLOAT, shape=[1]),
                    ModelSchemaItem(name="b", data_type=DataType.FLOAT, shape=[1]),
                ],
            )
            pred_schema = ModelSchema(
                input_schema=[
                    ModelSchemaItem(name="a", data_type=DataType.FLOAT, shape=[1]),
                    ModelSchemaItem(name="b", data_type=DataType.FLOAT, shape=[1]),
                ],
                output_schema=[
                    ModelSchemaItem(name="a_out", data_type=DataType.FLOAT, shape=[1]),
                    ModelSchemaItem(name="b_out", data_type=DataType.FLOAT, shape=[1]),
                ],
            )
            fuse_models_to_onnx(
                torch_model_path=pred_path,
                tx_model_path=tx_path,
                model_class=_class_path(_NamedTupleTensorPredictor),
                hyperparameters={},
                tx_model_class=_class_path(_PairTransform),
                tx_hyperparameters={},
                dest_path=dest_path,
                tx_model_schema=tx_schema,
                model_schema=pred_schema,
            )
            model_proto = onnx.load(dest_path)
            onnx_out = [o.name for o in model_proto.graph.output]
            self.assertEqual(onnx_out, ["b_out", "a_out"])

    def test_dest_filename_only_uses_dot_makedirs(self):
        """Dest filename only uses dot makedirs."""
        with tempfile.TemporaryDirectory() as d:
            tx_path = os.path.join(d, "tx.pt")
            pred_path = os.path.join(d, "pred.pt")
            torch.save(_DictTransform().state_dict(), tx_path)
            torch.save(_TensorPredictor().state_dict(), pred_path)
            tx_schema = ModelSchema(
                input_schema=[
                    ModelSchemaItem(name="in", data_type=DataType.FLOAT, shape=[1])
                ],
                output_schema=[
                    ModelSchemaItem(name="out", data_type=DataType.FLOAT, shape=[1])
                ],
            )
            pred_schema = ModelSchema(
                input_schema=[
                    ModelSchemaItem(name="out", data_type=DataType.FLOAT, shape=[1])
                ],
                output_schema=[
                    ModelSchemaItem(name="output", data_type=DataType.FLOAT, shape=[1])
                ],
            )
            old = os.getcwd()
            try:
                os.chdir(d)
                fuse_models_to_onnx(
                    torch_model_path=pred_path,
                    tx_model_path=tx_path,
                    model_class=_class_path(_TensorPredictor),
                    hyperparameters={},
                    tx_model_class=_class_path(_DictTransform),
                    tx_hyperparameters={},
                    dest_path="bare_name.onnx",
                    tx_model_schema=tx_schema,
                    model_schema=pred_schema,
                )
                self.assertTrue(os.path.isfile(os.path.join(d, "bare_name.onnx")))
            finally:
                os.chdir(old)

    def test_logs_when_sample_forward_probe_fails(self):
        """Logs when sample forward probe fails."""
        with tempfile.TemporaryDirectory() as d:
            tx_path = os.path.join(d, "tx.pt")
            pred_path = os.path.join(d, "pred.pt")
            dest_path = os.path.join(d, "fused_warn.onnx")
            torch.save(_DictTransform().state_dict(), tx_path)
            torch.save(_TensorPredictor().state_dict(), pred_path)
            tx_schema = ModelSchema(
                input_schema=[
                    ModelSchemaItem(name="in", data_type=DataType.FLOAT, shape=[1])
                ],
                output_schema=[
                    ModelSchemaItem(name="out", data_type=DataType.FLOAT, shape=[1])
                ],
            )
            pred_schema = ModelSchema(
                input_schema=[
                    ModelSchemaItem(name="out", data_type=DataType.FLOAT, shape=[1])
                ],
                output_schema=[
                    ModelSchemaItem(name="output", data_type=DataType.FLOAT, shape=[1])
                ],
            )
            orig_forward = FusedModel.forward
            calls = {"n": 0}

            def _flaky_forward(self, inputs):
                calls["n"] += 1
                if calls["n"] == 1:
                    raise RuntimeError("probe failure")
                return orig_forward(self, inputs)

            with (
                mock.patch.object(FusedModel, "forward", _flaky_forward),
                self.assertLogs(fuse_module._logger, level="WARNING") as cm,
            ):
                fuse_models_to_onnx(
                    torch_model_path=pred_path,
                    tx_model_path=tx_path,
                    model_class=_class_path(_TensorPredictor),
                    hyperparameters={},
                    tx_model_class=_class_path(_DictTransform),
                    tx_hyperparameters={},
                    dest_path=dest_path,
                    tx_model_schema=tx_schema,
                    model_schema=pred_schema,
                )
            self.assertTrue(
                any("Could not infer ONNX output names" in line for line in cm.output)
            )
            self.assertTrue(os.path.isfile(dest_path))
            onnx.load(dest_path)


# ===========================================================================
# Tranche 3: native-transform-gated (present but skipped -- see PR F)
# ===========================================================================

_NATIVE_TRANSFORM_SKIP_REASON = (
    "Requires the native-transform package (TransformSpec, TorchTransformModule), "
    "which has not yet been migrated to OSS michelangelo -- see migration bucket "
    "'PR F'. model_fuser.fuse._build_tx_hydra_spec raises NotImplementedError "
    "until then, which fuse_models_to_python always hits for its raw package."
)


@unittest.skip(_NATIVE_TRANSFORM_SKIP_REASON)
class BuildTxHydraSpecTest(unittest.TestCase):
    """Placeholder for the internal ``BuildTxHydraSpecTest`` suite.

    Internally this asserts that ``_build_tx_hydra_spec`` reconstructs a
    ``TransformSpec`` from its ``to_dict()`` output and emits a Hydra spec
    with layers grouped by topological level, matching the state dict's
    ``transform_module.*`` key layout. Unskip and port faithfully once
    native-transform lands (PR F).
    """

    def test_builds_hydra_spec_from_transform_spec_dict(self):
        """Builds hydra spec from transform spec dict."""
        raise NotImplementedError("native_transform not yet available in OSS")


@unittest.skip(_NATIVE_TRANSFORM_SKIP_REASON)
class FuseModelsToPythonTest(unittest.TestCase):
    """Placeholder for the internal ``FuseModelsToPythonTest`` suite.

    Internally this covers combining predictor + transform state dicts under
    ``predictor_module.*``/``transform_module.*`` prefixes and building the
    Hydra reconstruction spec. In OSS, ``fuse_models_to_python`` *always*
    calls ``_build_tx_hydra_spec`` for its raw package (see that function's
    docstring), so every scenario in this class -- including the state-dict
    combination itself -- currently raises ``NotImplementedError``. Unskip
    and port faithfully once native-transform lands (PR F).
    """

    def test_combines_state_dicts_and_builds_fused_hyperparameters(self):
        """Combines state dicts and builds fused hyperparameters."""
        with tempfile.TemporaryDirectory() as d:
            tx_path = os.path.join(d, "tx.pt")
            pred_path = os.path.join(d, "pred.pt")
            dest_path = os.path.join(d, "fused.pt")
            torch.save(_TensorPredictor().state_dict(), pred_path)
            torch.save(_DictTransform(), tx_path)
            with self.assertRaises(NotImplementedError):
                fuse_models_to_python(
                    torch_model_path=pred_path,
                    tx_model_path=tx_path,
                    model_class=_class_path(_TensorPredictor),
                    hyperparameters={},
                    tx_hyperparameters={},
                    dest_path=dest_path,
                    tx_model_schema=None,
                    model_schema=None,
                )


class FuseModelsToPythonNotImplementedTest(unittest.TestCase):
    """Documents today's actual (gated) behavior of ``fuse_models_to_python``.

    Unlike the placeholder class above (which mirrors the internal suite and
    stays skipped), this test runs today: it locks in that calling
    ``fuse_models_to_python`` raises a clear, actionable ``NotImplementedError``
    rather than failing confusingly or silently returning bad data.
    """

    def test_raises_notimplementederror_pointing_to_native_transform(self):
        """Raises notimplementederror pointing to native transform."""
        with tempfile.TemporaryDirectory() as d:
            tx_path = os.path.join(d, "tx.pt")
            pred_path = os.path.join(d, "pred.pt")
            dest_path = os.path.join(d, "fused.pt")
            torch.save(_TensorPredictor().state_dict(), pred_path)
            torch.save(_DictTransform(), tx_path)
            with self.assertRaises(NotImplementedError) as ctx:
                fuse_models_to_python(
                    torch_model_path=pred_path,
                    tx_model_path=tx_path,
                    model_class=_class_path(_TensorPredictor),
                    hyperparameters={},
                    tx_hyperparameters={},
                    dest_path=dest_path,
                    tx_model_schema=None,
                    model_schema=None,
                )
            self.assertIn("native-transform", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
