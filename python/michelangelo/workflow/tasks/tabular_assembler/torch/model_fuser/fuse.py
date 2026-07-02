"""Fuse a PyTorch predictor with a preceding native-transform model.

Both the predictor and the native-transform model are expected to be saved
as ``state_dict`` (or full-module) ``.pt``/``.pth`` files. They are loaded,
wrapped in a :class:`~..fused_model.FusedModel` (transform -> predictor) with
schema-driven input/output merge, and exported as TorchScript, ONNX, or a
combined state dict for Python-backend serving.

The ONNX export path consolidates the fixes needed for high-quality export
(MHA fused-fastpath disable, dynamo-with-legacy-fallback, IO shape
normalization, batch-size expansion) as module-private helpers below —
mirroring the same fixes applied by the non-fused
``model_manager`` Triton packagers, so fused and non-fused models get
equivalent ONNX output quality.

Building the Hydra reconstruction spec for a fused *native-transform* model's
Python-backend package (:func:`_build_tx_hydra_spec`) requires the concrete
``TransformSpec``/``TorchTransformModule`` implementation, which has not yet
been migrated to OSS — see
:mod:`~michelangelo.workflow.tasks.tabular_assembler.torch.model_fuser.transform_spec_protocol`.
Until it lands, that one function raises ``NotImplementedError``; every other
function in this module (TorchScript export, ONNX export, field-order
recovery, sample-data merge) works standalone.
"""

from __future__ import annotations

import importlib.util
import inspect
import logging
import os
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import onnx
import torch

from michelangelo.lib.model_manager.schema import DataType, ModelSchema
from michelangelo.lib.model_manager.utils.torch.data_type import (
    data_type_to_torch_dtype,
)
from michelangelo.uniflow.core.utils import import_attribute
from michelangelo.workflow.tasks.tabular_assembler._private.schema.fuse import (
    fuse_input_schema,
)

from .fused_model import FusedModel

if TYPE_CHECKING:
    from collections.abc import Iterator

try:
    from torch.export import Dim as _TorchExportDim
except ImportError:  # pragma: no cover - depends on installed torch version
    _TorchExportDim = None

try:
    import pytorch_lightning as pl
except ImportError:  # pragma: no cover - pytorch_lightning is a declared dependency
    pl = None

__all__ = [
    "build_fused_sample_data",
    "compute_python_fuse_metadata",
    "fuse_models_to_onnx",
    "fuse_models_to_python",
    "fuse_models_to_torchscript",
    "get_predictor_output_field_order",
]

_logger = logging.getLogger(__name__)

_ONNX_OPSET_LEGACY = 14
_ONNX_OPSET_DYNAMO = 18
_ONNX_EXPAND_BATCH_SIZE = 2


# ---------------------------------------------------------------------------
# Module loading / forward-signature helpers
# ---------------------------------------------------------------------------


def _forward_accepts_dict(module: torch.nn.Module) -> bool:
    """Return whether ``module.forward``'s first parameter is dict-annotated."""
    try:
        sig = inspect.signature(module.forward)
    except (ValueError, TypeError):
        return False
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        if param.annotation is inspect.Parameter.empty:
            return False
        return "dict" in str(param.annotation).lower()
    return False


def _forward_param_order(module: torch.nn.Module) -> list[str]:
    """Return ``module.forward``'s parameter names (excluding ``self``), in order."""
    try:
        sig = inspect.signature(module.forward)
    except (ValueError, TypeError):
        return []
    return [name for name in sig.parameters if name != "self"]


def _schema_input_keys(schema: ModelSchema | None) -> list[str]:
    """Return input feature names in schema order, or ``[]`` if unset."""
    if schema is None:
        return []
    return [item.name for item in schema.input_schema]


def _schema_output_keys(schema: ModelSchema | None) -> list[str]:
    """Return output feature names in schema order, or ``[]`` if unset."""
    if schema is None:
        return []
    return [item.name for item in schema.output_schema]


def _build_fused_sample_input(
    tx_model_schema: ModelSchema | None,
    model_schema: ModelSchema | None,
    batch_size: int = 1,
) -> dict[str, torch.Tensor]:
    """Build a sample input dict for the fused model, for tracing/inference.

    Uses the same input feature set as :func:`fuse_input_schema`. Each tensor
    has shape ``[batch_size, *feature_shape]`` and dtype derived from the
    item's ``data_type``. Tensors are placed on CUDA when available, else CPU
    (matching the device the fused module is traced on).

    Args:
        tx_model_schema: Native-transform model schema, or ``None``.
        model_schema: Predictor model schema, or ``None``.
        batch_size: Batch dimension for the sample tensors.

    Returns:
        Mapping of fused input feature name to a zero-filled sample tensor.
        Empty if the fused input schema is empty.
    """
    input_items = fuse_input_schema(tx_model_schema, model_schema)
    if not input_items:
        return {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample: dict[str, torch.Tensor] = {}
    for item in input_items:
        feature_shape = list(item.shape) if item.shape else [1]
        data_type = item.data_type if item.data_type is not None else DataType.UNKNOWN
        shape = [batch_size] + [max(1, int(s)) for s in feature_shape]
        dtype = data_type_to_torch_dtype(data_type)
        sample[item.name] = torch.zeros(shape, dtype=dtype, device=device)
    return sample


def _is_state_dict(obj: Any) -> bool:
    """Return whether ``obj`` is a state_dict (a dict of name -> Tensor)."""
    return isinstance(obj, dict) and all(
        isinstance(v, torch.Tensor) for v in obj.values()
    )


def _load_module_from_path(
    path: str,
    model_class: str,
    hyperparameters: dict[str, Any],
) -> torch.nn.Module:
    """Load an ``nn.Module`` from a local file (state_dict or full module).

    Args:
        path: Local path to a ``.pt``/``.pth`` file.
        model_class: Dotted class name to instantiate when the file contains
            a state_dict.
        hyperparameters: Constructor kwargs for ``model_class``.

    Returns:
        The loaded module in eval mode.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        TypeError: If the file contains neither a state_dict nor an
            ``nn.Module``.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Model file not found: {path}")

    loaded = torch.load(path, map_location="cpu", weights_only=False)

    if _is_state_dict(loaded):
        model_cls = import_attribute(model_class)
        model = model_cls(**(hyperparameters or {}))
        model.load_state_dict(loaded)
    else:
        model = loaded

    if isinstance(model, torch.nn.Module):
        model.eval()
        return model
    raise TypeError(f"File {path} did not contain a state_dict or nn.Module")


def _align_predictor_input_keys(
    pred_module: torch.nn.Module,
    predictor_input_keys: list[str],
    predictor_takes_dict: bool,
) -> list[str]:
    """Reorder predictor input keys to match ``forward()``'s parameter order.

    For dict-accepting predictors, order does not matter and the keys are
    returned unchanged. For positional predictors, schema keys are aligned to
    the ``forward()`` signature order.

    Args:
        pred_module: The predictor module.
        predictor_input_keys: Feature names from the predictor's input schema.
        predictor_takes_dict: Whether the predictor's ``forward`` takes a
            single dict argument.

    Returns:
        ``predictor_input_keys``, reordered to match ``forward()`` when the
        predictor takes positional tensors.

    Raises:
        ValueError: If a positional predictor's ``forward()`` is missing a
            parameter named in the schema.
    """
    if predictor_takes_dict:
        return predictor_input_keys
    forward_params = _forward_param_order(pred_module)
    schema_set = set(predictor_input_keys)
    if forward_params and schema_set:
        forward_param_set = set(forward_params)
        if not schema_set.issubset(forward_param_set):
            unknown = sorted(schema_set - forward_param_set)
            raise ValueError(
                "Predictor model_schema includes input names that are not "
                f"parameters of forward(): {unknown}. forward() parameters "
                f"(excluding self): {forward_params}. Align the model_schema "
                "with the module's forward(), or use "
                "forward(inputs: dict[str, torch.Tensor]) so the fused model "
                "passes a dict."
            )
        return [p for p in forward_params if p in schema_set]
    return predictor_input_keys


def _build_fused_model_and_sample(
    torch_model_path: str,
    tx_model_path: str,
    model_class: str,
    hyperparameters: dict[str, Any],
    tx_model_class: str,
    tx_hyperparameters: dict[str, Any],
    tx_model_schema: ModelSchema | None = None,
    model_schema: ModelSchema | None = None,
) -> tuple[FusedModel, dict[str, torch.Tensor], list[str]]:
    """Load transform + predictor, build the ``FusedModel``, and a sample batch.

    Args:
        torch_model_path: Local path to the predictor model.
        tx_model_path: Local path to the native-transform model.
        model_class: Dotted class name for the predictor.
        hyperparameters: Constructor kwargs for the predictor.
        tx_model_class: Dotted class name for the transform.
        tx_hyperparameters: Constructor kwargs for the transform.
        tx_model_schema: Native-transform model schema.
        model_schema: Predictor model schema.

    Returns:
        A tuple of ``(fused_module, sample_input, input_key_order)`` where
        ``fused_module`` is on the trace device, ``sample_input`` is a dict
        of sample tensors, and ``input_key_order`` is the sample's key order
        (matching ONNX/Triton input names).

    Raises:
        ValueError: If the fused input schema is empty, so no sample input
            can be built for tracing.
    """
    tx_hyperparameters = tx_hyperparameters or {}
    hyperparameters = hyperparameters or {}

    transform_module = _load_module_from_path(
        tx_model_path, tx_model_class, tx_hyperparameters
    )
    predictor_module = _load_module_from_path(
        torch_model_path, model_class, hyperparameters
    )

    transform_input_keys = _schema_input_keys(tx_model_schema)
    predictor_takes_dict = _forward_accepts_dict(predictor_module)
    predictor_input_keys = _schema_input_keys(model_schema)
    predictor_input_keys = _align_predictor_input_keys(
        predictor_module, predictor_input_keys, predictor_takes_dict
    )

    fused = FusedModel(
        transform_module=transform_module,
        predictor_module=predictor_module,
        transform_input_keys=transform_input_keys,
        predictor_input_keys=predictor_input_keys,
        predictor_takes_dict=predictor_takes_dict,
    )
    fused.eval()

    sample_input = _build_fused_sample_input(tx_model_schema, model_schema)
    if not sample_input:
        raise ValueError(
            "Cannot build sample input for trace: the fused input schema "
            "(from fuse_input_schema) is empty."
        )
    trace_device = next(iter(sample_input.values())).device
    fused = fused.to(trace_device)

    input_key_order = list(sample_input.keys())
    return fused, sample_input, input_key_order


def _build_tx_hydra_spec(tx_hyperparameters: dict[str, Any]) -> dict[str, Any]:
    """Build a Hydra reconstruction spec for a fused native-transform layer stack.

    Not yet implemented in OSS michelangelo: reconstructing a
    ``TorchTransformModule``/layer stack from a stored ``TransformSpec`` dict
    requires the native-transform package, which has not been migrated (see
    :mod:`.transform_spec_protocol`). This blocks only the Python-backend
    *raw* package for a native-transform-fused model
    (:func:`fuse_models_to_python`); the plain (no native-transform) path and
    the TorchScript/ONNX fused deployable paths do not call this function.

    Args:
        tx_hyperparameters: ``TransformSpec.to_dict()`` output for the
            transform model.

    Raises:
        NotImplementedError: Always, until native-transform support lands.
    """
    raise NotImplementedError(
        "Building a Hydra reconstruction spec for a fused native-transform "
        "model requires the native-transform package, which is not yet "
        "available in OSS michelangelo. See "
        "michelangelo.workflow.tasks.tabular_assembler.torch.model_fuser."
        "transform_spec_protocol.TransformSpec for the contract a future "
        "implementation must satisfy."
    )


# ---------------------------------------------------------------------------
# ONNX export helpers
#
# These mirror the fixes applied by the non-fused Triton packager's ONNX
# export path so fused and non-fused models get equivalent export quality:
# MHA fused-fastpath disable, dynamo-with-legacy-fallback, IO shape
# normalization from the model schema, and batch-size expansion.
# ---------------------------------------------------------------------------


def _onnx_dynamo_exporter_dependencies_available() -> bool:
    """Return whether ``onnxscript`` is installed (required for ``dynamo=True``)."""
    return importlib.util.find_spec("onnxscript") is not None


@contextmanager
def _disable_transformer_encoder_fused_fastpath_for_onnx(
    root: torch.nn.Module,
) -> Iterator[None]:
    """Disable the MHA fused fast path during ONNX export.

    The fused fast path in ``nn.MultiheadAttention``/``TransformerEncoderLayer``
    produces graphs that don't export cleanly; disabling it for the duration
    of export (via ``torch.backends.mha.set_fastpath_enabled``) avoids that
    without requiring any change to the model itself.

    Args:
        root: Unused; kept for a stable call signature across callers.
    """
    _ = root
    mha = getattr(torch.backends, "mha", None)
    setter = getattr(mha, "set_fastpath_enabled", None) if mha is not None else None
    if setter is None:
        yield
        return
    getter = getattr(mha, "get_fastpath_enabled", None)
    prev = getter() if callable(getter) else True
    setter(False)
    try:
        yield
    finally:
        setter(prev)


def _onnx_dynamo_dynamic_shapes_for_tuple_arg(
    tuple_in: tuple[torch.Tensor, ...],
) -> tuple[tuple[dict[int, Any], ...]] | None:
    """Build ``dynamic_shapes`` for ``export(model, (tuple_in,), dynamo=True)``.

    Args:
        tuple_in: The single tuple-of-tensors argument passed to dynamo
            export.

    Returns:
        A one-element tuple wrapping a per-tensor ``{0: Dim("batch")}`` dict,
        or ``None`` if ``torch.export.Dim`` is unavailable on this torch
        version.
    """
    if _TorchExportDim is None:
        return None
    batch = _TorchExportDim("batch")
    return (tuple({0: batch} for _ in tuple_in),)


def _onnx_dynamo_export_error_should_retry_legacy(exc: BaseException) -> bool:
    """Return whether a dynamo export failure should retry with ``dynamo=False``."""
    if isinstance(exc, (ImportError, ModuleNotFoundError)):
        return True
    msg = str(exc).lower()
    return (
        "onnxscript" in msg
        or "convertversionpass" in msg
        or "version conversion pass" in msg
        or "model contains functions" in msg
        or "passerror" in msg
        or "failed to convert 'dynamic_axes'" in msg
        or "treespec.unflatten" in msg
        # torch.export's Dim-based dynamic_shapes validation has changed shape
        # across torch releases (e.g. rejecting a Dim keyed by position for a
        # tuple-of-tensors arg on some versions); treat that whole class of
        # torch.export capture failures as dynamo-not-usable-here rather than
        # a genuine model bug, and fall back to the legacy exporter.
        or "torchexporterror" in msg
        or "unexpected dimension" in msg
    )


def _expand_batch_for_onnx_export(
    tensors: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, ...]:
    """Expand any size-1 batch dimension to size 2 so it isn't baked into the graph."""
    out: list[torch.Tensor] = []
    for inp in tensors:
        if inp.size(0) > 1:
            out.append(inp)
        else:
            out.append(inp.repeat(_ONNX_EXPAND_BATCH_SIZE, *[1] * (inp.dim() - 1)))
    return tuple(out)


def _force_onnx_io_shapes_from_schema(
    onnx_path: str,
    model_schemas: list[ModelSchema | None],
    batch_dim_param: str = "b",
) -> None:
    """Override ONNX graph IO shapes to match the schema's static dims.

    PyTorch's ONNX shape inference can drop a fixed non-batch dimension and
    emit ``[-1, -1]``; Triton's onnxruntime backend then rejects the model
    because its config declares a static shape. For every graph input/output
    whose name matches a schema item with a known shape, this overwrites dim
    0 with ``batch_dim_param`` and the remaining dims with the schema's
    static values. Names absent from every schema are left untouched.

    Args:
        onnx_path: Path to the ``.onnx`` file to rewrite in place.
        model_schemas: Schemas to source static shapes from (``None`` entries
            are skipped).
        batch_dim_param: Symbolic name to use for the batch dimension.
    """
    name_to_shape: dict[str, list[int]] = {}
    for schema in model_schemas:
        if schema is None:
            continue
        for item in list(schema.input_schema) + list(schema.output_schema):
            if item.shape is None:
                continue
            name_to_shape[item.name] = [int(s) for s in item.shape]

    if not name_to_shape:
        return

    model_proto = onnx.load(onnx_path)

    def _override(value_info: Any) -> None:
        schema_shape = name_to_shape.get(value_info.name)
        if schema_shape is None:
            return
        tensor_type = value_info.type.tensor_type
        expected_rank = 1 + len(schema_shape)
        existing_rank = len(tensor_type.shape.dim)
        if existing_rank != expected_rank:
            _logger.warning(
                "Skipping ONNX shape override for '%s': rank %d in graph != "
                "%d expected from schema.",
                value_info.name,
                existing_rank,
                expected_rank,
            )
            return
        tensor_type.shape.dim[0].ClearField("dim_value")
        tensor_type.shape.dim[0].dim_param = batch_dim_param
        for i, dim_size in enumerate(schema_shape, start=1):
            tensor_type.shape.dim[i].ClearField("dim_param")
            tensor_type.shape.dim[i].dim_value = dim_size

    for value_info in list(model_proto.graph.input) + list(model_proto.graph.output):
        _override(value_info)

    onnx.save(model_proto, onnx_path)


def _onnx_export_input_preserver(
    input_tensors: tuple[torch.Tensor, ...],
    *,
    ref_dtype: torch.dtype,
    ref_device: torch.device,
) -> torch.Tensor:
    """Return a scalar zero that data-depends on every tensor in ``input_tensors``.

    Used to keep every ONNX graph input alive even when the traced module
    doesn't otherwise use all of them (the exporter can otherwise prune an
    unused input from the graph).
    """
    acc = torch.zeros((), dtype=ref_dtype, device=ref_device)
    for tensor in input_tensors:
        acc = acc + (tensor * 0).sum().to(dtype=ref_dtype)
    return acc


def _onnx_export_attach_inputs_to_output(
    out: object, input_tensors: tuple[torch.Tensor, ...]
) -> object:
    """Add a zero-valued data dependency from each input into one output branch.

    Args:
        out: The module's forward output — a Tensor, NamedTuple, dict, tuple,
            or list.
        input_tensors: The tensors that must remain live graph inputs.

    Returns:
        ``out`` with one branch adjusted to depend on every input tensor.
        Returned unchanged if ``input_tensors`` is empty or ``out``'s shape
        isn't recognized.
    """
    if not input_tensors:
        return out
    if isinstance(out, torch.Tensor):
        preserver = _onnx_export_input_preserver(
            input_tensors, ref_dtype=out.dtype, ref_device=out.device
        )
        return out + preserver
    fields = getattr(out, "_fields", None)
    if fields:
        vals = list(out)
        for i, v in enumerate(vals):
            if isinstance(v, torch.Tensor):
                preserver = _onnx_export_input_preserver(
                    input_tensors, ref_dtype=v.dtype, ref_device=v.device
                )
                vals[i] = v + preserver
                return type(out)(*vals)
    if isinstance(out, dict):
        d = dict(out)
        for k, v in d.items():
            if isinstance(v, torch.Tensor):
                preserver = _onnx_export_input_preserver(
                    input_tensors, ref_dtype=v.dtype, ref_device=v.device
                )
                d[k] = v + preserver
                return d
        return out
    if isinstance(out, (tuple, list)):
        seq = list(out)
        for i, v in enumerate(seq):
            if isinstance(v, torch.Tensor):
                preserver = _onnx_export_input_preserver(
                    input_tensors, ref_dtype=v.dtype, ref_device=v.device
                )
                seq[i] = v + preserver
                return tuple(seq) if isinstance(out, tuple) else seq
    return out


class _FusedOnnxTupleWrapper(torch.nn.Module):
    """Adapts a dict-input fused module to positional tensors, for legacy ONNX export.

    ``torch.onnx.export`` (legacy path) traces with positional args; the
    fused model takes a single ``dict[str, Tensor]``. This wrapper converts
    ``*inputs`` (in ``input_key_order``) into that dict, calls the inner
    module, and preserves all inputs in the traced graph.
    """

    def __init__(self, inner: torch.nn.Module, input_key_order: list[str]) -> None:
        """Initialize the wrapper.

        Args:
            inner: The fused module to wrap.
            input_key_order: Feature name for each positional argument, in
                order.
        """
        super().__init__()
        self.inner = inner
        self._input_key_order = input_key_order

    def forward(self, *inputs: torch.Tensor) -> object:
        """Merge positional ``inputs`` into a dict and call the inner module."""
        merged = dict(zip(self._input_key_order, inputs))
        out = self.inner(merged)
        return _onnx_export_attach_inputs_to_output(out, inputs)


class _FusedOnnxDynamoTupleWrapper(torch.nn.Module):
    """Same as :class:`_FusedOnnxTupleWrapper`, for the dynamo ONNX export path.

    ``torch.onnx.export(..., dynamo=True)`` treats ``args=(tuple_in,)`` as a
    single pytree argument (a tuple of tensors), so this wrapper takes one
    tuple argument instead of ``*args``.
    """

    def __init__(self, inner: torch.nn.Module, input_key_order: list[str]) -> None:
        """Initialize the wrapper.

        Args:
            inner: The fused module to wrap.
            input_key_order: Feature name for each tuple element, in order.
        """
        super().__init__()
        self.inner = inner
        self._input_key_order = input_key_order

    def forward(self, inputs_tuple: tuple[torch.Tensor, ...]) -> object:
        """Merge the tuple argument into a dict and call the inner module."""
        merged = dict(zip(self._input_key_order, inputs_tuple))
        out = self.inner(merged)
        return _onnx_export_attach_inputs_to_output(out, inputs_tuple)


def _run_export_with_retry(
    export_args: tuple[Any, Any, str],
    export_kwargs: dict[str, Any],
    legacy_export_kwargs: dict[str, Any],
    use_dynamo: bool,
    use_tuple_wrapper: bool,
    model: torch.nn.Module,
    input_key_order: list[str] | None,
) -> None:
    """Run ``torch.onnx.export``, retrying with ``dynamo=False`` on failure.

    Args:
        export_args: ``(model_or_wrapper, sample_args, dest_path)`` for the
            dynamo attempt.
        export_kwargs: Keyword arguments for the dynamo (or sole) export
            attempt.
        legacy_export_kwargs: Keyword arguments for the legacy fallback
            export.
        use_dynamo: Whether to attempt the dynamo exporter first.
        use_tuple_wrapper: Whether inputs are wrapped via
            :class:`_FusedOnnxDynamoTupleWrapper`/:class:`_FusedOnnxTupleWrapper`.
        model: The unwrapped module, used for the legacy fallback.
        input_key_order: Feature name per positional/tuple input; required
            when ``use_tuple_wrapper`` is ``True``.
    """
    if use_dynamo and "dynamic_shapes" in export_kwargs and use_tuple_wrapper:
        dynamo_wrapped = _FusedOnnxDynamoTupleWrapper(model, input_key_order)
        dynamo_wrapped.eval()
        try:
            with _disable_transformer_encoder_fused_fastpath_for_onnx(dynamo_wrapped):
                sample_args = export_args[1]
                tuple_arg = (
                    sample_args[0] if isinstance(sample_args, tuple) else sample_args
                )
                torch.onnx.export(
                    dynamo_wrapped,
                    (tuple_arg,),
                    export_args[2],
                    **export_kwargs,
                )
        except Exception as e:
            if not _onnx_dynamo_export_error_should_retry_legacy(e):
                raise
            _logger.warning(
                "ONNX dynamo export failed (%s); retrying with legacy "
                "torch.onnx.export.",
                e,
            )
            with _disable_transformer_encoder_fused_fastpath_for_onnx(export_args[0]):
                model, sample_args, dest_path = export_args
                torch.onnx.export(model, sample_args, dest_path, **legacy_export_kwargs)
    else:
        with _disable_transformer_encoder_fused_fastpath_for_onnx(export_args[0]):
            model, sample_args, dest_path = export_args
            torch.onnx.export(model, sample_args, dest_path, **export_kwargs)


def _export_fused_onnx(
    model: torch.nn.Module,
    dest_path: str,
    sample_inputs: tuple[torch.Tensor, ...],
    input_names: list[str],
    output_names: list[str],
    model_schemas: list[ModelSchema | None],
    input_key_order: list[str],
) -> str:
    """Export a fused ``nn.Module`` to ONNX with dynamo-first, legacy-fallback.

    Applies MHA fastpath disable, batch-size expansion, dynamo/legacy
    fallback, and schema-driven IO shape normalization — the same fixes
    the non-fused Triton packager's ONNX export path applies.

    Args:
        model: The fused module to export (already in eval mode).
        dest_path: Local path where the ``.onnx`` file will be saved.
        sample_inputs: Trace tensors in ``input_key_order``.
        input_names: ONNX graph input names, in ``input_key_order``.
        output_names: ONNX graph output names.
        model_schemas: Schemas used to force static ONNX IO shapes.
        input_key_order: Feature name per element of ``sample_inputs``.

    Returns:
        ``dest_path``.
    """
    os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)

    sample_inputs = _expand_batch_for_onnx_export(sample_inputs)

    dynamic_axes: dict[str, dict[int, str]] = {
        name: {0: "b"} for name in list(input_names) + list(output_names)
    }

    export_sig = inspect.signature(torch.onnx.export)
    supports_dynamo = "dynamo" in export_sig.parameters
    use_dynamo = supports_dynamo and _onnx_dynamo_exporter_dependencies_available()

    wrapped: torch.nn.Module = _FusedOnnxTupleWrapper(model, input_key_order)
    wrapped.eval()

    export_kwargs: dict[str, Any] = {
        "input_names": list(input_names),
        "output_names": list(output_names) if output_names else None,
        "do_constant_folding": False,
    }
    if use_dynamo:
        export_kwargs["dynamo"] = True
        export_kwargs["opset_version"] = _ONNX_OPSET_DYNAMO
        dynamic_shapes = _onnx_dynamo_dynamic_shapes_for_tuple_arg(sample_inputs)
        if dynamic_shapes is not None:
            export_kwargs["dynamic_shapes"] = dynamic_shapes
        else:
            export_kwargs["dynamic_axes"] = dynamic_axes or None
    else:
        export_kwargs["dynamic_axes"] = dynamic_axes or None
        export_kwargs["opset_version"] = _ONNX_OPSET_LEGACY

    legacy_export_kwargs: dict[str, Any] = {
        "input_names": list(input_names),
        "output_names": list(output_names) if output_names else None,
        "do_constant_folding": False,
        "dynamic_axes": dynamic_axes or None,
        "opset_version": _ONNX_OPSET_LEGACY,
    }
    if supports_dynamo:
        legacy_export_kwargs["dynamo"] = False

    export_args = (wrapped, sample_inputs, dest_path)

    _run_export_with_retry(
        export_args,
        export_kwargs,
        legacy_export_kwargs,
        use_dynamo,
        use_tuple_wrapper=True,
        model=model,
        input_key_order=input_key_order,
    )

    try:
        _force_onnx_io_shapes_from_schema(dest_path, model_schemas)
    except Exception as e:
        _logger.warning("Could not normalize ONNX IO shapes from schema: %s", e)

    return dest_path


# ---------------------------------------------------------------------------
# Public fuse API
# ---------------------------------------------------------------------------


def get_predictor_output_field_order(
    model_path: str,
    model_class: str,
    hyperparameters: dict[str, Any],
    model_schema: ModelSchema | None,
) -> list[str] | None:
    """Recover a NamedTuple predictor's output ``_fields`` order.

    Tries two strategies in order:

    1. Inspect ``forward()``'s return-type annotation for a ``_fields``
       attribute. Works without running the model, so it needs no sample
       input.
    2. Run a forward pass with a synthetic sample input and read ``_fields``
       from the live output. Used only when the annotation strategy fails.

    Args:
        model_path: Local path to the predictor model.
        model_class: Dotted class name for the predictor.
        hyperparameters: Constructor kwargs for the predictor.
        model_schema: Predictor model schema, used to build a synthetic
            sample input for strategy 2.

    Returns:
        The output field names in order, or ``None`` if the predictor's
        output isn't a NamedTuple or loading/inference fails. Callers should
        fall back to the original schema order when ``None`` is returned.
    """
    try:
        module = _load_module_from_path(model_path, model_class, hyperparameters)

        try:
            sig = inspect.signature(module.forward)
            return_annotation = sig.return_annotation
            if return_annotation is not inspect.Parameter.empty and hasattr(
                return_annotation, "_fields"
            ):
                _logger.info(
                    "Predictor output field order recovered from forward() "
                    "return annotation."
                )
                return list(return_annotation._fields)
        except Exception:
            pass

        sample_input = _build_fused_sample_input(None, model_schema)
        with torch.no_grad():
            if _forward_accepts_dict(module):
                output = module(sample_input)
            else:
                forward_params = _forward_param_order(module)
                args = [sample_input[p] for p in forward_params if p in sample_input]
                output = module(*args)
        if hasattr(output, "_fields"):
            return list(output._fields)
        _logger.info("Predictor output has no _fields; output_schema order unchanged.")
        return None
    except Exception as e:
        _logger.warning(
            "Could not determine predictor output field order: %s. Output "
            "schema unchanged.",
            e,
        )
        return None


@contextmanager
def _pl_jit_scripting_guard() -> Iterator[None]:
    """Best-effort wrapper around PyTorch Lightning's private jit-scripting context.

    ``pl.core.module._jit_is_scripting`` avoids tracing LightningModule-only
    hooks during ``torch.jit.trace``. It's a Lightning-internal API with no
    public equivalent and has moved across Lightning releases before, so this
    degrades to a no-op if the installed pytorch_lightning no longer exposes
    it, rather than breaking the fused-model export.
    """
    module_attr = getattr(getattr(pl, "core", None), "module", None) if pl else None
    jit_is_scripting = getattr(module_attr, "_jit_is_scripting", None)
    if jit_is_scripting is None:
        yield
        return
    with jit_is_scripting():
        yield


def fuse_models_to_torchscript(
    torch_model_path: str,
    tx_model_path: str,
    model_class: str,
    hyperparameters: dict[str, Any],
    tx_model_class: str,
    tx_hyperparameters: dict[str, Any],
    dest_path: str,
    tx_model_schema: ModelSchema | None = None,
    model_schema: ModelSchema | None = None,
) -> str:
    """Fuse the predictor and transform models and save as TorchScript.

    Loads both models from local paths, composes them as a ``FusedModel``
    with schema-driven merge (transform input/output, predictor input), and
    exports to ``dest_path`` as TorchScript.

    Args:
        torch_model_path: Local path to the predictor model.
        tx_model_path: Local path to the native-transform model.
        model_class: Dotted class name for the predictor.
        hyperparameters: Constructor kwargs for the predictor.
        tx_model_class: Dotted class name for the transform.
        tx_hyperparameters: Constructor kwargs for the transform.
        dest_path: Local path where the fused TorchScript model is saved.
        tx_model_schema: Native-transform model schema.
        model_schema: Predictor model schema.

    Returns:
        ``dest_path``.
    """
    fused, sample_input, _ = _build_fused_model_and_sample(
        torch_model_path,
        tx_model_path,
        model_class,
        hyperparameters,
        tx_model_class,
        tx_hyperparameters,
        tx_model_schema=tx_model_schema,
        model_schema=model_schema,
    )

    os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)

    trace_device = next(iter(sample_input.values())).device
    _logger.info("Tracing fused model on device: %s", trace_device)
    # torch.no_grad ensures a consistent eval-mode fast path in
    # nn.TransformerEncoderLayer during trace; without it, trace vs.
    # verification runs can diverge.
    with _pl_jit_scripting_guard(), torch.no_grad():
        traced = torch.jit.trace(fused, (sample_input,))
        torch.jit.save(traced, dest_path)
    return dest_path


def fuse_models_to_onnx(
    torch_model_path: str,
    tx_model_path: str,
    model_class: str,
    hyperparameters: dict[str, Any],
    tx_model_class: str,
    tx_hyperparameters: dict[str, Any],
    dest_path: str,
    tx_model_schema: ModelSchema | None = None,
    model_schema: ModelSchema | None = None,
) -> str:
    """Fuse the predictor and transform models and save as ONNX.

    Uses the same composition rules as :func:`fuse_models_to_torchscript`,
    then exports with the dynamo-first/legacy-fallback ONNX pipeline (see the
    module docstring) so fused and non-fused models get equivalent export
    quality.

    Args:
        torch_model_path: Local path to the predictor model.
        tx_model_path: Local path to the native-transform model.
        model_class: Dotted class name for the predictor.
        hyperparameters: Constructor kwargs for the predictor.
        tx_model_class: Dotted class name for the transform.
        tx_hyperparameters: Constructor kwargs for the transform.
        dest_path: Local path where the fused ``.onnx`` model is saved.
        tx_model_schema: Native-transform model schema.
        model_schema: Predictor model schema.

    Returns:
        ``dest_path``.
    """
    fused, sample_input, input_key_order = _build_fused_model_and_sample(
        torch_model_path,
        tx_model_path,
        model_class,
        hyperparameters,
        tx_model_class,
        tx_hyperparameters,
        tx_model_schema=tx_model_schema,
        model_schema=model_schema,
    )

    tuple_in = tuple(sample_input[k] for k in input_key_order)
    input_names = list(input_key_order)
    output_names = [
        item.name for item in (model_schema.output_schema if model_schema else [])
    ]

    try:
        with torch.no_grad():
            out = fused(dict(zip(input_key_order, tuple_in)))
        if hasattr(out, "_fields"):
            output_names = list(out._fields)
    except Exception as e:
        _logger.warning(
            "Could not infer ONNX output names from forward sample run: %s", e
        )

    return _export_fused_onnx(
        model=fused,
        dest_path=dest_path,
        sample_inputs=tuple_in,
        input_names=input_names,
        output_names=output_names,
        model_schemas=[tx_model_schema, model_schema],
        input_key_order=input_key_order,
    )


def compute_python_fuse_metadata(
    torch_model_path: str,
    model_class: str,
    hyperparameters: dict[str, Any],
    tx_model_schema: ModelSchema | None,
    model_schema: ModelSchema | None,
) -> tuple[list[str], list[str], bool]:
    """Compute fuse routing metadata for Python-backend packaging.

    Loads the predictor to inspect its ``forward()`` signature and derives
    the key lists and predictor call style needed to reconstruct the fused
    model at serve time, using the same key-ordering logic as
    :func:`fuse_models_to_torchscript` but without tracing.

    Args:
        torch_model_path: Local path to the predictor model.
        model_class: Dotted class name for the predictor.
        hyperparameters: Constructor kwargs for the predictor.
        tx_model_schema: Native-transform model schema.
        model_schema: Predictor model schema.

    Returns:
        A tuple ``(transform_input_keys, predictor_input_keys,
        predictor_takes_dict)``: the feature names routed to the transform,
        the feature names routed to the predictor (after merge), and whether
        the predictor's ``forward()`` takes a single dict argument.
    """
    hyperparameters = hyperparameters or {}
    pred_module = _load_module_from_path(torch_model_path, model_class, hyperparameters)
    predictor_takes_dict = _forward_accepts_dict(pred_module)

    transform_input_keys = _schema_input_keys(tx_model_schema)
    predictor_input_keys = _schema_input_keys(model_schema)
    predictor_input_keys = _align_predictor_input_keys(
        pred_module, predictor_input_keys, predictor_takes_dict
    )

    return transform_input_keys, predictor_input_keys, predictor_takes_dict


def fuse_models_to_python(
    torch_model_path: str,
    tx_model_path: str,
    model_class: str,
    hyperparameters: dict[str, Any],
    tx_hyperparameters: dict[str, Any],
    dest_path: str,
    tx_model_schema: ModelSchema | None = None,
    model_schema: ModelSchema | None = None,
) -> tuple[str, str, dict[str, Any]]:
    """Fuse predictor and transform into a combined state dict for serving.

    Unlike :func:`fuse_models_to_torchscript`/:func:`fuse_models_to_onnx`,
    which trace a single ``nn.Module``, this path keeps weights separate and
    reconstructs the ``FusedModel`` at serve time from the returned
    hyperparameters spec. The predictor and transform state dicts are
    combined with submodule prefixes (``predictor_module.*``,
    ``transform_module.*``) matching ``FusedModel``'s attribute names.

    Args:
        torch_model_path: Local path to the predictor model (state_dict or
            full module).
        tx_model_path: Local path to the native-transform model (full
            ``nn.Module``).
        model_class: Dotted class name for the predictor.
        hyperparameters: Constructor kwargs for the predictor.
        tx_hyperparameters: The transform's ``to_dict()`` output.
        dest_path: Local path where the combined state dict is saved.
        tx_model_schema: Native-transform model schema.
        model_schema: Predictor model schema.

    Returns:
        A tuple ``(dest_path, fused_model_class, fused_hyperparameters)``:
        the saved state dict path, ``FusedModel``'s dotted class name, and
        the serve-time reconstruction spec.

    Raises:
        NotImplementedError: Building ``fused_hyperparameters["transform_module"]``
            requires the native-transform package (see
            :func:`_build_tx_hydra_spec`), which is not yet available in OSS.
    """
    hyperparameters = hyperparameters or {}
    tx_hyperparameters = tx_hyperparameters or {}

    transform_input_keys, predictor_input_keys, predictor_takes_dict = (
        compute_python_fuse_metadata(
            torch_model_path,
            model_class,
            hyperparameters,
            tx_model_schema,
            model_schema,
        )
    )

    predictor_sd = torch.load(torch_model_path, map_location="cpu", weights_only=True)
    if "state_dict" in predictor_sd and isinstance(predictor_sd["state_dict"], dict):
        predictor_sd = predictor_sd["state_dict"]

    tx_obj = torch.load(tx_model_path, map_location="cpu", weights_only=False)
    tx_sd = tx_obj.state_dict()
    del tx_obj

    combined_sd = {f"predictor_module.{k}": v for k, v in predictor_sd.items()}
    combined_sd.update({f"transform_module.{k}": v for k, v in tx_sd.items()})

    os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
    torch.save(combined_sd, dest_path)

    tx_spec = _build_tx_hydra_spec(tx_hyperparameters)

    fused_model_class = f"{FusedModel.__module__}.{FusedModel.__qualname__}"
    fused_hyperparameters = {
        "predictor_module": {"_target_": model_class, **hyperparameters},
        "transform_module": tx_spec,
        "transform_input_keys": transform_input_keys,
        "predictor_input_keys": predictor_input_keys,
        "predictor_takes_dict": predictor_takes_dict,
    }

    return dest_path, fused_model_class, fused_hyperparameters


def build_fused_sample_data(
    tx_sample_data: list[dict[str, Any]] | None,
    predictor_sample_data: list[dict[str, Any]] | None,
    fused_input_cols: set[str],
) -> list[dict[str, Any]]:
    """Merge transform + predictor sample data and filter to the fused input schema.

    The predictor is trained on post-transform data, so its sample data
    includes transform output columns. The fused model's input schema only
    has pre-transform columns, so this merges both sample sets and filters
    to the fused input columns.

    Args:
        tx_sample_data: Native-transform model's sample data.
        predictor_sample_data: Predictor model's sample data.
        fused_input_cols: Column names in the fused model's input schema.

    Returns:
        A single-element list containing the merged, filtered sample dict.
        When a key appears in both inputs, ``tx_sample_data`` wins — it holds
        the raw pre-transform values the fused model's input schema expects.
    """
    merged: dict[str, Any] = {}
    for sample in predictor_sample_data or []:
        merged.update(sample)
    for sample in tx_sample_data or []:
        merged.update(sample)
    return [{k: v for k, v in merged.items() if k in fused_input_cols}]
