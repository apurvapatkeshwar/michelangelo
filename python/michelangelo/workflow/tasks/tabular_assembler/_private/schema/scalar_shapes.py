"""Scalar-shape normalization for the torch tabular assembler.

No internal ``uber-one`` equivalent -- this is an OSS-only defensive
backstop, not a ported function. It originally guarded against
``ColumnConfig.shape`` defaulting to ``[]`` (an OSS-only deviation from
internal's required, no-default field); that default has since been
removed to match internal, so this module now only matters if a caller
explicitly constructs a schema item with ``shape=[]``.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from michelangelo.lib.model_manager.schema import ModelSchema, ModelSchemaItem


def normalize_scalar_shapes(
    schema: ModelSchema, sample_data: list[dict[str, Any]] | None
) -> tuple[ModelSchema, list[dict[str, Any]] | None]:
    """Return copies of ``schema``/``sample_data`` with scalar shapes set to ``[1]``.

    ``ColumnConfig``'s documented usage for a scalar column omits ``shape``,
    defaulting it to ``[]`` (see ``workflow.schema.tabular_trainer
    .ColumnConfig``), and ``tabular_trainer`` keeps the matching sample-data
    value at its natural rank-0 shape to match. Triton's schema validation
    (``validate_model_schema_item``) requires a non-empty shape, and its
    sample-data validation requires the sample's rank to equal the schema
    shape's length -- so ``schema`` and ``sample_data`` must be renormalized
    together, not independently, or the two fall out of sync.

    Args:
        schema: Schema to normalize.
        sample_data: Sample inference inputs to normalize alongside
            ``schema``, or ``None``.

    Returns:
        ``(normalized_schema, normalized_sample_data)``. Any schema item with
        an empty ``shape`` is replaced by a copy with ``shape=[1]``; any
        ``sample_data`` value for a field renamed this way is reshaped from
        scalar to a 1-element array to match.

    Raises:
        ValueError: If a field marked scalar in ``schema`` (empty ``shape``)
            has a sample-data value with more than one element -- schema and
            sample data have fallen out of sync, and packaging with a
            mismatched shape would silently produce a corrupted package
            rather than a clear error at this boundary.
    """
    scalar_fields = {
        item.name
        for item in (
            *schema.input_schema,
            *schema.feature_store_features_schema,
            *schema.output_schema,
        )
        if not item.shape
    }

    def _normalized_item(item: ModelSchemaItem) -> ModelSchemaItem:
        return item if item.shape else replace(item, shape=[1])

    normalized_schema = ModelSchema(
        input_schema=[_normalized_item(i) for i in schema.input_schema],
        feature_store_features_schema=[
            _normalized_item(i) for i in schema.feature_store_features_schema
        ],
        output_schema=[_normalized_item(i) for i in schema.output_schema],
    )

    if not sample_data or not scalar_fields:
        return normalized_schema, sample_data

    def _normalized_value(name: str, value: Any) -> Any:
        if name not in scalar_fields:
            return value
        try:
            return np.reshape(value, (1,))
        except ValueError as exc:
            raise ValueError(
                f"sample_data field {name!r} is marked scalar in schema "
                f"(shape=[]) but its value has more than one element "
                f"(shape {np.asarray(value).shape}); schema and sample_data "
                "have fallen out of sync."
            ) from exc

    normalized_sample_data = [
        {name: _normalized_value(name, value) for name, value in record.items()}
        for record in sample_data
    ]
    return normalized_schema, normalized_sample_data
