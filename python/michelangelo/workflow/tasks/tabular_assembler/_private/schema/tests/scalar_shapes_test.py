"""Unit tests for ``...tabular_assembler._private.schema.scalar_shapes``."""

from __future__ import annotations

import unittest

import numpy as np

from michelangelo.lib.model_manager.schema import DataType, ModelSchema, ModelSchemaItem
from michelangelo.workflow.tasks.tabular_assembler._private.schema.scalar_shapes import (  # noqa: E501
    normalize_scalar_shapes,
)


class NormalizeScalarShapesTest(unittest.TestCase):
    """Tests for ``normalize_scalar_shapes``.

    Regression coverage for a real bug, now closed off at the source but
    kept as defensive hardening: ``ColumnConfig.shape`` used to default to
    ``[]``, so a plain tabular model built the then-documented way
    (``ColumnConfig("torch.float32")`` with no ``shape`` argument, the
    common scalar-feature case) produced ``ModelSchemaItem(shape=[])``
    entries that Triton's schema validator rejects outright
    (``ValueError: Shape must be provided for item: ...``), so
    ``torch_assembler`` could never package the most common kind of tabular
    model. ``ColumnConfig.shape`` is now a required field with no default
    (matching internal), so this can no longer happen by omission -- but
    ``normalize_scalar_shapes`` still widens an explicit ``shape=[]`` to
    ``[1]`` (and reshapes any matching sample-data values) before
    packaging, as a defensive backstop.
    """

    def test_empty_shape_normalized_to_one(self):
        """A scalar (``shape=[]``) item is widened to ``shape=[1]``."""
        schema = ModelSchema(
            input_schema=[
                ModelSchemaItem(name="scalar_in", data_type=DataType.FLOAT, shape=[]),
            ],
            output_schema=[
                ModelSchemaItem(name="out", data_type=DataType.FLOAT, shape=[1]),
            ],
        )
        normalized_schema, _ = normalize_scalar_shapes(schema, None)
        self.assertEqual(normalized_schema.input_schema[0].shape, [1])
        self.assertEqual(normalized_schema.output_schema[0].shape, [1])

    def test_non_empty_shape_left_unchanged(self):
        """An item with a real shape is untouched (same object, not a copy)."""
        schema = ModelSchema(
            input_schema=[
                ModelSchemaItem(name="vec_in", data_type=DataType.FLOAT, shape=[8]),
            ],
        )
        normalized_schema, _ = normalize_scalar_shapes(schema, None)
        self.assertIs(normalized_schema.input_schema[0], schema.input_schema[0])

    def test_sample_data_for_scalar_field_reshaped_to_match(self):
        """Sample-data values for a normalized field are reshaped to ``(1,)``."""
        schema = ModelSchema(
            input_schema=[
                ModelSchemaItem(name="scalar_in", data_type=DataType.FLOAT, shape=[]),
            ],
        )
        sample_data = [{"scalar_in": np.float32(3.0)}]
        _, normalized_sample_data = normalize_scalar_shapes(schema, sample_data)
        self.assertEqual(normalized_sample_data[0]["scalar_in"].shape, (1,))
        np.testing.assert_array_equal(
            normalized_sample_data[0]["scalar_in"], np.array([3.0])
        )

    def test_scalar_field_with_multi_element_value_raises_clear_error(self):
        """A scalar/sample_data mismatch raises a clear error, not a raw numpy one."""
        schema = ModelSchema(
            input_schema=[
                ModelSchemaItem(name="scalar_in", data_type=DataType.FLOAT, shape=[]),
            ],
        )
        sample_data = [{"scalar_in": np.array([1.0, 2.0, 3.0])}]
        with self.assertRaises(ValueError) as ctx:
            normalize_scalar_shapes(schema, sample_data)
        self.assertIn("scalar_in", str(ctx.exception))
        self.assertIn("fallen out of sync", str(ctx.exception))

    def test_sample_data_for_non_scalar_field_untouched(self):
        """Sample-data values for a non-scalar field pass through unchanged."""
        schema = ModelSchema(
            input_schema=[
                ModelSchemaItem(name="vec_in", data_type=DataType.FLOAT, shape=[2]),
            ],
        )
        sample_data = [{"vec_in": np.array([1.0, 2.0])}]
        _, normalized_sample_data = normalize_scalar_shapes(schema, sample_data)
        self.assertIs(normalized_sample_data[0]["vec_in"], sample_data[0]["vec_in"])

    def test_none_sample_data_returns_none(self):
        """``sample_data=None`` returns ``None`` rather than an empty list."""
        schema = ModelSchema(
            input_schema=[
                ModelSchemaItem(name="scalar_in", data_type=DataType.FLOAT, shape=[]),
            ],
        )
        _, normalized_sample_data = normalize_scalar_shapes(schema, None)
        self.assertIsNone(normalized_sample_data)


if __name__ == "__main__":
    unittest.main()
