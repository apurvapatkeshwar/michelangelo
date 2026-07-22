"""Tests for serialize_additional_imports."""

import os
import tempfile
from unittest import TestCase

from michelangelo.lib.model_manager._private.packager.custom_triton.additional_imports import (  # noqa: E501
    serialize_additional_imports,
)


class SerializeAdditionalImportsTest(TestCase):
    """Tests serialization of dynamically-imported module prefixes."""

    def test_serialize_additional_imports_adds_files(self):
        """It copies the resolved prefix's source file into the target dir."""
        prefix = (
            "michelangelo.lib.model_manager._private.utils.module_finder.tests."
            "fixtures.simple_module"
        )

        with tempfile.TemporaryDirectory() as target_dir:
            serialize_additional_imports(
                [prefix],
                target_dir,
                include_import_prefixes=["michelangelo"],
            )

            expected_file = os.path.join(
                target_dir,
                "michelangelo",
                "lib",
                "model_manager",
                "_private",
                "utils",
                "module_finder",
                "tests",
                "fixtures",
                "simple_module.py",
            )
            self.assertTrue(os.path.exists(expected_file))

            with open(expected_file) as f:
                content = f.read()
            self.assertIn("module_attr", content)

    def test_serialize_additional_imports_none_is_noop(self):
        """It does nothing when additional_import_prefixes is None."""
        with tempfile.TemporaryDirectory() as target_dir:
            serialize_additional_imports(None, target_dir)
            self.assertEqual(os.listdir(target_dir), [])

    def test_serialize_additional_imports_empty_list_is_noop(self):
        """It does nothing when additional_import_prefixes is empty."""
        with tempfile.TemporaryDirectory() as target_dir:
            serialize_additional_imports([], target_dir)
            self.assertEqual(os.listdir(target_dir), [])

    def test_serialize_additional_imports_multiple_prefixes(self):
        """It resolves and copies every prefix in the given list."""
        prefixes = [
            "michelangelo.lib.model_manager._private.utils.module_finder.tests."
            "fixtures.simple_module",
            "michelangelo.lib.model_manager._private.utils.module_finder.tests."
            "fixtures.module_with_imports",
        ]

        with tempfile.TemporaryDirectory() as target_dir:
            serialize_additional_imports(
                prefixes,
                target_dir,
                include_import_prefixes=["michelangelo"],
            )

            base = os.path.join(
                target_dir,
                "michelangelo",
                "lib",
                "model_manager",
                "_private",
                "utils",
                "module_finder",
                "tests",
                "fixtures",
            )
            self.assertTrue(os.path.exists(os.path.join(base, "simple_module.py")))
            self.assertTrue(
                os.path.exists(os.path.join(base, "module_with_imports.py"))
            )
