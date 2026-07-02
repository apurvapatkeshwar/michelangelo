"""Unit tests for ``download_file_tree`` using real local paths and fsspec."""

from __future__ import annotations

import os
import tempfile
import unittest

from michelangelo.workflow.tasks.tabular_assembler._private.file.download import (
    download_file_tree,
)


class DownloadFileTreeTest(unittest.TestCase):
    """Tests for ``download_file_tree``."""

    def test_downloads_single_file_into_existing_dest_dir(self) -> None:
        with (
            tempfile.TemporaryDirectory() as src_root,
            tempfile.TemporaryDirectory() as dest_dir,
        ):
            src_file = os.path.join(src_root, "model.bin")
            with open(src_file, "wb") as f:
                f.write(b"payload-bytes")

            download_file_tree(src_file, dest_dir)

            out = os.path.join(dest_dir, "model.bin")
            self.assertTrue(os.path.isfile(out), f"expected {out!r} to exist")
            with open(out, "rb") as f:
                self.assertEqual(f.read(), b"payload-bytes")

    def test_creates_dest_dir_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as parent:
            src_file = os.path.join(parent, "src.txt")
            with open(src_file, "w", encoding="utf-8") as f:
                f.write("ok")

            dest_dir = os.path.join(parent, "nested", "out")
            self.assertFalse(os.path.exists(dest_dir))

            download_file_tree(src_file, dest_dir)

            self.assertTrue(os.path.isdir(dest_dir))
            self.assertTrue(os.path.isfile(os.path.join(dest_dir, "src.txt")))

    def test_downloads_directory_contents_recursively(self) -> None:
        with (
            tempfile.TemporaryDirectory() as src_root,
            tempfile.TemporaryDirectory() as dest_dir,
        ):
            sub = os.path.join(src_root, "inner")
            os.makedirs(sub)
            with open(os.path.join(src_root, "root.txt"), "w", encoding="utf-8") as f:
                f.write("root")
            with open(os.path.join(sub, "nested.txt"), "w", encoding="utf-8") as f:
                f.write("nested")

            download_file_tree(src_root, dest_dir)

            root_out = os.path.join(dest_dir, "root.txt")
            nested_out = os.path.join(dest_dir, "inner", "nested.txt")
            self.assertTrue(
                os.path.isfile(root_out),
                f"missing {root_out!r}, got {os.listdir(dest_dir)}",
            )
            self.assertTrue(os.path.isdir(os.path.join(dest_dir, "inner")))
            self.assertTrue(os.path.isfile(nested_out))
            with open(root_out, encoding="utf-8") as f:
                self.assertEqual(f.read(), "root")
            with open(nested_out, encoding="utf-8") as f:
                self.assertEqual(f.read(), "nested")
