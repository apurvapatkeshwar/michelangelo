"""Local filesystem plumbing for the tabular assembler's native-transform path."""

from __future__ import annotations

from michelangelo.workflow.tasks.tabular_assembler._private.file.download import (
    download_file_tree,
)

__all__ = ["download_file_tree"]
