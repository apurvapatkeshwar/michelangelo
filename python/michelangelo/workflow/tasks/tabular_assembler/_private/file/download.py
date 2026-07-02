"""Local filesystem download helper for the tabular assembler.

Uses fsspec directly rather than the pluggable ``StorageBackend`` abstraction:
this helper merges a predictor and a native-transform artifact into a single
local directory layout (see ``custom_assembler``'s combined-model branch),
which is local filesystem plumbing internal to the assembler rather than a
single-artifact upload/download operation.
"""

from __future__ import annotations

import os

import fsspec


def download_file_tree(url: str, dest: str) -> None:
    """Download a file or directory tree from a URL to a local destination.

    Args:
        url: The fsspec-resolvable URL of the file or directory to download.
        dest: The local destination directory. Created if it does not exist.
    """
    if not os.path.exists(dest):
        os.makedirs(dest)

    fs, path = fsspec.core.url_to_fs(url)
    if fs.isdir(path):
        fs.get(f"{path}/*", dest, recursive=True)
    else:
        fs.get(path, dest)
