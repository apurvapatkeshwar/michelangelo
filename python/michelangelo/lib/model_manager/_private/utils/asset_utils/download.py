"""Asset download module."""

from __future__ import annotations

import os
import shutil
from typing import TYPE_CHECKING

from michelangelo.lib.model_manager.constants import StorageType

if TYPE_CHECKING:
    from michelangelo.lib.artifact_manager import StorageBackend


def download_assets(
    src: str,
    des: str,
    source_type: str,
    storage_backend: StorageBackend | None = None,
):
    """Download the assets from source to destination.

    Args:
        src: The path of the source
        des: The destination path to store the assets.
        source_type: The source type of the source path,
            currently only ``StorageType.LOCAL`` is supported without an
            injected storage backend.
        storage_backend: Optional storage backend used to download non-local
            artifact URIs. This mirrors the pusher's storage abstraction.
    """
    if storage_backend is not None:
        storage_backend.download(src, des)
        return

    if source_type == StorageType.LOCAL and src != des:
        if os.path.isdir(src):
            shutil.copytree(src, des, dirs_exist_ok=True)
        else:
            shutil.copy(src, des)
        return

    if source_type != StorageType.LOCAL:
        raise ValueError(
            f"Unsupported source_type {source_type!r}. Pass a StorageBackend "
            "to download remote artifacts."
        )
