"""Structural stub for the native-transform ``TransformSpec`` contract.

Full native-transform support (``TorchTransformModule``, layer specs, and
``TransformSpec`` (de)serialization) has not yet been migrated to OSS
michelangelo — that lands in a follow-up migration bucket ("PR F"). Until
then, ``model_fuser.fuse._build_tx_hydra_spec`` cannot reconstruct a
native-transform module at serve time, and raises ``NotImplementedError``
rather than importing a package that does not exist.

This module defines the minimal structural contract a concrete
``TransformSpec`` implementation must satisfy so that call sites can type
against it now (via :class:`TransformSpec`) without a hard dependency on the
concrete implementation. It documents the eventual shape of that dependency
for whoever implements PR F, but is not itself invoked.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

__all__ = ["TransformSpec"]


@runtime_checkable
class TransformSpec(Protocol):
    """Structural contract for a native-transform pipeline specification.

    A concrete ``TransformSpec`` (not yet available in OSS) reconstructs a
    transform pipeline from the dict produced by its own ``to_dict()``, and
    exposes the topologically-leveled layer specs that
    ``model_fuser.fuse._build_tx_hydra_spec`` needs to build a Hydra
    reconstruction spec matching the fused state dict's layer ordering.
    """

    transform_specs: dict[str, Any]
    """Mapping of layer name to that layer's spec object."""

    transform_levels: dict[str, int]
    """Mapping of layer name to its topological level (0, 1, 2, ...)."""

    def load_from_dict(self, data: dict[str, Any]) -> None:
        """Populate this spec in place from a ``to_dict()``-serialized payload.

        Args:
            data: The dict previously produced by a ``TransformSpec``'s own
                ``to_dict()``.
        """
        ...

    def get_max_transform_level(self) -> int:
        """Return the highest topological level among this spec's layers.

        Returns:
            The maximum value present in ``transform_levels``.
        """
        ...
