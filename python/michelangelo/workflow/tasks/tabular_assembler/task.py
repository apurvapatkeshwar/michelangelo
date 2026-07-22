"""Tabular assembler task — dispatches to the framework-specific assembler."""

from __future__ import annotations

from typing import TYPE_CHECKING

from michelangelo.workflow.tasks.tabular_assembler._private.model_class.resolve import (
    resolve_training_framework,
)
from michelangelo.workflow.tasks.tabular_assembler.custom.assembler import (
    custom_assembler,
)
from michelangelo.workflow.tasks.tabular_assembler.torch.assembler import (
    torch_assembler,
)
from michelangelo.workflow.variables.metadata import (
    TRAINING_FRAMEWORK_CUSTOM,
    TRAINING_FRAMEWORK_LIGHTNING,
    TRAINING_FRAMEWORK_PYTORCH,
)
from michelangelo.workflow.variables.types import AssembledModel, ModelArtifact

if TYPE_CHECKING:
    from michelangelo.lib.artifact_manager.storage_backend import StorageBackend
    from michelangelo.workflow.schema.assembler import TabularAssemblerConfig

__all__ = ["tabular_assembler"]


def tabular_assembler(
    config: TabularAssemblerConfig,
    raw_model: ModelArtifact,
    native_transform_model: ModelArtifact | None = None,
    *,
    storage_backend: StorageBackend,
) -> AssembledModel:
    """Assemble a trained tabular model into deployable and raw packages.

    Dispatches to the custom (Python-backend) or PyTorch/Lightning assembler
    based on ``raw_model.metadata.training_framework``, falling back to
    resolving the framework from ``config.model_class`` when the metadata
    field is unset. A custom ``Model`` subclass referenced by
    ``config.model_class`` always routes to the custom path, even when the
    recorded training framework is ``lightning`` — this lets a config
    explicitly force custom packaging of a model whose training framework was
    recorded generically.

    Args:
        config: The assembler configuration.
        raw_model: The trained model to package.
        native_transform_model: Optional native-transform model preceding
            ``raw_model``. Passed through to the custom or PyTorch/Lightning
            assembler; ignored when the framework can't be resolved at all.
        storage_backend: Backend used to download source artifacts and upload
            produced packages. Required, keyword-only — this task boundary is
            an explicit injection point, not a place to silently default to
            throwaway local storage.

    Returns:
        An ``AssembledModel`` with the deployable and raw packaged artifacts.
        When no training framework is recorded and none can be resolved from
        ``config.model_class``, both ``raw_model`` and ``deployable_model``
        are empty placeholder artifacts (``path=""``) rather than ``None``,
        matching this task's "always return a pair" contract.

    Raises:
        ValueError: If ``raw_model.metadata.training_framework`` is set to a
            non-empty value that doesn't match any known framework. This is
            almost always a caller bug (e.g. an internal training-framework
            identifier that was never translated to this package's
            constants) — failing loudly here is far easier to diagnose than
            letting it surface as a silently-empty package several pipeline
            stages later.
    """
    if (
        raw_model.metadata.training_framework == TRAINING_FRAMEWORK_CUSTOM
        or resolve_training_framework(config.model_class) == TRAINING_FRAMEWORK_CUSTOM
    ):
        return custom_assembler(
            config, raw_model, native_transform_model, storage_backend=storage_backend
        )
    if raw_model.metadata.training_framework == TRAINING_FRAMEWORK_PYTORCH:
        return torch_assembler(config, raw_model, storage_backend=storage_backend)
    if raw_model.metadata.training_framework == TRAINING_FRAMEWORK_LIGHTNING:
        return torch_assembler(
            config, raw_model, native_transform_model, storage_backend=storage_backend
        )
    if raw_model.metadata.training_framework:
        raise ValueError(
            "Unrecognized raw_model.metadata.training_framework: "
            f"{raw_model.metadata.training_framework!r}. Expected one of "
            f"{TRAINING_FRAMEWORK_CUSTOM!r}, {TRAINING_FRAMEWORK_PYTORCH!r}, "
            f"{TRAINING_FRAMEWORK_LIGHTNING!r}, or an unset (None) value."
        )

    return AssembledModel(
        raw_model=ModelArtifact(path=""),
        deployable_model=ModelArtifact(path=""),
    )
