"""Tabular assembler task — dispatches to the framework-specific assembler."""

from __future__ import annotations

from typing import TYPE_CHECKING

from michelangelo.workflow.tasks.tabular_assembler._private.model_class.resolve import (
    resolve_training_framework,
)
from michelangelo.workflow.tasks.tabular_assembler.custom.assembler import (
    custom_assembler,
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
            assembler; ignored for unrecognized training frameworks.
        storage_backend: Backend used to download source artifacts and upload
            produced packages. Required, keyword-only — this task boundary is
            an explicit injection point, not a place to silently default to
            throwaway local storage.

    Returns:
        An ``AssembledModel`` with the deployable and raw packaged artifacts.
        When the training framework is not recognized, both ``raw_model`` and
        ``deployable_model`` are empty placeholder artifacts (``path=""``)
        rather than ``None``, matching this task's "always return a pair"
        contract.

    Raises:
        NotImplementedError: If dispatch resolves to the PyTorch/Lightning
            path, which is not yet implemented in this package.
    """
    if (
        raw_model.metadata.training_framework == TRAINING_FRAMEWORK_CUSTOM
        or resolve_training_framework(config.model_class) == TRAINING_FRAMEWORK_CUSTOM
    ):
        return custom_assembler(
            config, raw_model, native_transform_model, storage_backend=storage_backend
        )
    if raw_model.metadata.training_framework == TRAINING_FRAMEWORK_PYTORCH:
        return _torch_assembler(config, raw_model, storage_backend=storage_backend)
    if raw_model.metadata.training_framework == TRAINING_FRAMEWORK_LIGHTNING:
        return _torch_assembler(
            config, raw_model, native_transform_model, storage_backend=storage_backend
        )

    return AssembledModel(
        raw_model=ModelArtifact(path=""),
        deployable_model=ModelArtifact(path=""),
    )


def _torch_assembler(
    config: TabularAssemblerConfig,
    raw_model: ModelArtifact,
    native_transform_model: ModelArtifact | None = None,
    *,
    storage_backend: StorageBackend,
) -> AssembledModel:
    """Import and delegate to the real ``torch_assembler``, once it exists.

    The PyTorch/Lightning assembler path (``torch_assembler``) is implemented
    in a follow-up bucket that adds
    ``michelangelo.workflow.tasks.tabular_assembler.torch.assembler``. This
    indirection lets that module land without any further change to this
    dispatch function — once it exists, the import below resolves and this
    branch becomes live automatically.

    Raises:
        NotImplementedError: Always, until the ``torch`` subpackage lands.
    """
    try:
        from michelangelo.workflow.tasks.tabular_assembler.torch.assembler import (
            torch_assembler,
        )
    except ImportError as exc:
        raise NotImplementedError(
            "The PyTorch/Lightning assembler path is not yet implemented — "
            "michelangelo.workflow.tasks.tabular_assembler.torch.assembler "
            "lands in a follow-up change."
        ) from exc
    return torch_assembler(
        config, raw_model, native_transform_model, storage_backend=storage_backend
    )
