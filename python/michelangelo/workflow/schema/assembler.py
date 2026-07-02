"""Typed configuration dataclasses for the tabular assembler task.

Mirror the internal ``*(JSONData)`` schemas as plain dataclasses so they can
be validated at pipeline-definition time, serialised, and inspected by the
workflow engine without the internal pydantic/uniflow base.

Internal consumers may subclass these to add provider-specific fields (e.g.
an HDFS storage toggle) without modifying the OSS package.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "CustomAssemblerConfig",
    "TabularAssemblerConfig",
    "TorchAssemblerConfig",
]


@dataclass
class CustomAssemblerConfig:
    """Configuration for the custom (Python-backend) assembler path.

    Attributes:
        custom_batch_processing: When ``True``, the model implementation
            handles batching itself and receives inputs with an extra leading
            batch dimension on top of the model schema (schema shape
            ``[n, ..., m]`` becomes ``[batch, n, ..., m]``). When
            ``None``/``False``, Triton batches automatically and the model
            sees schema-shaped inputs.
        additional_import_prefixes: Extra Python module prefixes whose source
            files are bundled into the package. Useful when the model class
            uses dynamic imports (e.g. ``importlib``) that static analysis
            misses. Each prefix is resolved recursively.

    Example:
        >>> CustomAssemblerConfig(custom_batch_processing=True)
        CustomAssemblerConfig(custom_batch_processing=True, ...)
    """

    custom_batch_processing: bool | None = None
    additional_import_prefixes: list[str] | None = None


@dataclass
class TorchAssemblerConfig:
    """Configuration for the PyTorch/Lightning assembler path.

    Attributes:
        backend: Triton backend used for the deployable package — one of
            ``"pytorch"``, ``"tensorrt"``, ``"python"``, ``"onnxruntime"``.
            ``None`` selects the packager default (TorchScript/PyTorch).
            Validation of supported values is performed by the packager, not
            here, to keep a single source of truth.

    Example:
        >>> TorchAssemblerConfig(backend="onnxruntime")
        TorchAssemblerConfig(backend='onnxruntime')
    """

    backend: str | None = None


@dataclass
class TabularAssemblerConfig:
    """Top-level configuration for the tabular assembler task.

    Selects and parameterises the framework-specific assembler path.
    ``custom`` and ``torch`` carry path-specific options; ``model_class``
    overrides the model class resolved from the trained model's metadata.

    Attributes:
        model_class: Fully-qualified model class (e.g.
            ``"mypkg.models.MyModel"``). When set, overrides the class
            recorded in the trained model's metadata and can force the
            custom path.
        custom: Options for the custom (Python-backend) path.
        torch: Options for the PyTorch/Lightning path.

    Example:
        >>> TabularAssemblerConfig(torch=TorchAssemblerConfig(backend="pytorch"))
        TabularAssemblerConfig(model_class=None, custom=None, ...)
    """

    model_class: str | None = None
    custom: CustomAssemblerConfig | None = None
    torch: TorchAssemblerConfig | None = None
