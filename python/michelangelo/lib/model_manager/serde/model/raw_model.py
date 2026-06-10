"""Raw model loader."""

from typing import Optional, Union

import torch

from michelangelo.lib.model_manager._private.serde.model import (
    get_raw_model_type,
    load_custom_raw_model,
    load_torch_raw_model,
)
from michelangelo.lib.model_manager.constants import RawModelType
from michelangelo.lib.model_manager.interface.custom_model import Model


def load_raw_model(
    model_path: str,
    submodule: Optional[str] = None,
) -> Union[Model, torch.nn.Module]:
    """Load the raw model from the model package.

    Args:
        model_path: The model package path
        submodule: Optional dotted submodule path for torch models. This is
            reserved for compatibility and is not supported for custom models.

    Returns:
        The raw model
        For custom python model, it returns the custom Model instance
        For torch model, it returns the PyTorch nn.Module instance
    """
    raw_model_type = get_raw_model_type(model_path)

    if raw_model_type == RawModelType.CUSTOM_PYTHON:
        if submodule is not None:
            raise ValueError("submodule is not supported for custom python models.")
        return load_custom_raw_model(model_path)
    if raw_model_type == RawModelType.TORCH:
        return load_torch_raw_model(model_path, submodule=submodule)

    raise NotImplementedError(
        f"The loader for {raw_model_type} model is not supported yet. "
        "Please check back in future updates."
    )
