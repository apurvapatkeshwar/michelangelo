"""Generate metadata for torch raw model packages."""

from michelangelo.lib.model_manager.constants import RawModelType


def generate_type_yaml() -> str:
    """Generate the raw model type metadata.

    Returns:
        YAML content describing the raw package as a torch model.
    """
    return f"type: {RawModelType.TORCH}\n"
