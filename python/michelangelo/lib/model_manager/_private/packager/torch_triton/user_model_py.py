"""Generate Triton Python backend user model wrapper for torch models."""

from michelangelo.lib.model_manager._private.packager.template_renderer import (
    TritonTemplateRenderer,
)


def generate_torch_python_user_model_content(
    gen: TritonTemplateRenderer,
    output_names: list[str],
) -> str:
    """Generate ``user_model.py`` content for the torch Python backend.

    Args:
        gen: Triton template renderer.
        output_names: Output field names from the model schema.

    Returns:
        Rendered Python backend wrapper content.
    """
    return gen.render("torch_python/user_model.py.tmpl", {"output_names": output_names})
