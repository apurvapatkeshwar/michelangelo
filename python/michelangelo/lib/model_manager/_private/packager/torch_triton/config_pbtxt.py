"""Generate Triton config.pbtxt content for torch packages."""

from typing import Any, Optional

from michelangelo.lib.model_manager._private.constants import TritonBackendType
from michelangelo.lib.model_manager._private.packager.template_renderer import (
    TritonTemplateRenderer,
)
from michelangelo.lib.model_manager._private.schema.triton import convert_model_schema
from michelangelo.lib.model_manager.schema import ModelSchema


def generate_config_pbtxt_content(
    gen: TritonTemplateRenderer,
    model_name: str,
    model_revision: Optional[str],
    model_schema: ModelSchema,
    backend: str = TritonBackendType.TORCH,
    enable_dynamic_batching: bool = True,
    triton_parameters: Optional[dict[str, Any]] = None,
) -> str:
    """Generate Triton model configuration for a torch deployable package.

    Args:
        gen: Triton template renderer.
        model_name: Model name to place in the Triton config.
        model_revision: Optional revision suffix for the model name.
        model_schema: Michelangelo model schema.
        backend: Triton backend name.
        enable_dynamic_batching: Whether to enable Triton dynamic batching.
        triton_parameters: Optional template overrides such as
            ``preferred_batch_size`` or ``instance_count``.

    Returns:
        Rendered ``config.pbtxt`` content.
    """
    input_schema, output_schema = convert_model_schema(model_schema)

    if model_name and model_revision:
        model_name = f"{model_name}-{model_revision}"

    template_vars: dict[str, Any] = {
        "model_name": model_name,
        "backend": backend,
        "max_batch_size": 256 if enable_dynamic_batching else 0,
        "enable_dynamic_batching": enable_dynamic_batching,
        "instance_count": 1,
        "inputs": input_schema,
        "outputs": output_schema,
    }
    if enable_dynamic_batching:
        template_vars["max_queue_delay_microseconds"] = 300
    if triton_parameters:
        template_vars.update(triton_parameters)

    return gen.render("config.pbtxt.tmpl", template_vars).rstrip() + "\n"
