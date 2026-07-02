"""Tabular assembler task — packages a trained tabular model for serving.

Exposes :func:`tabular_assembler`, the framework-dispatching entry point that
routes a raw trained model to the custom (Python-backend) or PyTorch/Lightning
assembler path based on ``ModelMetadata.training_framework`` and the
assembler configuration.
"""

from __future__ import annotations

from michelangelo.workflow.tasks.tabular_assembler.task import tabular_assembler

__all__ = ["tabular_assembler"]
