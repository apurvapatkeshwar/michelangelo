"""A model that structurally implements the Model interface without
inheriting from ``michelangelo.lib.model_manager.interface.custom_model.Model``.

Simulates an independently-defined interface (e.g. an internal SDK's own
``Model`` ABC) with an identical method surface, used to verify that packager
validation accepts structurally-conformant models.
"""

from __future__ import annotations

from numpy import ndarray


class DuckTypedModel:
    """A model implementing save/load/predict without subclassing Model."""

    def __init__(self, content: str):
        """Initialize the model.

        Args:
            content: Arbitrary content held by the model.
        """
        self.content = content

    def save(self, path: str) -> None:
        """Save the model to the given path.

        Args:
            path: The local filesystem path where the model should be saved.
        """

    @classmethod
    def load(cls, path: str) -> DuckTypedModel:
        """Load the model from the given path.

        Args:
            path: The local filesystem path containing the saved model.

        Returns:
            A fully initialized DuckTypedModel instance.
        """
        return cls("loaded")

    def predict(self, inputs: dict[str, ndarray]) -> dict[str, ndarray]:
        """Predict on the given data.

        Args:
            inputs: A dictionary mapping feature names to numpy arrays.

        Returns:
            A dictionary mapping output feature names to numpy arrays.
        """
        return inputs
