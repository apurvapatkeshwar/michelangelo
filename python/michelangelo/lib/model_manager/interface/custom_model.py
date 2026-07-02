"""Interface for custom models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

from numpy import ndarray  # noqa: TC002


class Model(ABC):
    """The base class for custom models.

    This abstract base class defines the interface that all custom models must
    implement. Custom models are used with the Model Manager to create Triton
    model packages for serving.

    Subclasses must implement the save, load, and predict methods to handle
    model persistence and inference.
    """

    @abstractmethod
    def save(self, path: str):
        """Save the model to the given path.

        This method should serialize the model to disk. It is strongly
        recommended to avoid using pickle or torch.save directly, as they can
        have compatibility and security issues. Instead, prefer using format-
        specific serialization methods (e.g., SavedModel for TensorFlow,
        state_dict for PyTorch).

        Args:
            path: The local filesystem path where the model should be saved.
                This path should be a directory that will contain all model
                artifacts.
        """

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> Model:
        """Load the model from the given path.

        This method should deserialize the model from disk and return a fully
        initialized Model instance ready for inference.

        Args:
            path: The local filesystem path containing the saved model
                artifacts. This should be the same directory path that was
                used in the save() method.

        Returns:
            A fully initialized Model instance loaded from the specified path.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError("load method is not implemented")

    @abstractmethod
    def predict(
        self,
        inputs: dict[str, ndarray],
    ) -> dict[str, ndarray]:
        """Predict on the given data.

        This method performs inference on the provided input data and returns
        predictions. The input and output dictionaries must match the model
        schema defined when creating the model package.

        Args:
            inputs: A dictionary mapping feature names to numpy arrays. Each
                key should correspond to an input feature name defined in the
                model schema, and each value should be a numpy array with shape
                matching the schema specification.

        Returns:
            A dictionary mapping output feature names to numpy arrays. Each key
            should correspond to an output feature name defined in the model
            schema, and each value should be a numpy array containing the
            predictions.
        """


@runtime_checkable
class ModelProtocol(Protocol):
    """Structural counterpart to :class:`Model`.

    Declares the same public interface as :class:`Model` (``save``, ``load``,
    ``predict``) without requiring nominal inheritance. This lets packager-time
    validation accept any class that implements the expected shape, including
    classes defined against an independently-defined but structurally
    identical interface (e.g. across package boundaries). Prefer subclassing
    :class:`Model` directly for new code; this protocol exists for
    packager-time validation and cross-package interoperability where nominal
    inheritance from this module's ``Model`` isn't available or desired.
    """

    def save(self, path: str) -> None:
        """Save the model to the given path.

        Args:
            path: The local filesystem path where the model should be saved.
                This path should be a directory that will contain all model
                artifacts.
        """
        ...

    @classmethod
    def load(cls, path: str) -> ModelProtocol:
        """Load the model from the given path.

        Args:
            path: The local filesystem path containing the saved model
                artifacts. This should be the same directory path that was
                used in the save() method.

        Returns:
            A fully initialized model instance loaded from the specified path.
        """
        ...

    def predict(self, inputs: dict[str, ndarray]) -> dict[str, ndarray]:
        """Predict on the given data.

        Args:
            inputs: A dictionary mapping feature names to numpy arrays. Each
                key should correspond to an input feature name defined in the
                model schema, and each value should be a numpy array with shape
                matching the schema specification.

        Returns:
            A dictionary mapping output feature names to numpy arrays. Each key
            should correspond to an output feature name defined in the model
            schema, and each value should be a numpy array containing the
            predictions.
        """
        ...
