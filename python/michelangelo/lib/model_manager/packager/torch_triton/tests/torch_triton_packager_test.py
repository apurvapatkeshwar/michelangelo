"""Tests for TorchTritonPackager."""

import os
import tempfile
from unittest import TestCase

import numpy as np
import torch

from michelangelo.lib.model_manager._private.constants import TritonBackendType
from michelangelo.lib.model_manager.constants import RawModelType
from michelangelo.lib.model_manager.packager.torch_triton import TorchTritonPackager
from michelangelo.lib.model_manager.packager.torch_triton.tests.fixtures.simple_model import (  # noqa: E501
    SimpleTorchModel,
)
from michelangelo.lib.model_manager.schema import DataType, ModelSchema, ModelSchemaItem
from michelangelo.lib.model_manager.serde.model import load_raw_model

MODEL_CLASS = (
    "michelangelo.lib.model_manager.packager.torch_triton.tests.fixtures."
    "simple_model.SimpleTorchModel"
)


class TorchTritonPackagerTest(TestCase):
    """Tests torch Triton packager workflows."""

    def setUp(self):
        """Set up model schema and sample data."""
        self.model_schema = ModelSchema(
            input_schema=[
                ModelSchemaItem(name="x", data_type=DataType.FLOAT, shape=[2]),
            ],
            output_schema=[
                ModelSchemaItem(name="y", data_type=DataType.FLOAT, shape=[1]),
            ],
        )
        self.sample_data = [{"x": np.array([1.0, 2.0], dtype=np.float32)}]

    def _write_state_dict(self, path: str) -> None:
        """Write a deterministic state dict to ``path``."""
        torch.save(SimpleTorchModel().state_dict(), path)

    def _write_torchscript(self, path: str) -> None:
        """Write a deterministic torchscript model to ``path``."""
        torch.jit.save(torch.jit.script(SimpleTorchModel()), path)

    def test_create_raw_model_package(self):
        """It creates a raw torch model package and loads it back."""
        packager = TorchTritonPackager()
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "model.pt")
            dest_model_path = os.path.join(temp_dir, "raw_model")
            self._write_state_dict(model_path)

            result_path = packager.create_raw_model_package(
                model_path=model_path,
                model_class=MODEL_CLASS,
                model_schema=self.model_schema,
                sample_data=self.sample_data,
                dest_model_path=dest_model_path,
                include_import_prefixes=[
                    "michelangelo.lib.model_manager.packager.torch_triton.tests"
                ],
            )

            self.assertEqual(result_path, dest_model_path)
            self.assertTrue(os.path.exists(os.path.join(result_path, "model")))
            self.assertTrue(os.path.exists(os.path.join(result_path, "defs")))
            with open(os.path.join(result_path, "metadata", "type.yaml")) as f:
                self.assertEqual(f.read(), f"type: {RawModelType.TORCH}\n")

            loaded_model = load_raw_model(result_path)
            self.assertIsInstance(loaded_model, SimpleTorchModel)
            with torch.no_grad():
                output = loaded_model(torch.tensor([[1.0, 2.0]]))
            self.assertEqual(output.detach().numpy().tolist(), [[3.0]])

    def test_create_torchscript_model_package(self):
        """It creates a PyTorch backend deployable package."""
        packager = TorchTritonPackager()
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "scripted.pt")
            dest_model_path = os.path.join(temp_dir, "deployable_model")
            self._write_torchscript(model_path)

            result_path = packager.create_model_package(
                model_path=model_path,
                model_schema=self.model_schema,
                model_name="torch-model",
                dest_model_path=dest_model_path,
                backend=TritonBackendType.TORCH,
            )

            self.assertEqual(result_path, dest_model_path)
            self.assertTrue(os.path.exists(os.path.join(result_path, "0", "model.pt")))
            with open(os.path.join(result_path, "config.pbtxt")) as f:
                config = f.read()
            self.assertIn('name: "torch-model-0"', config)
            self.assertIn('backend: "pytorch"', config)

    def test_create_python_backend_model_package(self):
        """It creates a Python backend deployable torch package."""
        packager = TorchTritonPackager()
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "model.pt")
            dest_model_path = os.path.join(temp_dir, "python_model")
            self._write_state_dict(model_path)

            result_path = packager.create_model_package(
                model_path=model_path,
                model_schema=self.model_schema,
                model_name="torch-python-model",
                dest_model_path=dest_model_path,
                backend=TritonBackendType.PYTHON,
                model_class=MODEL_CLASS,
                include_import_prefixes=[
                    "michelangelo.lib.model_manager.packager.torch_triton.tests"
                ],
            )

            self.assertTrue(os.path.exists(os.path.join(result_path, "0", "model.py")))
            self.assertTrue(
                os.path.exists(os.path.join(result_path, "0", "user_model.py"))
            )
            self.assertTrue(
                os.path.exists(os.path.join(result_path, "0", "model", "model.pt"))
            )
            with open(os.path.join(result_path, "config.pbtxt")) as f:
                config = f.read()
            self.assertIn('backend: "python"', config)

    def test_create_model_package_rejects_tensorrt_backend(self):
        """It rejects TensorRT because OSS does not ship TensorRT conversion."""
        packager = TorchTritonPackager()
        with self.assertRaises(ValueError):
            packager.create_model_package(
                model_path="/tmp/model.pt",
                model_schema=self.model_schema,
                backend=TritonBackendType.TENSORRT,
            )
