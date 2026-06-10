"""Torch model fixtures for packager tests."""

import torch


class SimpleTorchModel(torch.nn.Module):
    """Small deterministic torch module used by packager tests."""

    def __init__(self):
        """Initialize a linear model with deterministic parameters."""
        super().__init__()
        self.linear = torch.nn.Linear(2, 1)
        with torch.no_grad():
            self.linear.weight.fill_(1.0)
            self.linear.bias.fill_(0.0)

    def forward(self, x):
        """Run a forward pass.

        Args:
            x: Input tensor of shape ``[batch_size, 2]``.

        Returns:
            Output tensor of shape ``[batch_size, 1]``.
        """
        return self.linear(x.float())
