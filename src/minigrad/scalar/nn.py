"""Neural network module and layers."""

import abc
from collections.abc import Sequence

from minigrad.scalar.engine import Scalar


class Module(abc.ABC):
    """Base class for all neural network modules."""

    def __init__(self, *args, **kwargs) -> None:
        self._parameters: Sequence[Scalar] = []

    @property
    def parameters(self) -> Sequence[Scalar]:
        return self._parameters

    def zero_grad(self) -> None:
        """Zeros gradients of all parameters."""
        for p in self.parameters:
            p.grad = None

    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        """Defines the forward pass."""

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
