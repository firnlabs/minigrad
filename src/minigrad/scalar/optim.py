"""Neural net optimizers."""

import abc
from collections.abc import Sequence

from minigrad.scalar import Scalar


class Optimizer(abc.ABC):
    """Base class for all optimizers."""

    def __init__(self, parameters: Sequence[Scalar]) -> None:
        self._parameters = parameters

    @property
    def parameters(self) -> Sequence[Scalar]:
        return self._parameters

    def zero_grad(self) -> None:
        """Resets the gradients of all optimized parameters to None."""
        for p in self.parameters:
            p.grad = None

    @abc.abstractmethod
    def step(self) -> None:
        """Performs a single optimization step (parameter update)."""


class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer."""

    def __init__(self, parameters: list[Scalar], lr: float = 0.1) -> None:
        super().__init__(parameters)
        self.lr = lr

    def step(self) -> None:
        for p in self.parameters:
            if p.grad is not None:
                p.data -= self.lr * p.grad
