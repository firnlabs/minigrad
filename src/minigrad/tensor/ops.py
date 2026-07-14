"""Supported operations on `Tensor`."""

import abc
from collections.abc import Sequence

from numpy.typing import NDArray

from minigrad.tensor import Tensor


class Op(abc.ABC):
    """Base class for operations on Tensors."""

    def __init__(self, *inputs) -> None:
        assert all(isinstance(x, Tensor) for x in inputs), (
            "All inputs must be Tensor instances."
        )
        self._parents: Sequence[Tensor] = inputs
        self._saved_values: list[int | float | NDArray] = []

    @property
    def parents(self) -> Sequence[Tensor]:
        return self._parents

    @property
    def saved_values(self) -> list[int | float | NDArray]:
        return self._saved_values

    def save_for_backward(self, *values) -> None:
        """Store values to be used for the backward pass."""
        self._saved_values.extend(values)

    @abc.abstractmethod
    def forward(self, *args, **kwargs) -> NDArray:
        """Forward pass of the operation."""

    @abc.abstractmethod
    def backward(self, *args, **kwargs) -> NDArray:
        """Backward pass of the operation."""
