"""Supported operations on `Tensor`."""

import abc
from collections.abc import Sequence
import enum
from typing import cast

from numpy.typing import NDArray

from minigrad.tensor import Tensor
from minigrad.tensor._types import NumericArray


class OpType(enum.StrEnum):
    """Operation types."""

    ADD = enum.auto()
    UNDEFINED = enum.auto()


class Op(abc.ABC):
    """Base class for operations on Tensors."""

    op_type: OpType = OpType.UNDEFINED

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        if cls.op_type == OpType.UNDEFINED:
            raise ValueError(
                "Subclasses of Op must define an op_type class variable."
                f" {cls.__name__} does not."
            )

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
    def backward(self, *args, **kwargs) -> NDArray | tuple[NDArray, ...]:
        """Backward pass of the operation."""


class Add(Op):
    """Addition operation."""

    op_type = OpType.ADD

    def forward(self, x: NumericArray, y: NumericArray) -> NumericArray:
        return cast(NumericArray, x + y)

    def backward(self, grad_output: NumericArray) -> tuple[NumericArray, NumericArray]:
        # TODO: support "unbroadcasting" of upstream gradient depending on input shapes
        return grad_output, grad_output


def _add_dispatcher(self: Tensor, other: Tensor) -> Tensor:
    """Dispatches addition operation."""
    op = Add(self, other)
    return Tensor(
        op.forward(self.data, other.data),
        op=op,
        requires_grad=self.requires_grad or other.requires_grad,
    )


Tensor.__add__ = _add_dispatcher
