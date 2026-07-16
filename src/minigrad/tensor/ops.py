"""Supported operations on `Tensor`."""

import abc
from collections.abc import Sequence
import enum

from minigrad.tensor import Tensor
from minigrad.tensor._types import NestedNumSequence
from minigrad.tensor._types import NumArray


class OpType(enum.StrEnum):
    """Operation types."""

    ADD = enum.auto()
    MULTIPLY = enum.auto()
    UNDEFINED = enum.auto()
    MATMUL = enum.auto()


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
        self._saved_values: list[NumArray] = []

    @property
    def parents(self) -> Sequence[Tensor]:
        return self._parents

    @property
    def saved_values(self) -> list[NumArray]:
        return self._saved_values

    def save_for_backward(self, *values) -> None:
        """Store values to be used for the backward pass."""
        self._saved_values.extend(values)

    @abc.abstractmethod
    def forward(self, *args, **kwargs) -> NumArray:
        """Forward pass of the operation."""

    @abc.abstractmethod
    def backward(self, *args, **kwargs) -> NumArray | tuple[NumArray, ...]:
        """Backward pass of the operation."""


def unbroadcast(grad: NumArray, target_shape: tuple[int, ...]) -> NumArray:
    """Sum out broadcasted dimensions of `grad` to match `target_shape`.

    Args:
        grad: Gradient array to unbroadcast.
        target_shape: Target shape of the unbroadcasted gradient.

    Returns:
        grad array with certain dimensions summed to match `target_shape` if needed.
    """
    # Shape is the same, no change needed
    if grad.shape == target_shape:
        return grad

    # Determine axes to sum
    ndim_diff = grad.ndim - len(target_shape)
    axes = []
    for i in range(grad.ndim):
        # Treat prepended dimensions as size 1
        # e.g. given grad.shape = (2, 5, 3) and target_shape = (5, 1),
        # target_shape is treated as (1, 5, 1)
        target_dim = 1 if i < ndim_diff else target_shape[i - ndim_diff]
        if grad.shape[i] != target_dim:
            axes.append(i)

    if axes:
        grad = grad.sum(axis=tuple(axes), keepdims=True)

    # Remove any extra dimensions in final grad to match target_shape
    return grad.reshape(target_shape)


class Add(Op):
    """Addition operation."""

    op_type = OpType.ADD

    def forward(self, x: NumArray, y: NumArray) -> NumArray:
        self.save_for_backward(x, y)
        return x + y

    def backward(self, grad_output: NumArray) -> tuple[NumArray, NumArray]:
        x, y = self.saved_values
        return (
            unbroadcast(grad_output, x.shape),
            unbroadcast(grad_output, y.shape),
        )


class Multiply(Op):
    """Tensor Hadamard product operation."""

    op_type = OpType.MULTIPLY

    def forward(self, x: NumArray, y: NumArray) -> NumArray:
        self.save_for_backward(x, y)
        return x * y

    def backward(self, grad_output: NumArray) -> tuple[NumArray, NumArray]:
        x, y = self.saved_values
        return (
            unbroadcast(grad_output * y, x.shape),
            unbroadcast(grad_output * x, y.shape),
        )


class MatMul(Op):
    """Tensor matrix multiplication operation."""

    op_type = OpType.MATMUL

    def forward(self, x: NumArray, y: NumArray) -> NumArray:
        self.save_for_backward(x, y)
        return x @ y

    def backward(self, grad_output: NumArray) -> tuple[NumArray, NumArray]:
        x, y = self.saved_values
        # Use .swapaxes instead of .T for >2 dimensions as the latter reverses all
        # axes.
        # e.g. transpose of tensor with shape (batch, M, N) should result in
        # (batch, N, M)
        return (
            unbroadcast(grad_output @ y.swapaxes(-1, -2), x.shape),
            unbroadcast(x.swapaxes(-1, -2) @ grad_output, y.shape),
        )


# TODO: implement Negate
# TODO: implement Power
# TODO: implement Tanh


def _coerce_to_tensor(value: NestedNumSequence | NumArray | Tensor) -> Tensor:
    if isinstance(value, Tensor):
        return value

    return Tensor(value)


def _add_dispatcher(
    self: Tensor, other: NestedNumSequence | NumArray | Tensor
) -> Tensor:
    """Dispatches addition operation."""
    other = _coerce_to_tensor(other)
    op = Add(self, other)
    return Tensor(
        op.forward(self.data, other.data),
        op=op,
        requires_grad=self.requires_grad or other.requires_grad,
    )


def _radd_dispatcher(
    self: Tensor, other: NestedNumSequence | NumArray | Tensor
) -> Tensor:
    other = _coerce_to_tensor(other)
    return other + self


def _mul_dispatcher(
    self: Tensor, other: NestedNumSequence | NumArray | Tensor
) -> Tensor:
    """Dispatches addition operation."""
    other = _coerce_to_tensor(other)
    op = Multiply(self, other)
    return Tensor(
        op.forward(self.data, other.data),
        op=op,
        requires_grad=self.requires_grad or other.requires_grad,
    )


def _rmul_dispatcher(
    self: Tensor, other: NestedNumSequence | NumArray | Tensor
) -> Tensor:
    other = _coerce_to_tensor(other)
    return other * self


def _matmul_dispatcher(
    self: Tensor, other: NestedNumSequence | NumArray | Tensor
) -> Tensor:
    """Dispatches addition operation."""
    other = _coerce_to_tensor(other)
    op = MatMul(self, other)
    return Tensor(
        op.forward(self.data, other.data),
        op=op,
        requires_grad=self.requires_grad or other.requires_grad,
    )


def _rmatmul_dispatcher(
    self: Tensor, other: NestedNumSequence | NumArray | Tensor
) -> Tensor:
    other = _coerce_to_tensor(other)
    return other @ self


Tensor.__add__ = _add_dispatcher
Tensor.__radd__ = _radd_dispatcher
Tensor.__mul__ = _mul_dispatcher
Tensor.__rmul__ = _rmul_dispatcher
# TODO: implement __neg__
Tensor.__matmul__ = _matmul_dispatcher
Tensor.__rmatmul__ = _rmatmul_dispatcher
# TODO: implement __sub__
# TODO: implement __rsub__
# TODO: implement __pow__
