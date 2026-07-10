"""Supported operations on `Scalar`."""

from __future__ import annotations

import abc
from collections.abc import Sequence
import enum
import math

from minigrad.scalar.engine import Scalar


class OpType(enum.StrEnum):
    """Operation types."""

    ADD = enum.auto()
    MULTIPLY = enum.auto()
    NEGATE = enum.auto()
    SUBTRACT = enum.auto()
    POWER = enum.auto()
    EXP = enum.auto()
    TANH = enum.auto()
    UNDEFINED = enum.auto()


class Op(abc.ABC):
    """Base class for operations on Values."""

    op_type: OpType = OpType.UNDEFINED

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        if cls.op_type == OpType.UNDEFINED:
            raise ValueError(
                "Subclasses of Op must define an op_type class variable."
                f" {cls.__name__} does not."
            )

    def __init__(self, *inputs) -> None:
        assert all(isinstance(x, Scalar) for x in inputs), (
            "All inputs must be Scalar instances."
        )
        self._parents: Sequence[Scalar] = inputs
        self._saved_values: list[float] = []

    def save_for_backward(self, *values) -> None:
        """Store values to be used for the backward pass."""
        self._saved_values.extend(values)

    @property
    def parents(self) -> Sequence[Scalar]:
        return self._parents

    @property
    def saved_values(self) -> Sequence[float]:
        return self._saved_values

    @abc.abstractmethod
    def forward(self, *args) -> float:
        """Forward pass of the operation."""

    @abc.abstractmethod
    def backward(self, *args) -> float | tuple[float, ...]:
        """Backward pass of the operation."""


class Add(Op):
    """Addition operation."""

    op_type = OpType.ADD

    def forward(self, x: float, y: float) -> float:
        return x + y

    def backward(self, grad_output: float) -> tuple[float, float]:
        return grad_output, grad_output


class Multiply(Op):
    """Multiplication operation."""

    op_type = OpType.MULTIPLY

    def forward(self, x: float, y: float):
        self.save_for_backward(x, y)
        return x * y

    def backward(self, grad_output: float) -> tuple[float, float]:
        x, y = self.saved_values
        return grad_output * y, grad_output * x


class Negate(Op):
    """Negation operation.

    NOTE: While negation cam be implemented using a Multiply (i.e. a * -1),
    It results in an unnecessary dummy Value created for -1 (i.e. Value(-1.0)).
    Also, Multiply is a binary operator that expects to have two operands,
    which would result in the creation of two Value objects in the childen of the
    resulting computed Value.
    """

    op_type = OpType.NEGATE

    def forward(self, x: float) -> float:
        return -x

    def backward(self, grad_output: float) -> float:
        return -grad_output


class Power(Op):
    """Power operation."""

    op_type = OpType.POWER

    def forward(self, x: float, c: float) -> float:
        self.save_for_backward(x, c)
        return x**c

    def backward(self, grad_output: float) -> float:
        x, c = self.saved_values
        return grad_output * c * x ** (c - 1)


class Exp(Op):
    """Exponentiates a Value."""

    op_type = OpType.EXP

    def forward(self, x: float) -> float:
        self.save_for_backward(x)
        return math.exp(x)

    def backward(self, grad_output: float):
        x = self.saved_values[0]
        return grad_output * math.exp(x)


class Tanh(Op):
    """Computes hyperbolic tangent of a Value."""

    op_type = OpType.TANH

    def forward(self, x: float) -> float:
        self.save_for_backward(x)
        return math.tanh(x)

    def backward(self, grad_output: float):
        x = self.saved_values[0]
        return grad_output * (1 - math.tanh(x) ** 2)


def _coerce_to_scalar(x: Scalar | int | float) -> Scalar:
    return x if isinstance(x, Scalar) else Scalar(x)


def _add_dispatcher(self: Scalar, other: Scalar | int | float) -> Scalar:
    """Dispatches addition operation."""
    other = _coerce_to_scalar(other)
    op_instance = Add(self, other)
    return Scalar(
        op_instance.forward(self.data, other.data),
        op=op_instance,
        requires_grad=self.requires_grad or other.requires_grad,
    )


def _radd_dispatcher(self: Scalar, other: Scalar | int | float) -> Scalar:
    other = _coerce_to_scalar(other)
    return other + self


def _mul_dispatcher(self: Scalar, other: Scalar | int | float) -> Scalar:
    """Dispatches multiplication operation."""
    other = _coerce_to_scalar(other)
    op_instance = Multiply(self, other)
    return Scalar(
        op_instance.forward(self.data, other.data),
        op=op_instance,
        requires_grad=self.requires_grad or other.requires_grad,
    )


def _neg_dispatcher(self: Scalar) -> Scalar:
    """Dispatches negation operation."""
    op_instance = Negate(self)
    return Scalar(
        op_instance.forward(self.data),
        op=op_instance,
        requires_grad=self.requires_grad,
    )


def _sub_dispatcher(self: Scalar, other: Scalar | int | float) -> Scalar:
    """Dispatches subtraction operation."""
    return self + (-other)


def _rsub_dispatcher(self: Scalar, other: Scalar | int | float) -> Scalar:
    """Dispatches subtraction operation."""
    # Prevents triggering __radd__ and doing another reversal
    other = _coerce_to_scalar(other)
    return other + (-self)


def _pow_dispatcher(self: Scalar, other: int | float) -> Scalar:
    """Dispatches power operation."""
    op_instance = Power(self)
    return Scalar(
        op_instance.forward(self.data, other),
        op=op_instance,
        requires_grad=self.requires_grad,
    )


def exp(x: Scalar | int | float) -> Scalar:
    """Exponentiates a Scalar."""
    x = _coerce_to_scalar(x)
    op_instance = Exp(x)
    return Scalar(
        op_instance.forward(x.data),
        op=op_instance,
        requires_grad=x.requires_grad,
    )


def tanh(x: Scalar | int | float) -> Scalar:
    """Computes tanh of a Scalar."""
    x = _coerce_to_scalar(x)
    op_instance = Tanh(x)
    return Scalar(
        op_instance.forward(x.data),
        op=op_instance,
        requires_grad=x.requires_grad,
    )


Scalar.__add__ = _add_dispatcher
Scalar.__radd__ = _radd_dispatcher
Scalar.__mul__ = _mul_dispatcher
Scalar.__rmul__ = _mul_dispatcher  # create _rmul_dispatcher to address operand reversal
Scalar.__neg__ = _neg_dispatcher
Scalar.__sub__ = _sub_dispatcher
Scalar.__rsub__ = _rsub_dispatcher
Scalar.__pow__ = _pow_dispatcher
