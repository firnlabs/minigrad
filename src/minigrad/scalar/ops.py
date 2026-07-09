"""Supported scalar operations on `Value`."""

from __future__ import annotations

import abc
import enum
import math
from collections.abc import Sequence

from minigrad.scalar.engine import Value


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

    def __init__subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        if cls.op_type == OpType.UNDEFINED:
            raise ValueError(
                "Subclasses of Op must define an op_type class variable."
                f" {cls.__name__} does not."
            )

    def __init__(self, *inputs) -> None:
        assert all(isinstance(x, Value) for x in inputs), (
            "All inputs must be Value instances."
        )
        self._inputs: Sequence[Value] = inputs
        self._out: Value | None = None

    @property
    def out(self) -> Value | None:
        """The Value resulting from this operation, if any."""
        return self._out

    @abc.abstractmethod
    def forward(self, *args):
        """Forward pass of the operation."""

    @abc.abstractmethod
    def backward(self, *args):
        """Backward pass of the operation."""


class Add(Op):
    """Adds two Values."""

    op_type = OpType.ADD

    def forward(self):
        a: Value = self._inputs[0]
        b: Value = self._inputs[1]
        self._out = Value(a.data + b.data, _children=[a, b], _op=self)
        return self._out

    def backward(self, grad_output: float):
        a: Value = self._inputs[0]
        b: Value = self._inputs[1]
        a.grad += grad_output * 1.0
        b.grad += grad_output * 1.0


class Multiply(Op):
    """Multiplies two Values."""

    op_type = OpType.MULTIPLY

    def forward(self):
        a: Value = self._inputs[0]
        b: Value = self._inputs[1]
        self._out = Value(a.data * b.data, _children=[a, b], _op=self)
        return self._out

    def backward(self, grad_output: float):
        a: Value = self._inputs[0]
        b: Value = self._inputs[1]
        a.grad += grad_output * b.data
        b.grad += grad_output * a.data


class Negate(Op):
    """Negates a Value.

    NOTE: While negation cam be implemented using a Multiply (i.e. a * -1),
    It results in an unnecessary dummy Value created for -1 (i.e. Value(-1.0)).
    Also, Multiply is a binary operator that expects to have two operands,
    which would result in the creation of two Value objects in the childen of the
    resulting computed Value.
    """

    op_type = OpType.NEGATE

    def forward(self):
        a: Value = self._inputs[0]
        self._out = Value(-a.data, _children=[a], _op=self)
        return self._out

    def backward(self, grad_output: float):
        a: Value = self._inputs[0]
        a.grad += grad_output * -1.0


class Power(Op):
    """Computes the Value raised to a constant."""

    op_type = OpType.POWER

    def forward(self, exponent: int | float):
        a: Value = self._inputs[0]
        self._out = Value(a.data**exponent, _children=[a], _op=self)
        self._exponent = exponent
        return self._out

    def backward(self, grad_output: float):
        a: Value = self._inputs[0]
        a.grad += grad_output * self._exponent * self._out.data * a.data**-1


class Exp(Op):
    """Exponentiates a Value."""

    op_type = OpType.EXP

    def forward(self):
        a: Value = self._inputs[0]
        self._out = Value(math.exp(a.data), _children=[a], _op=self)
        return self._out

    def backward(self, grad_output: float):
        a: Value = self._inputs[0]
        a.grad += grad_output * self._out.data


class Tanh(Op):
    """Computes hyperbolic tangent of a Value."""

    op_type = OpType.TANH

    def forward(self):
        a: Value = self._inputs[0]
        self._out = Value(math.tanh(a.data), _children=[a], _op=self)
        return self._out

    def backward(self, grad_output: float):
        a: Value = self._inputs[0]
        a.grad += grad_output * (1 - self._out.data**2)


def _coerce_to_value(x) -> Value:
    return x if isinstance(x, Value) else Value(x)


def _add_dispatcher(a, b):
    """Dispatches addition operation."""
    a = _coerce_to_value(a)
    b = _coerce_to_value(b)
    return Add(a, b).forward()


def _mul_dispatcher(a, b):
    """Dispatches multiplication operation."""
    a = _coerce_to_value(a)
    b = _coerce_to_value(b)
    return Multiply(a, b).forward()


def _neg_dispatcher(a):
    """Dispatches negation operation."""
    a = _coerce_to_value(a)
    return Negate(a).forward()


def _sub_dispatcher(a, b):
    """Dispatches subtraction operation."""
    return a + (-b)


def _rsub_dispatcher(b, a):
    """Dispatches subtraction operation."""
    return a + (-b)


def _pow_dispatcher(a, b):
    """Dispatches power operation."""
    return Power(a).forward(b)


def exp(a) -> Value:
    """Exponentiates a Value."""
    a = _coerce_to_value(a)
    return Exp(a).forward()


def tanh(a) -> Value:
    a = _coerce_to_value(a)
    return Tanh(a).forward()


Value.__add__ = _add_dispatcher
Value.__radd__ = _add_dispatcher
Value.__mul__ = _mul_dispatcher
Value.__rmul__ = _mul_dispatcher
Value.__neg__ = _neg_dispatcher
Value.__sub__ = _sub_dispatcher
Value.__rsub__ = _rsub_dispatcher
Value.__pow__ = _pow_dispatcher
