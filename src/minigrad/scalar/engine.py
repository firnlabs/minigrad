"""A value in the computation graph for auto-differentiation."""

from __future__ import annotations

from collections.abc import Sequence


class Value:
    """Represents a scalar value with auto-differentiation support."""

    def __init__(
        self,
        data: int | float,
        name: str | None = None,
        _children: Sequence[Value] | None = None,
        _op: None = None,  # TODO: define Op class
    ) -> None:
        """Initialization.

        Args:
            data: Data to store. Converted to Python `float`.
            operator: Used when creating a new Value from an operation.
            children: Any previous `Value`s used to compute this `Value`.
            name: A name for the `Value`.
        """
        self._data = float(data)
        self._op = _op
        self._children = _children or []
        self._name = name or "unnamed"
        self._grad: float = 0.0

    def __repr__(self) -> str:
        """String representation of `Value`."""
        op_type = self._op.op_type if self._op else "none"
        return f"Value(data={self._data}, operator={op_type}, name={self._name})"

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        self._name = value

    @property
    def children(self) -> Sequence[Value]:
        return self._children

    @property
    def op(self):
        return self._op

    @op.setter
    def op(self, op_instance) -> None:
        self._op = op_instance

    @property
    def grad(self) -> float:
        """Returns the gradient of this `Value`."""
        return self._grad

    @grad.setter
    def grad(self, value: float) -> None:
        self._grad = value

    @property
    def data(self) -> float:
        """Returns the stored data."""
        return self._data
