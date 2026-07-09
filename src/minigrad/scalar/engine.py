"""A value in the computation graph for auto-differentiation."""

from __future__ import annotations


class Scalar:
    """Represents a scalar value with auto-differentiation support."""

    def __init__(
        self,
        data: int | float,
        op=None,
        name: str | None = None,
    ) -> None:
        """Initialization.

        Args:
            data: Data to store. Converted to Python `float`.
            op: Op when creating a new Value from an operation.
            name: A name for the Ccalar .
        """
        self._data = float(data)
        self._op = op
        self._name = name or "unnamed"
        self._grad: float = 0.0

    def __repr__(self) -> str:
        """String representation of `Value`."""
        op_type = self.op.op_type if self.op else "none"
        return f"Value(data={self.data}, operator={op_type}, name={self.name})"

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        self._name = value

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
