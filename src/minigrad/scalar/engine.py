"""A value in the computation graph for auto-differentiation."""

from __future__ import annotations

from typing import Any


class Scalar:
    """Represents a scalar value with auto-differentiation support."""

    def __init__(
        self,
        data: int | float,
        op: Any = None,
        name: str | None = None,
    ) -> None:
        """Initialize the Scalar.

        Args:
            data: Data to store. Converted to Python `float`.
            op: Op when creating a new Value from an operation.
            name: A name for the Scalar.
        """
        self._data = float(data)
        self._op = op
        self._name = name or "unnamed"
        self._grad: float = 0.0

    def __repr__(self) -> str:
        """Return the string representation of `Value`."""
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

    def backward(self) -> None:
        """Runs backpropagation and computes gradients."""
        topo = []
        visited = set()

        def build_topo(v: Scalar) -> None:
            if v not in visited:
                visited.add(v)
                if v.op is not None:
                    for parent in v.op.parents:
                        build_topo(parent)
                topo.append(v)

        build_topo(self)

        self._grad = 1.0
        for node in reversed(topo):
            if node.op is not None:
                input_grads = node.op.backward(node.grad)
                if not isinstance(input_grads, tuple):
                    input_grads = (input_grads,)

                for parent, grad in zip(node.op.parents, input_grads, strict=True):
                    parent.grad += grad
