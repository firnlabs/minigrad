"""Autograd engine."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Avoid circular import of ops by placing it under TYPE_CHECKING
    from minigrad.scalar.ops import Op


class Scalar:
    """Represents a scalar value with auto-differentiation support."""

    def __init__(
        self,
        data: int | float,
        op: Op | None = None,
        name: str | None = None,
        requires_grad: bool = False,
    ) -> None:
        """Initialize the Scalar.

        Args:
            data: Data to store. Converted to Python float.
            op: Op used to create a new Scalar.
            name: A name for the Scalar.
            requires_grad: True if a gradient computation is required.
        """
        self._data = float(data)
        self._op = op
        self._name = name or "unnamed"
        self._grad: float | None = None
        self._requires_grad = requires_grad

    def __repr__(self) -> str:
        """Returns string representation."""
        op_type = self.op.op_type if self.op else "none"
        return (
            f"Scalar(data={self.data}, op={op_type}, name={self.name},"
            f" requires_grad={self.requires_grad})"
        )

    @property
    def data(self) -> float:
        """Returns the stored data."""
        return self._data

    @property
    def op(self) -> Op | None:
        return self._op

    @op.setter
    def op(self, op_instance: Op | None) -> None:
        self._op = op_instance

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        self._name = value

    @property
    def grad(self) -> float | None:
        """Returns computed gradient if it exists, None otherwise."""
        return self._grad

    @grad.setter
    def grad(self, value: float) -> None:
        self._grad = value

    @property
    def requires_grad(self) -> bool:
        return self._requires_grad

    @requires_grad.setter
    def requires_grad(self, value: bool) -> None:
        self._requires_grad = value

    def backward(self) -> None:
        """Runs backpropagation and computes gradients."""
        if not self.requires_grad:  # early exit if this doesn't require grad
            return

        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                if v.op is not None:
                    for parent in v.op.parents:
                        build_topo(parent)
                topo.append(v)

        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            if node.requires_grad and node.grad is not None and node.op is not None:
                input_grads = node.op.backward(node.grad)
                if not isinstance(input_grads, tuple):
                    input_grads = (input_grads,)

                for parent, grad in zip(node.op.parents, input_grads, strict=True):
                    if parent.requires_grad:
                        if parent.grad is None:
                            parent.grad = 0.0
                        parent.grad += grad
