"""Tensor autograd engine."""

from typing import TYPE_CHECKING

import numpy as np

from minigrad.tensor._types import NestedNumSequence
from minigrad.tensor._types import NumArray

if TYPE_CHECKING:
    from minigrad.tensor.ops import Op


class Tensor:
    """Represents a tensor with auto-differentiation support."""

    def __init__(
        self,
        data: NestedNumSequence | NumArray,
        op: "Op | None" = None,
        name: str | None = None,
        requires_grad: bool = False,
    ) -> None:
        """Initialize the Tensor.

        Args:
            data: Data to store. Converted to Python float.
            op: Op used to create a new Tensor.
            name: A name for the Tensor.
            requires_grad: True if a gradient computation is required.
        """
        self._data = np.array(data)
        self._op = op
        self._name = name or "unnamed"
        self._requires_grad = requires_grad
        self._grad: NumArray | None = None

    @property
    def data(self) -> NumArray:
        return self._data

    @property
    def op(self) -> "Op | None":
        return self._op

    @property
    def name(self) -> str:
        return self._name

    @property
    def requires_grad(self) -> bool:
        return self._requires_grad

    @property
    def grad(self) -> NumArray | None:
        return self._grad

    @grad.setter
    def grad(self, value: NestedNumSequence | NumArray | None) -> None:
        self._grad = np.array(value) if value is not None else value

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

        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            if node.requires_grad and node.grad is not None and node.op is not None:
                input_grads = node.op.backward(node.grad)
                if not isinstance(input_grads, tuple):
                    input_grads = (input_grads,)

                for parent, grad in zip(node.op.parents, input_grads, strict=True):
                    if parent.requires_grad:
                        if parent.grad is None:
                            parent.grad = np.zeros_like(parent.data)
                        parent.grad += grad
