"""Tensor autograd engine."""

from typing import TYPE_CHECKING

import numpy as np

from minigrad.tensor._types import NestedNumSequence
from minigrad.tensor._types import NumArray

if TYPE_CHECKING:
    from minigrad.tensor.ops import Op


class Tensor:
    """Represents a tenso with auto-differentiation support."""

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
