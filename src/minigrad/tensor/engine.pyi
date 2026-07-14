from collections.abc import Sequence

from numpy.typing import NDArray

from minigrad.tensor.ops import Op

class Tensor:
    def __init__(
        self,
        data: Sequence[int | float] | NDArray,
        op: Op | None = None,
        name: str | None = None,
        requires_grad: bool = False,
    ) -> None: ...
    @property
    def data(self) -> NDArray: ...
    @property
    def op(self) -> Op | None: ...
    @property
    def name(self) -> str: ...
    @property
    def requires_grad(self) -> bool: ...
    def __add__(self, other: Tensor) -> Tensor: ...
