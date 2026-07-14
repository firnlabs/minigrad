"""Tests for `tensor/ops.py`."""

from numpy.typing import NDArray
import pytest

from minigrad.tensor import ops
from minigrad.tensor import Tensor


class SimpleOp(ops.Op):
    def forward(self, array: NDArray) -> NDArray:
        return array

    def backward(self, array: NDArray) -> NDArray:
        return array


@pytest.fixture
def inputs() -> tuple[Tensor, Tensor]:
    return (
        Tensor([1, 2, 3]),
        Tensor([4, 5, 6]),
    )


@pytest.fixture
def simple_op(inputs) -> SimpleOp:
    a, b = inputs
    return SimpleOp(a, b)


class TestOp:
    """Tests for tensor.Op base class."""

    def test_init_no_inputs(self) -> None:
        with pytest.raises(AssertionError, match="All inputs must be Tensor instances"):
            SimpleOp([])

    def test_init(self, simple_op, inputs) -> None:
        assert simple_op.parents == inputs
        assert simple_op.saved_values == []

    def test_save_for_backward(self, simple_op) -> None:
        a = Tensor([1, 2, 3])
        c = 0.01

        simple_op.save_for_backward(a, c)

        assert simple_op.saved_values == [a, c]
