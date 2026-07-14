"""Tests for `tensor/ops.py`."""

import numpy as np
from numpy.typing import NDArray
import pytest

from minigrad.tensor import ops
from minigrad.tensor import Tensor


class SimpleOp(ops.Op):
    op_type = ops.Add

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


class TestAdd:
    """Tests Add class."""

    @pytest.mark.parametrize(
        "x_data, y_data, exp_forward, upstream_grad, exp_x_grad, exp_y_grad",
        [
            pytest.param(
                [1.0, 2.0],
                [3.0, 4.0],
                [4.0, 6.0],
                [2.0, 2.0],
                [2.0, 2.0],
                [2.0, 2.0],
                id="same_shape_1d",
            ),
        ],
    )
    def test_forward_backward(
        self, x_data, y_data, exp_forward, upstream_grad, exp_x_grad, exp_y_grad
    ) -> None:
        x = Tensor(x_data)
        y = Tensor(y_data)

        # forward
        z = x + y
        assert np.allclose(z.data, np.array(exp_forward))

        # backward
        assert z.op is not None
        x_grad, y_grad = z.op.backward(np.array(upstream_grad))
        assert np.allclose(x_grad, np.array(exp_x_grad))
        assert np.allclose(y_grad, np.array(exp_y_grad))
