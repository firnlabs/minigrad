"""Tests for `tensor/ops.py`."""

import numpy as np
import pytest

from minigrad.tensor import ops
from minigrad.tensor import Tensor
from minigrad.tensor._types import NumArray


class SimpleOp(ops.Op):
    op_type = ops.Add

    def forward(self, array: NumArray) -> NumArray:
        return array

    def backward(self, array: NumArray) -> NumArray:
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
                id="1d_same_shapes",
            ),
            pytest.param(
                [1.0, 2.0, 3.0],
                [4.0],
                [5.0, 6.0, 7.0],
                [2.0, 2.0, 2.0],
                [2.0, 2.0, 2.0],
                [6.0],
                id="1d_diff_shapes",
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

    def test_add_tensor_and_non_tensor(self) -> None:
        x = Tensor([1.0, 2.0])
        y = 3.0
        z = x + y

        assert np.allclose(z.data, np.array([4.0, 5.0]))

    def test_add_non_tensor_and_tensor(self) -> None:
        y = Tensor([1.0, 2.0])
        x = 3.0
        z = x + y

        assert np.allclose(z.data, np.array([4.0, 5.0]))


class TestMultiply:
    """Tests for Multiply class."""

    def test_forward_backward_no_broadcast(self) -> None:
        x = Tensor(
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ]
        )
        y = Tensor(
            [
                [2.0, 3.0],
                [4.0, 5.0],
            ]
        )

        # forward pass
        z = x * y

        exp_data = np.array(
            [
                [2.0, 6.0],
                [12.0, 20.0],
            ]
        )
        assert np.allclose(z.data, exp_data)

        # backward pass
        grad_output = np.array(
            [
                [2.0, 2.0],
                [2.0, 2.0],
            ]
        )
        assert z.op is not None
        x_grad, y_grad = z.op.backward(grad_output)

        exp_x_grad = np.array(
            [
                [4.0, 6.0],
                [8.0, 10.0],
            ]
        )
        exp_y_grad = np.array(
            [
                [2.0, 4.0],
                [6.0, 8.0],
            ]
        )
        assert np.allclose(x_grad, exp_x_grad)
        assert np.allclose(y_grad, exp_y_grad)


@pytest.mark.parametrize(
    "grad, target_shape, expected",
    [
        # input grad: (2, 5, 3) of 1s
        # target_shape: (5, 1) (requires summing over axis 0 and 2: 2 * 3 = 6)
        # expected: (5, 1) of 6s
        [np.ones((2, 5, 3)), (5, 1), np.ones((5, 1)) * 6],
        # input grad: (1, 3) of 1s
        # target_shape: (3,) (requires dropping prepended axis 0)
        # expected: (3,) of 1s
        [np.ones((1, 3)), (3,), np.ones((3,))],
    ],
)
def test_unbroadcast(grad, target_shape, expected) -> None:
    actual = ops.unbroadcast(grad, target_shape)

    assert actual.shape == target_shape
    assert np.allclose(actual, expected)
