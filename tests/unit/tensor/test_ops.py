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

    @pytest.mark.parametrize(
        "x_data, y_data, exp_forward, upstream_grad, exp_x_grad, exp_y_grad",
        [
            pytest.param(
                [1.0, 2.0],
                [3.0, 4.0],
                [3.0, 8.0],
                [2.0, 2.0],
                [6.0, 8.0],
                [2.0, 4.0],
                id="1d_same_shapes",
            ),
            pytest.param(
                [1.0, 2.0, 3.0],
                [2.0],
                [2.0, 4.0, 6.0],
                [1.0, 2.0, 3.0],
                [2.0, 4.0, 6.0],
                [14.0],
                id="1d_diff_shapes",
            ),
            pytest.param(
                [
                    [1.0, 2.0],
                    [3.0, 4.0],
                ],
                [
                    [2.0, 3.0],
                    [4.0, 5.0],
                ],
                [
                    [2.0, 6.0],
                    [12.0, 20.0],
                ],
                [
                    [2.0, 2.0],
                    [2.0, 2.0],
                ],
                [
                    [4.0, 6.0],
                    [8.0, 10.0],
                ],
                [
                    [2.0, 4.0],
                    [6.0, 8.0],
                ],
                id="2d_same_shapes",
            ),
            pytest.param(
                [
                    [1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0],
                ],
                [1.0, 2.0, 3.0],
                [
                    [1.0, 4.0, 9.0],
                    [4.0, 10.0, 18.0],
                ],
                [
                    [1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0],
                ],
                [
                    [1.0, 2.0, 3.0],
                    [1.0, 2.0, 3.0],
                ],
                [5.0, 7.0, 9.0],
                id="2d_1d_broadcasting",
            ),
            pytest.param(
                [
                    [1.0],
                    [2.0],
                ],
                [
                    [3.0, 4.0],
                ],
                [
                    [3.0, 4.0],
                    [6.0, 8.0],
                ],
                [
                    [1.0, 1.0],
                    [1.0, 1.0],
                ],
                [
                    [7.0],
                    [7.0],
                ],
                [
                    [3.0, 3.0],
                ],
                id="2d_2d_broadcasting_different_input_shapes",
            ),
            pytest.param(
                [
                    [[1.0, 2.0]],
                    [[3.0, 4.0]],
                ],
                [2.0, 3.0],
                [
                    [[2.0, 6.0]],
                    [[6.0, 12.0]],
                ],
                [
                    [[1.0, 1.0]],
                    [[1.0, 1.0]],
                ],
                [
                    [[2.0, 3.0]],
                    [[2.0, 3.0]],
                ],
                [4.0, 6.0],
                id="3d_1d_broadcasting",
            ),
        ],
    )
    def test_forward_backward(
        self, x_data, y_data, exp_forward, upstream_grad, exp_x_grad, exp_y_grad
    ) -> None:
        x = Tensor(x_data)
        y = Tensor(y_data)

        # forward
        z = x * y
        assert np.allclose(z.data, np.array(exp_forward))

        # backward
        assert z.op is not None
        x_grad, y_grad = z.op.backward(np.array(upstream_grad))
        assert np.allclose(x_grad, np.array(exp_x_grad))
        assert np.allclose(y_grad, np.array(exp_y_grad))

    def test_mul_tensor_and_non_tensor(self) -> None:
        x = Tensor([1.0, 2.0])
        y = 3.0
        z = x * y

        assert np.allclose(z.data, np.array([3.0, 6.0]))

    def test_mul_non_tensor_and_tensor(self) -> None:
        y = Tensor([1.0, 2.0])
        x = 3.0
        z = x * y

        assert np.allclose(z.data, np.array([3.0, 6.0]))


class TestMatMul:
    @pytest.mark.parametrize(
        "x_data, y_data, exp_forward, upstream_grad, exp_x_grad, exp_y_grad",
        [
            pytest.param(
                [[1.0, 2.0]],
                [[3.0], [4.0]],
                [[11.0]],
                [[1.0]],
                [[3.0, 4.0]],
                [[1.0], [2.0]],
                id="col_and_row_tensors",
            ),
            pytest.param(
                [
                    [1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0],
                ],
                [
                    [7.0, 8.0],
                    [9.0, 10.0],
                    [11.0, 12.0],
                ],
                [
                    [58.0, 64.0],
                    [139.0, 154.0],
                ],
                [
                    [1.0, 2.0],
                    [3.0, 4.0],
                ],
                [
                    [23.0, 29.0, 35.0],
                    [53.0, 67.0, 81.0],
                ],
                [
                    [13.0, 18.0],
                    [17.0, 24.0],
                    [21.0, 30.0],
                ],
                id="matrix_matrix",
            ),
            pytest.param(
                [
                    [
                        [1.0, 2.0],
                        [3.0, 4.0],
                    ],
                    [
                        [5.0, 6.0],
                        [7.0, 8.0],
                    ],
                ],
                [
                    [1.0],
                    [2.0],
                ],
                [
                    [[5.0], [11.0]],
                    [[17.0], [23.0]],
                ],
                [
                    [[1.0], [1.0]],
                    [[1.0], [1.0]],
                ],
                [
                    [[1.0, 2.0], [1.0, 2.0]],
                    [[1.0, 2.0], [1.0, 2.0]],
                ],
                [
                    [16.0],
                    [20.0],
                ],
                id="batch_matmul_requires_unbroadcast",
            ),
        ],
    )
    def test_forward_backward(
        self, x_data, y_data, exp_forward, upstream_grad, exp_x_grad, exp_y_grad
    ) -> None:
        x = Tensor(x_data)
        y = Tensor(y_data)

        # forward pass
        z = x @ y

        assert np.allclose(z.data, np.array(exp_forward))

        # backward pass
        assert z.op is not None
        x_grad, y_grad = z.op.backward(np.array(upstream_grad))

        assert np.allclose(x_grad, exp_x_grad)
        assert np.allclose(y_grad, exp_y_grad)

    def test_matmul_tensor_and_non_tensor(self) -> None:
        x = Tensor([[1.0, 2.0]])
        y = [[3.0], [4.0]]
        z = x @ y

        assert np.allclose(z.data, np.array([[11.0]]))

    def test_matmul_non_tensor_and_tensor(self) -> None:
        x = [[1.0, 2.0]]
        y = Tensor([[3.0], [4.0]])
        z = x @ y

        assert np.allclose(z.data, np.array([[11.0]]))


class TestNegate:
    """Tests for Negate class."""

    @pytest.mark.parametrize(
        "x_data, exp_forward, upstream_grad, exp_x_grad",
        [
            pytest.param(
                5.0,
                -5.0,
                2.0,
                -2.0,
                id="0d_scalar",
            ),
            pytest.param(
                [2.0, 3.0],
                [-2.0, -3.0],
                [1.0, 1.0],
                [-1.0, -1.0],
                id="1d_tensor",
            ),
            pytest.param(
                [
                    [1.0, -2.0],
                    [-3.0, 4.0],
                ],
                [
                    [-1.0, 2.0],
                    [3.0, -4.0],
                ],
                [
                    [1.0, 2.0],
                    [3.0, 4.0],
                ],
                [
                    [-1.0, -2.0],
                    [-3.0, -4.0],
                ],
                id="2d_tensor",
            ),
            pytest.param(
                [
                    [[1.0], [2.0]],
                    [[3.0], [4.0]],
                ],
                [
                    [[-1.0], [-2.0]],
                    [[-3.0], [-4.0]],
                ],
                [
                    [[2.0], [2.0]],
                    [[2.0], [2.0]],
                ],
                [
                    [[-2.0], [-2.0]],
                    [[-2.0], [-2.0]],
                ],
                id="3d_tensor",
            ),
        ],
    )
    def test_forward_backward(
        self, x_data, exp_forward, upstream_grad, exp_x_grad
    ) -> None:
        x = Tensor(x_data)
        z = -x

        assert np.allclose(z.data, np.array(exp_forward))

        assert z.op is not None
        x_grad = z.op.backward(np.array(upstream_grad))
        assert np.allclose(x_grad, np.array(exp_x_grad))


class TestPower:
    """Tests for Power class."""

    @pytest.mark.parametrize(
        "x_data, exponent, exp_forward, upstream_grad, exp_x_grad",
        [
            pytest.param(
                [1.0, 2.0, 3.0],
                2,
                [1.0, 4.0, 9.0],
                [2.0, 2.0, 2.0],
                [4.0, 8.0, 12.0],
                id="1d_tensor",
            ),
            pytest.param(
                [
                    [1.0, 2.0],
                    [3.0, 4.0],
                ],
                3,
                [
                    [1.0, 8.0],
                    [27.0, 64.0],
                ],
                [
                    [1.0, 2.0],
                    [3.0, 4.0],
                ],
                [
                    [3.0, 24.0],
                    [81.0, 192.0],
                ],
                id="2d_tensor",
            ),
            pytest.param(
                [
                    [[1.0, 2.0], [3.0, 4.0]],
                    [[5.0, 6.0], [7.0, 8.0]],
                ],
                2,
                [
                    [[1.0, 4.0], [9.0, 16.0]],
                    [[25.0, 36.0], [49.0, 64.0]],
                ],
                [
                    [[1.0, 1.0], [1.0, 1.0]],
                    [[1.0, 1.0], [1.0, 1.0]],
                ],
                [
                    [[2.0, 4.0], [6.0, 8.0]],
                    [[10.0, 12.0], [14.0, 16.0]],
                ],
                id="3d_tensor",
            ),
        ],
    )
    def test_forward_backward(
        self, x_data, exponent, exp_forward, upstream_grad, exp_x_grad
    ) -> None:
        x = Tensor(x_data)
        z = x**exponent

        assert np.allclose(z.data, np.array(exp_forward))

        assert z.op is not None
        x_grad = z.op.backward(np.array(upstream_grad))
        assert np.allclose(x_grad, np.array(exp_x_grad))


class TestTanh:
    """Tests for Tanh class."""

    @pytest.mark.parametrize(
        "x_data, upstream_grad",
        [
            pytest.param(
                [1.0, 2.0, 3.0],
                [2.0, 2.0, 2.0],
                id="1d_tensor",
            ),
            pytest.param(
                [
                    [1.0, 2.0],
                    [-1.0, -2.0],
                ],
                [
                    [0.5, 1.5],
                    [2.5, 3.5],
                ],
                id="2d_tensor",
            ),
            pytest.param(
                [
                    [[1.0, -1.0], [2.0, -2.0]],
                    [[3.0, -3.0], [4.0, -4.0]],
                ],
                [
                    [[0.5, 0.5], [1.0, 1.0]],
                    [[1.5, 1.5], [2.0, 2.0]],
                ],
                id="3d_tensor",
            ),
        ],
    )
    def test_forward_backward(self, x_data, upstream_grad) -> None:
        x = Tensor(x_data)

        # forward pass
        z = ops.tanh(x)

        exp_data = np.tanh(x_data)
        assert np.allclose(z.data, exp_data)

        upstream_grad = np.array(upstream_grad)
        assert z.op is not None
        x_grad = z.op.backward(upstream_grad)

        exp_x_grad = upstream_grad * (1 - np.tanh(x_data) ** 2)
        assert np.allclose(x_grad, exp_x_grad)
