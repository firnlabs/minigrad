"""Tests backpropagation of gradients in Tensor."""

import numpy as np
import pytest

from minigrad.tensor import Tensor
from minigrad.tensor.ops import tanh


def test_backprop_one_op() -> None:
    x = Tensor([1.0, 2.0], name="a", requires_grad=True)
    y = Tensor([3.0, 4.0], name="b", requires_grad=False)
    z = x + y

    z.backward()

    assert z.grad is not None
    assert np.allclose(z.grad, np.ones((2,)))
    assert x.grad is not None
    assert np.allclose(x.grad, z.grad)
    assert y.grad is None


@pytest.mark.parametrize(
    "x_data, w_data, b_data",
    [
        pytest.param(
            [
                [1.5, -0.5, 2.0],
            ],
            [
                [0.8],
                [0.5],
                [-0.4],
            ],
            [-0.2],
            id="single_sample_1d_weights",
        ),
        pytest.param(
            [
                [1.0, -1.0, 2.0],
                [0.0, 0.5, -0.5],
            ],
            [
                [0.5, -0.5],
                [-0.4, 0.2],
                [-0.3, 0.8],
            ],
            [0.1, -0.2],
            id="batch_samples_2d_weights",
        ),
    ],
)
def test_perceptron(x_data, w_data, b_data) -> None:
    x = Tensor(x_data, requires_grad=True)
    w = Tensor(w_data, requires_grad=True)
    b = Tensor(b_data, requires_grad=True)

    y = tanh(x @ w + b)
    y.backward()

    # Compute expected gradients
    x_np = np.array(x_data)
    w_np = np.array(w_data)
    b_np = np.array(b_data)

    z_np = x_np @ w_np + b_np
    y_np = np.tanh(z_np)
    dy = np.ones_like(y_np)
    dz = dy * (1.0 - y_np**2)

    dx = dz @ w_np.T
    dw = x_np.T @ dz
    db = dz.sum(axis=0)

    assert np.allclose(y.data, y_np)
    assert x.grad is not None
    assert np.allclose(x.grad, dx)
    assert w.grad is not None
    assert np.allclose(w.grad, dw)
    assert b.grad is not None
    assert np.allclose(b.grad, db)

    # TODO: add numeric gradient check


# TODO: add perceptron loss test
