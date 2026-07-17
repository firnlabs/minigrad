"""Tests backpropagation of gradients in Tensor."""

import numpy as np

from minigrad.tensor import Tensor


def test_backprop_one_op() -> None:
    x = Tensor([1.0, 2.0], name="a", requires_grad=True)
    y = Tensor([3.0, 4.0], name="b", requires_grad=True)
    z = x + y

    z.backward()

    assert z.grad is not None
    assert np.allclose(z.grad, np.ones((2,)))
    assert x.grad is not None
    assert np.allclose(x.grad, z.grad)
    assert y.grad is not None
    assert np.allclose(y.grad, z.grad)
