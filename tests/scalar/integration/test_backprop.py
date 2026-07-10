"""Tests backpropagation of gradients in Scalar."""

from collections.abc import Callable
import math

from minigrad.scalar import ops
from minigrad.scalar import Scalar
from tests.verification import gradcheck


def _check_gradient(target: Scalar, f: Callable) -> None:
    """Check if the analytical gradient matches the numerical gradient."""
    is_match, num_grad = gradcheck.check_gradient(grad=target.grad, f=f, x=target.data)
    assert is_match, (
        f"Gradient check failed for {target.name}:"
        f" expected {num_grad}, actual: {target.grad}"
    )


def test_backprop_no_value_reuse() -> None:
    """Tests backpropagation of gradients with no value reuse."""
    a = Scalar(2.0, name="a", requires_grad=True)
    b = Scalar(3.0, name="b", requires_grad=True)
    c = Scalar(4.0, name="c", requires_grad=True)
    d = a * b
    e = d + c

    e.backward()

    assert a.grad == 3.0
    assert b.grad == 2.0
    assert c.grad == 1.0
    assert d.grad == 1.0
    assert e.grad == 1.0


def test_backprop_value_reuse() -> None:
    """Tests backpropagation of gradients with value reuse."""
    a = Scalar(2.0, name="a", requires_grad=True)
    b = Scalar(3.0, name="b", requires_grad=True)
    c = a * b
    d = c + a
    e = d * b

    e.backward()

    assert a.grad == b.data * b.data + b.data
    assert b.grad == d.data + b.data * a.data
    assert c.grad == b.data
    assert d.grad == b.data
    assert e.grad == 1.0


def test_backprop_perceptron() -> None:
    """Tests backpropagation of gradients in a simple perceptron."""
    x1 = Scalar(1.0, name="x1")
    x2 = Scalar(2.0, name="x2")
    w1 = Scalar(3.0, name="w1", requires_grad=True)
    w2 = Scalar(-1.0, name="w2", requires_grad=True)
    b = Scalar(0.5, name="b", requires_grad=True)

    # Forward pass
    z = x1 * w1 + x2 * w2 + b
    y_pred = ops.tanh(z)

    # Backward pass
    y_pred.backward()

    # Perceptron as a function
    def func(w1_val: float, w2_val: float, b_val: float) -> float:
        z_val = x1.data * w1_val + x2.data * w2_val + b_val
        return math.tanh(z_val)

    _check_gradient(w1, lambda w1_val: func(w1_val, w2.data, b.data))
    _check_gradient(w2, lambda w2_val: func(w1.data, w2_val, b.data))
    _check_gradient(b, lambda b_val: func(w1.data, w2.data, b_val))
    assert x1.grad is None
    assert x2.grad is None


def test_backprop_perceptron_mse_loss() -> None:
    """Tests backpropagation on loss function over a simple perceptron."""
    x1 = Scalar(1.0, name="x1")
    x2 = Scalar(2.0, name="x2")
    w1 = Scalar(3.0, name="w1", requires_grad=True)
    w2 = Scalar(-1.0, name="w2", requires_grad=True)
    b = Scalar(0.5, name="b", requires_grad=True)
    y = Scalar(-2, name="y")

    # Forward pass
    z1 = x1 * w1
    z2 = x2 * w2
    a = z1 + z2
    c = a + b
    y_pred = ops.tanh(c)
    d = y_pred - y
    p = d**2
    loss = 0.5 * p

    # Backward pass
    loss.backward()

    assert loss.grad == 1.0
    assert p.grad == 0.5
    assert d.grad == p.grad * 2 * d.data
    assert y.grad is None
    assert y_pred.grad == d.grad
    assert c.grad == y_pred.grad * (1 - y_pred.data**2)
    assert a.grad == c.grad
    assert b.grad == c.grad
    assert z1.grad == a.grad
    assert z2.grad == a.grad
    assert w1.grad == z1.grad * x1.data
    assert w2.grad == z2.grad * x2.data
    assert x1.grad is None
    assert x2.grad is None

    # Numeric gradient checks
    def func(w1_val: float, w2_val: float, b_val: float) -> float:
        z1_val = x1.data * w1_val
        z2_val = x2.data * w2_val
        a_val = z1_val + z2_val
        c_val = a_val + b_val
        y_pred_val = math.tanh(c_val)
        d_val = y_pred_val - y.data
        p_val = d_val**2
        loss_val = 0.5 * p_val
        return loss_val

    _check_gradient(w1, lambda w1_val: func(w1_val, w2.data, b.data))
    _check_gradient(w2, lambda w2_val: func(w1.data, w2_val, b.data))
    _check_gradient(b, lambda b_val: func(w1.data, w2.data, b_val))
