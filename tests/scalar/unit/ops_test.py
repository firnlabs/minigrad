"""Tests for `ops.py`."""

import math

import pytest

from minigrad.scalar import Scalar, ops

# x, y, upstream_grad
BINARY_OPERAND_CASES = [(2.0, 3.0, 2.0)]

# x, upstream_grad
UNARY_OPERAND_CASES = [(2.0, 2.0)]


def test_init_subclass():
    with pytest.raises(ValueError):

        class MockOpSubclass(ops.Op):
            def forward(self, *args) -> float:
                return 0.0

            def backward(self, *args) -> float:
                return 0.0


@pytest.mark.parametrize("x_val, y_val, upstream_grad", BINARY_OPERAND_CASES)
def test_add_forward_backward(x_val, y_val, upstream_grad):
    x = Scalar(x_val)
    y = Scalar(y_val)
    z = x + y
    exp_val = x_val + y_val
    assert z.data == exp_val
    grads = z.op.backward(upstream_grad)
    exp_grads = (upstream_grad, upstream_grad)
    assert grads == exp_grads


@pytest.mark.parametrize("x_val, y_val, upstream_grad", BINARY_OPERAND_CASES)
def test_radd_forward_backward(x_val, y_val, upstream_grad):
    x = x_val
    y = Scalar(y_val)
    z = x + y
    exp_val = x_val + y_val
    assert z.data == exp_val
    grads = z.op.backward(upstream_grad)
    exp_grads = (upstream_grad, upstream_grad)
    assert grads == exp_grads


@pytest.mark.parametrize("x_val, y_val, upstream_grad", BINARY_OPERAND_CASES)
def test_mul_forward_backward(x_val, y_val, upstream_grad):
    x = Scalar(x_val)
    y = Scalar(y_val)
    z = x * y
    exp_val = x_val * y_val
    assert z.data == exp_val
    grads = z.op.backward(upstream_grad)
    exp_grads = (upstream_grad * y_val, upstream_grad * x_val)
    assert grads == exp_grads


@pytest.mark.parametrize("x_val, y_val, upstream_grad", BINARY_OPERAND_CASES)
def test_rmul_forward_backward(x_val, y_val, upstream_grad):
    x = x_val
    y = Scalar(y_val)
    z = x * y
    exp_val = x_val * y_val
    assert z.data == exp_val
    grads = z.op.backward(upstream_grad)
    # operands are swapped (i.e. y * x)
    exp_grads = (upstream_grad * x_val, upstream_grad * y_val)
    assert grads == exp_grads


@pytest.mark.parametrize("x_val, upstream_grad", UNARY_OPERAND_CASES)
def test_neg_forward_backward(x_val, upstream_grad):
    x = Scalar(x_val)
    z = -x
    exp_val = -x_val
    assert z.data == exp_val
    grad = z.op.backward(upstream_grad)
    exp_grad = -upstream_grad
    assert grad == exp_grad


@pytest.mark.parametrize("x_val, y_val, upstream_grad", BINARY_OPERAND_CASES)
def test_sub_forward_backward(x_val, y_val, upstream_grad):
    x = Scalar(x_val)
    y = Scalar(y_val)
    z = x - y
    exp_val = x_val - y_val
    assert z.data == exp_val
    grads = z.op.backward(upstream_grad)
    # NOTE: subtraction is implemented as addition, i.e. x + (-y)
    # Need to perform manual backprop to get gradient for y
    x_grad = grads[0]
    assert x_grad == upstream_grad
    neg_y = z.op.parents[1]
    y_grad = neg_y.op.backward(grads[1])
    assert y_grad == -upstream_grad


@pytest.mark.parametrize("x_val, y_val, upstream_grad", BINARY_OPERAND_CASES)
def test_rsub_forward_backward(x_val, y_val, upstream_grad):
    x = x_val
    y = Scalar(y_val)
    z = x - y
    exp_val = x_val - y_val
    assert z.data == exp_val
    grads = z.op.backward(upstream_grad)
    # NOTE: subtraction is implemented as addition, i.e. x + (-y)
    # Need to perform manual backprop to get gradient for y
    # `rsub` will swap the operands, since x is a float, i.e.  (-y) + x
    neg_y = z.op.parents[0]
    y_grad = neg_y.op.backward(grads[0])
    assert y_grad == -upstream_grad
    x_grad = grads[1]
    assert x_grad == upstream_grad


@pytest.mark.parametrize("x_val, c, upstream_grad", BINARY_OPERAND_CASES)
def test_pow_forward_backward(x_val, c, upstream_grad):
    x = Scalar(x_val)
    z = x**c
    exp_val = x_val**c
    assert z.data == exp_val
    grad = z.op.backward(upstream_grad)
    exp_grad = upstream_grad * c * x_val ** (c - 1)
    assert grad == exp_grad


@pytest.mark.parametrize("x_val, upstream_grad", UNARY_OPERAND_CASES)
def test_exp_forward_backward(x_val, upstream_grad):
    x = Scalar(x_val)
    z = ops.exp(x)
    exp_val = math.exp(x_val)
    assert z.data == exp_val
    grad = z.op.backward(upstream_grad)
    exp_grad = upstream_grad * exp_val
    assert grad == exp_grad


@pytest.mark.parametrize("x_val, upstream_grad", UNARY_OPERAND_CASES)
def test_tanh_forward_backward(x_val, upstream_grad):
    x = Scalar(x_val)
    z = ops.tanh(x)
    exp_val = math.tanh(x_val)
    assert z.data == exp_val
    grad = z.op.backward(upstream_grad)
    exp_grad = upstream_grad * (1 - exp_val**2)
    assert grad == exp_grad
