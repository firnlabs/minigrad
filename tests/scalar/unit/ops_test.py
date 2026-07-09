"""Tests for `ops.py`."""

import math
import unittest

from minigrad.scalar import Scalar, ops


class ValueTest(unittest.TestCase):
    def test_add_forward(self) -> None:
        a_val = 2.0
        b_val = 3.0
        a = Scalar(a_val)
        b = Scalar(b_val)
        exp_data = a_val + b_val
        result = a + b
        self.assertEqual(exp_data, result.data)
        self.assertEqual(ops.OpType.ADD, result.op.op_type)
        self.assertEqual([a, b], result.children)

    def test_add_backward(self) -> None:
        a = Scalar(2.0)
        b = Scalar(3.0)
        result = a + b
        grad_output = 2.0
        result.op.backward(grad_output)
        self.assertEqual(grad_output, a.grad)
        self.assertEqual(grad_output, b.grad)

    def test_radd_forward(self) -> None:
        a_val = 2.0
        a = Scalar(a_val)
        b = 3.0
        exp_data = b + a_val
        result = b + a
        self.assertEqual(exp_data, result.data)
        self.assertEqual(ops.OpType.ADD, result.op.op_type)
        self.assertEqual(a.data, result.children[0].data)
        self.assertEqual(b, result.children[1].data)

    def test_multiply_forward(self) -> None:
        a_val = 2.0
        b_val = 3.0
        a = Scalar(a_val)
        b = Scalar(b_val)
        exp_data = a_val * b_val
        result = a * b
        self.assertEqual(exp_data, result.data)
        self.assertEqual(ops.OpType.MULTIPLY, result.op.op_type)
        self.assertEqual([a, b], result.children)

    def test_multiply_backward(self) -> None:
        a_val = 2.0
        b_val = 3.0
        a = Scalar(a_val)
        b = Scalar(b_val)
        result = a * b
        grad_output = 2.0
        result.op.backward(grad_output)
        self.assertEqual(grad_output * b_val, a.grad)
        self.assertEqual(grad_output * a_val, b.grad)

    def test_rmul_forward(self) -> None:
        a_val = 2.0
        a = Scalar(a_val)
        b = 3.0
        exp_data = b * a_val
        result = b * a
        self.assertEqual(exp_data, result.data)
        self.assertEqual(ops.OpType.MULTIPLY, result.op.op_type)
        self.assertEqual(a.data, result.children[0].data)
        self.assertEqual(b, result.children[1].data)

    def test_negate_forward(self) -> None:
        a_val = 2.0
        a = Scalar(a_val)
        exp_data = -a_val
        result = -a
        self.assertEqual(exp_data, result.data)
        self.assertEqual(ops.OpType.NEGATE, result.op.op_type)
        self.assertEqual([a], result.children)

    def test_sub_forward(self) -> None:
        a_val = 2.0
        b_val = 3.0
        a = Scalar(a_val)
        b = Scalar(b_val)
        exp_data = a_val - b_val
        result = a - b
        self.assertEqual(exp_data, result.data)
        self.assertEqual(ops.OpType.ADD, result.op.op_type)
        self.assertEqual(a, result.children[0])
        self.assertEqual(-b.data, result.children[1].data)

    def test_sub_backward(self) -> None:
        a_val = 2.0
        b_val = 3.0
        a = Scalar(a_val)
        b = Scalar(b_val)
        result = a - b
        grad_output = 2.0
        result.op.backward(grad_output)
        self.assertEqual(grad_output, a.grad)
        z = result.children[1]  # result of (-b)
        self.assertEqual(grad_output, z.grad)
        z.op.backward(z.grad)
        self.assertEqual(-grad_output, z.children[0].grad)

    def test_rsub_forward(self) -> None:
        a_val = 2.0
        b_val = 3.0
        b = Scalar(b_val)
        exp_data = a_val - b_val
        result = a_val - b
        self.assertEqual(exp_data, result.data)
        self.assertEqual(ops.OpType.ADD, result.op.op_type)
        self.assertEqual(-b_val, result.children[0].data)
        self.assertEqual(a_val, result.children[1].data)

    def test_pow_forward(self) -> None:
        data = 2.0
        exponent = 3
        a = Scalar(data)
        exp_data = data**exponent
        result = a**exponent
        self.assertAlmostEqual(exp_data, result.data)
        self.assertEqual(ops.OpType.POWER, result.op.op_type)
        self.assertEqual([a], result.children)

    def test_pow_backward(self) -> None:
        data = 3.0
        exponent = 4
        a = Scalar(data)
        result = a**exponent
        grad_output = 2.0
        result.op.backward(grad_output)
        exp_grad = grad_output * exponent * data ** (exponent - 1)
        self.assertEqual(exp_grad, a.grad)

    def test_exp_forward(self) -> None:
        data = 2.0
        a = Scalar(data)
        exp_data = math.exp(data)
        result = ops.exp(a)
        self.assertAlmostEqual(exp_data, result.data)
        self.assertEqual(ops.OpType.EXP, result.op.op_type)
        self.assertEqual([a], result.children)

    def test_exp_backward(self) -> None:
        data = 2.0
        a = Scalar(data)
        result = ops.exp(a)
        grad_output = 2.0
        exp_grad = grad_output * math.exp(data)
        result.op.backward(grad_output)
        self.assertAlmostEqual(exp_grad, a.grad)

    def test_tanh_forward(self) -> None:
        data = 1.0
        a = Scalar(data)
        exp_data = math.tanh(data)
        result = ops.tanh(a)
        self.assertAlmostEqual(exp_data, result.data)
        self.assertEqual(ops.OpType.TANH, result.op.op_type)
        self.assertEqual([a], result.children)

    def test_tanh_backward(self) -> None:
        data = 1.0
        a = Scalar(data)
        result = ops.tanh(a)
        grad_output = 2.0
        exp_grad = grad_output * (1 - math.tanh(data) ** 2)
        result.op.backward(grad_output)
        self.assertAlmostEqual(exp_grad, a.grad)
