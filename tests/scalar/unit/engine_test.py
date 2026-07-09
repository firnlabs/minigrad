"""Tests for `scalar/engine.py`."""

from minigrad.scalar import Scalar


def test_init_default() -> None:
    x = Scalar(2)

    assert x._data == 2.0
    assert x.op is None
    assert x.name == "unnamed"
    assert x.grad == 0.0


def test_name_setter() -> None:
    x = Scalar(2, name="initial")
    assert x.name == "initial"
    x.name = "updated"
    assert x.name == "updated"


def test_op_setter() -> None:
    x = Scalar(2)
    assert x.op is None

    class MockOp:
        op_type = "mock"

    mock_op = MockOp()
    x.op = mock_op
    assert x.op is mock_op


def test_grad_setter() -> None:
    x = Scalar(2)
    assert x.grad == 0.0
    x.grad = 1.5
    assert x.grad == 1.5


def test_data_property() -> None:
    x = Scalar(2.5)
    assert x.data == 2.5


def test_repr_without_op() -> None:
    x = Scalar(2, name="x")
    assert repr(x) == "Value(data=2.0, operator=none, name=x)"


def test_repr_with_op() -> None:
    class MockOp:
        op_type = "mock_op"

    mock_op = MockOp()
    x = Scalar(2, op=mock_op, name="x")
    assert repr(x) == "Value(data=2.0, operator=mock_op, name=x)"
