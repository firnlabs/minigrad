from typing import Any

from minigrad.scalar import Scalar


def test_init_default() -> None:
    x = Scalar(2)

    assert x.data == 2.0
    assert x.op is None
    assert x.name == "unnamed"
    assert x.grad is None
    assert x.requires_grad is False


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

    mock_op: Any = MockOp()
    x.op = mock_op
    assert x.op is mock_op


def test_grad_setter() -> None:
    x = Scalar(2)
    x.grad = 1.5
    assert x.grad == 1.5


def test_requires_grad_setter() -> None:
    x = Scalar(2)
    x.requires_grad = False
    assert not x.requires_grad


def test_data_property() -> None:
    x = Scalar(2.5)
    assert x.data == 2.5


def test_data_setter() -> None:
    x = Scalar(2.5)
    x.data = 3.0
    assert x.data == 3.0


def test_repr_without_op() -> None:
    x = Scalar(2, name="x")
    assert repr(x) == "Scalar(data=2.0, op=none, name=x, requires_grad=False)"


def test_repr_with_op() -> None:
    class MockOp:
        op_type = "mock_op"

    mock_op: Any = MockOp()
    x = Scalar(2, op=mock_op, name="x")
    assert repr(x) == "Scalar(data=2.0, op=mock_op, name=x, requires_grad=False)"


def test_backward_grad_disabled() -> None:
    x = Scalar(2.0)
    y = Scalar(3.0)
    z = x * y
    z.requires_grad = False  # Manually disable
    w = z * 4.0
    w.backward()
    assert z.grad is None
    assert x.grad is None
    assert y.grad is None


def test_increment() -> None:
    x = Scalar(2.0)
    x += 1.0
    assert x.data == 3.0
