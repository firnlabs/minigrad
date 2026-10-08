from collections.abc import Sequence

import pytest

from minigrad.scalar import optim
from minigrad.scalar import Scalar


class SimpleOptimizer(optim.Optimizer):
    def step(self) -> None:
        for p in self._parameters:
            p.data += 1.0


@pytest.fixture
def parameters() -> Sequence[Scalar]:
    p1 = Scalar(1.0, name="p1", requires_grad=True)
    p2 = Scalar(2.0, name="p2", requires_grad=True)
    return [p1, p2]


@pytest.fixture
def simple_optimizer(parameters) -> SimpleOptimizer:
    return SimpleOptimizer(parameters)


class TestOptimizer:
    """Tests for the Optimizer base class."""

    def test_init(self, parameters) -> None:
        opt = SimpleOptimizer(parameters)

        assert opt._parameters == parameters

    def test_zero_grad(self, simple_optimizer) -> None:
        for i, p in enumerate(simple_optimizer.parameters):
            p.grad = float(i)

        simple_optimizer.zero_grad()

        assert all(p.grad is None for p in simple_optimizer.parameters)


class TestSGD:
    """Tests for SGD class."""

    def test_init(self, parameters) -> None:
        sgd = optim.SGD(parameters)

        assert sgd._parameters == parameters
        assert sgd.lr == 0.1

    def test_step(self, parameters) -> None:
        sgd = optim.SGD(parameters, lr=0.5)
        for p in sgd._parameters:
            p.grad = 1.0

        sgd.step()

        assert sgd._parameters[0].data == 0.5
        assert sgd._parameters[1].data == 1.5

    def test_step_skips_none_grad(self, parameters) -> None:
        sgd = optim.SGD(parameters, lr=0.5)
        sgd._parameters[0].grad = 1.0
        sgd._parameters[1].grad = None

        sgd.step()

        assert sgd._parameters[0].data == 0.5
        assert sgd._parameters[1].data == 2.0  # no change
