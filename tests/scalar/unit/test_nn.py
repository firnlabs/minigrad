"""Tests for `scalar/nn.py`."""

from collections.abc import Sequence

import pytest

from minigrad.scalar import nn
from minigrad.scalar import Scalar


class SimpleModule(nn.Module):
    def __init__(self, parameters: Sequence[Scalar]) -> None:
        self._parameters = parameters

    def forward(self, increment: int) -> Scalar:
        return sum(self._parameters) + Scalar(increment)


@pytest.fixture
def parameters() -> Sequence[Scalar]:
    return [
        Scalar(1.0, requires_grad=True),
        Scalar(2.0, requires_grad=True),
    ]


class TestModule:
    """Tests the Module class."""

    def test_init(self, parameters) -> None:
        module = SimpleModule(parameters)
        assert len(module.parameters) == len(parameters)

    def test_zero_grad_no_params(self) -> None:
        module = SimpleModule([])
        assert module.zero_grad() is None

    def test_zero_grad_with_params(self, parameters) -> None:
        module = SimpleModule(parameters)
        for i, p in enumerate(module.parameters, start=1):
            p.grad = i

        module.zero_grad()

        for p in module.parameters:
            assert p.grad is None

    def test_call(self, parameters) -> None:
        module = SimpleModule(parameters)
        increment = 3.0
        result = module(increment)

        assert result.data == (parameters[0] + parameters[1] + increment).data
