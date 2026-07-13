"""Tests for `scalar/nn.py`."""

import pytest

from minigrad.scalar import nn
from minigrad.scalar import Scalar


class SimpleModule(nn.Module):
    def forward(self, *args, **kwargs):
        return args[0] * 2


@pytest.fixture
def simple_module() -> SimpleModule:
    return SimpleModule()


class TestModule:
    """Tests the Module class."""

    def test_init(self, simple_module: SimpleModule) -> None:
        assert simple_module.parameters is None

    def test_zero_grad_no_params(self, simple_module: SimpleModule) -> None:
        assert simple_module.zero_grad() is None

    def test_zero_grad_with_params(self, simple_module: SimpleModule) -> None:
        param1 = Scalar(0.0, requires_grad=True)
        param1.grad = 1.0
        param2 = Scalar(0.0, requires_grad=True)
        param2.grad = 2.0
        simple_module._parameters = [param1, param2]

        simple_module.zero_grad()

        assert param1.grad is None
        assert param2.grad is None

    def test_call(self, simple_module: SimpleModule) -> None:
        val = 2.0

        result = simple_module(val)

        assert result == 4.0
