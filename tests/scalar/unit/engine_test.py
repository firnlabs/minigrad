"""Tests for `scalar/engine.py`."""

import unittest

from minigrad.scalar import Scalar


class ValueTest(unittest.TestCase):
    """Tests for `Value` class."""

    def test_init_default(self) -> None:
        x = Scalar(2)

        self.assertEqual(2.0, x._data)
        self.assertEqual(None, x._op)
        self.assertEqual([], x._children)
        self.assertEqual("unnamed", x.name)
        self.assertEqual(0.0, x.grad)
