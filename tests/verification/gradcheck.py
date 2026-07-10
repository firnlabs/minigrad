"""Check gradient computations."""

from collections.abc import Callable
import math


def numeric_gradient(f: Callable, x: float, h: float = 1e-6) -> float:
    """Compute the numerical gradient of f(x).

    Uses symmetric difference quotient to approximate the derivative of f at x.

    Args:
        f: The function for which to compute the gradient.
        x: The point at which to compute the gradient.
        h: The step size for numerical differentiation.

    Returns:
        The approximated numerical gradient.
    """
    f_plus = f(x + h)
    f_minus = f(x - h)
    return (f_plus - f_minus) / (2 * h)


def check_gradient(
    grad: float, f: Callable, x: float, h: float = 1e-6, tol: float = 1e-4
) -> tuple[bool, float]:
    """Check if the analytical gradient matches the numerical gradient.

    Args:
        grad: The analytical gradient.
        f: The function for which to check the gradient.
        x: The point at which to check the gradient.
        h: The step size for numerical differentiation.
        tol: The tolerance for the gradient check.

    Returns:
        A tuple containing a boolean indicating if the gradients match and the
        numerical gradient.
    """
    num_grad = numeric_gradient(f, x, h)
    is_match = math.isclose(num_grad, grad, rel_tol=tol, abs_tol=tol)
    return is_match, num_grad
