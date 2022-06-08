import numpy as np
from .const import LOG_FLOOR


def is_number(number):
    """
    Find out if a string is a number.

    :param number: The string to test.
    :return: bool
    """
    try:
        float(number)
        return True
    except ValueError:
        return False


def set_log_floor(x):
    """
    This function takes a vector x and removes negatives, nans, and infs
    to make it safe for log/log10.

    :param x: Pointer to the vector.
    :return: None
    """
    x[np.where((x <= 0) | (np.isnan(x)) | (np.isinf(x)))] = LOG_FLOOR


def trapezoidal(f, a, b, n):
    """
    This is a vectorized trapezoidal rule for integration. See:
      https://hplgit.github.io/prog4comp/doc/pub/p4c-sphinx-Python/._pylight004.html

    :param f: The function to integrate (callable).
    :param a: The lower bound.
    :param b: The upper bound.
    :param n: How many trapezoids.
    :return: The integral.
    """

    h = float(b - a) / n
    x = np.linspace(a, b, n + 1)
    s = np.sum(f(x)) - 0.5 * f(a) - 0.5 * f(b)
    return h * s
