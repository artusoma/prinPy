from functools import wraps
import numpy as np


def _check_shape(expected_dim: int):
    """A decorator to check the shape of the input data. It checks if the input
    data has the expected number of dimensions, and raises a ValueError if it
    does not.
    """

    def _dec(f):
        @wraps(f)
        def _do_check(self, X: np.ndarray, *args, **kwargs):
            if X.ndim != expected_dim:
                raise ValueError(
                    f"Expected input to have {expected_dim} dimensions, but got {X.ndim}."
                )
            return f(self, X, *args, **kwargs)

        return _do_check

    return _dec
