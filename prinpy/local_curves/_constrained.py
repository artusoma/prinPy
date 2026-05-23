"""
This file implements algorithms from
https://www.sciencedirect.com/science/article/pii/S0377042715005956
"""

from prinpy.interfaces import ICurveFitter, ICurve, Projection
from prinpy._rs import clpg, clppca
from prinpy.containers import SplineCurve
import numpy as np
from dataclasses import dataclass


class FitAlgorithm:
    pass


@dataclass(frozen=True)
class GreedyFit(FitAlgorithm):
    """
    GreedyFit class for fitting curves using a greedy algorithm.

    Implementation of CLPCg (CLPC using greedy thinking) algorithm (section 3.2 of paper).
    Sets the next vertex to the mean of the points within the current radius but beyond an inner radius.
    If the current radius is r, then the next vertex is the mean of points that satisfy r * inner_radius < distance <= r.

    Args:
        inner_radius (float): The inner radius parameter for the greedy fit algorithm.
    """

    inner_radius: float = 0.9


@dataclass(frozen=True)
class SearchFit(FitAlgorithm):
    """
    SearchFit class for fitting curves using a search algorithm.

    Implementation of CLPCs (CLPC using one dimensional search) algorithm (section 3.3 of paper).
    Sets the next vertex to the point that minimizes the local fitting error among a set of candidates.

    Args:
        trials (int): The number of candidate points to evaluate for the next vertex.
    """

    trials: int = 20


@dataclass(frozen=True)
class SVDFit(FitAlgorithm):
    """
    SVDFit class for fitting
    curves using Singular Value Decomposition (SVD).
    """

    pass


class ConstrainedFitter(ICurveFitter):
    """
    ConstrainedFitter class for fitting curves with constraints.

    Fits a piecewise-linear curve to ordered data using the
    Constrained Local Principal Curve (CLPC) procedure.

    Starting from an initial vertex, the algorithm repeatedly:
      1. Gathers all points within an adaptive radius of the current
         vertex, preserving their original ordering.
      2. Computes the next vertex using a FitAlgorithm (e.g., GreedyFit or SearchFit).
      3. Projects the points onto the candidate segment and evaluates
         the local fitting error.
      4. Accepts the segment if the error is below a threshold, then
         discards the points used for this segment.
      5. Expands or shrinks the radius when the error is too large,
         retrying until a valid segment is found.

    The process continues until no points remain, producing a monotone,
    forward-marching polyline that respects local geometric constraints.

    Args:
        algorithm (FitAlgorithm): The algorithm to use for fitting the curve.
        tolerance (float): The tolerance for the fitting algorithm.
    """

    def __init__(self, algorithm: FitAlgorithm, tolerance: float = 1e-3):
        self._algorithm = algorithm
        self._tolerance = tolerance

    def fit(self, data: np.ndarray) -> SplineCurve:
        data = np.ascontiguousarray(data, dtype=np.float32)
        match self._algorithm:
            case GreedyFit(inner_radius=inner_radius):
                fit_points = clpg(data, self._tolerance, inner_radius)
            case SVDFit():
                fit_points = clppca(data, self._tolerance)
            case _:
                raise ValueError(f"Unsupported algorithm: {self._algorithm}")
        return SplineCurve(fit_points)
