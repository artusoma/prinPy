"""Public interfaces for principal curve fitting.

This module defines the two core abstractions used throughout prinpy:

- :class:`PrincipalCurve` — a fitted curve that can project data, interpolate
  points, and report its arc length.
- :class:`CurveFitter` — an algorithm that produces a :class:`PrincipalCurve`
  from raw data.

All concrete implementations (spline-based, segment-based, neural-network-based,
etc.) implement these interfaces, so downstream code only ever depends on the
types defined here.
"""

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class Projection:
    """The result of projecting a set of points onto a :class:`PrincipalCurve`.

    All three arrays are parallel: index *i* in each array corresponds to the
    same input point.

    Attributes:
        arc_lengths: Distances along the curve from its start to each projected
            point, measured in the same units as the input data.
            Shape ``(n,)``.
        unit_lengths: Normalised position of each projected point along the
            curve in the range ``[0, 1]``, where ``0`` is the start and ``1``
            is the end. Equivalent to ``arc_lengths / curve.length()``.
            Shape ``(n,)``.
        points: Coordinates of the projected points on the curve.
            Shape ``(n, d)`` where *d* is the dimensionality of the data.
    """

    arc_lengths: np.ndarray
    unit_lengths: np.ndarray
    points: np.ndarray


class PrincipalCurve(ABC):
    """Abstract base class for a fitted principal curve.

    A principal curve is a smooth, one-dimensional manifold that passes through
    the middle of a dataset. Once fitted, it supports two primary operations:

    - **Projection**: mapping arbitrary data points onto their nearest location
      on the curve (see :meth:`project`).
    - **Interpolation**: evaluating the curve at a given arc length or
      normalised position (see :meth:`interpolate_from_length` and
      :meth:`interpolate_from_unit`).

    Concrete subclasses must implement all abstract methods. Consumers of this
    interface should type-hint against :class:`PrincipalCurve` rather than any
    specific implementation.
    """

    @abstractmethod
    def project(self, X: np.ndarray) -> Projection:
        """Project data points onto the nearest location on the curve.

        For each point in *X*, finds the closest point on the curve and returns
        its coordinates together with the corresponding arc length and
        normalised position.

        Args:
            X: Input data of shape ``(n, d)``.

        Returns:
            A :class:`Projection` containing the projected coordinates,
            arc lengths, and unit lengths for every input point.
        """

    @abstractmethod
    def interpolate_from_length(self, X: np.ndarray) -> Projection:
        """Evaluate the curve at given arc-length positions.

        Args:
            X: 1-D array of arc lengths in the range ``[0, curve.length()]``,
               shape ``(n,)``.

        Returns:
            A :class:`Projection` whose ``points`` are the curve coordinates
            at each requested arc length.
        """

    @abstractmethod
    def interpolate_from_unit(self, X: np.ndarray) -> Projection:
        """Evaluate the curve at given normalised positions.

        Args:
            X: 1-D array of unit positions in the range ``[0, 1]``,
               shape ``(n,)``.

        Returns:
            A :class:`Projection` whose ``points`` are the curve coordinates
            at each requested unit position.
        """

    @abstractmethod
    def length(self) -> float:
        """Total arc length of the curve.

        Returns:
            The arc length measured in the same units as the input data.
        """

    @abstractmethod
    def control_points(self) -> np.ndarray:
        """Key points that define or summarise the shape of the curve.

        The meaning of "control points" varies by implementation — they may be
        spline knots, segment endpoints, or representative samples — but they
        always lie on the curve and are ordered from start to end.

        Returns:
            Array of shape ``(k, d)`` where *k* is the number of control points
            and *d* is the dimensionality of the data.
        """


class CurveFitter(ABC):
    """Abstract base class for algorithms that fit a :class:`PrincipalCurve`.

    Subclasses encapsulate a specific fitting strategy (e.g. greedy segment
    growing, SVD-based fitting, or neural-network optimisation). The fitted
    curve is always returned as a :class:`PrincipalCurve` so that callers
    remain decoupled from the underlying implementation.
    """

    @abstractmethod
    def fit(self, data: np.ndarray) -> PrincipalCurve:
        """Fit a principal curve to the given data.

        Args:
            data: Input data of shape ``(n, d)`` where *n* is the number of
                samples and *d* is the number of dimensions.

        Returns:
            A fitted :class:`PrincipalCurve` that summarises the structure of
            *data*.
        """

