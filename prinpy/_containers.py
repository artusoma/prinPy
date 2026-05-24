from prinpy.interfaces import PrincipalCurve, Projection
import numpy as np
import scipy.interpolate as si
from prinpy._rs import find_nearest_points
from scipy.integrate import quad
from prinpy._utils import _check_shape
from dataclasses import dataclass

@dataclass(frozen=True)
class _Segment:
    start: np.ndarray
    end: np.ndarray

    def length(self) -> float:
        return float(np.linalg.norm(self.end - self.start)) 

class _SegmentCurve(PrincipalCurve):
    """
    Special case of spline curve where number of points is less than 4.
    Connects control points via straight line segments.

    Args:
        control_points (np.ndarray): The control points of the segment curve.
    """

    def __init__(self, control_points: np.ndarray):
        self._control_points = control_points
        diffs = np.diff(control_points, axis=0)
        self._segment_lengths = np.linalg.norm(diffs, axis=1)
        self._cumulative_lengths = np.concatenate([[0.0], np.cumsum(self._segment_lengths)])
        self._arc_length = float(self._cumulative_lengths[-1])

    def control_points(self) -> np.ndarray:
        return self._control_points

    def length(self) -> float:
        return self._arc_length

    @_check_shape(2)
    def project(self, X: np.ndarray) -> Projection:
        pts = self._control_points
        best_points = np.empty_like(X)
        best_arc_lengths = np.empty(len(X))
        best_dists_sq = np.full(len(X), np.inf)

        for i in range(len(pts) - 1):
            a, b = pts[i], pts[i + 1]
            ab = b - a
            seg_len_sq = float(np.dot(ab, ab))

            if seg_len_sq == 0.0:
                t = np.zeros(len(X))
                proj = np.tile(a, (len(X), 1))
            else:
                t = np.clip(((X - a) @ ab) / seg_len_sq, 0.0, 1.0)
                proj = a + t[:, None] * ab

            dists_sq = np.sum((X - proj) ** 2, axis=1)
            mask = dists_sq < best_dists_sq
            best_points[mask] = proj[mask]
            best_dists_sq[mask] = dists_sq[mask]
            best_arc_lengths[mask] = self._cumulative_lengths[i] + t[mask] * self._segment_lengths[i]

        return Projection(
            points=best_points,
            arc_lengths=best_arc_lengths,
            unit_lengths=best_arc_lengths / self._arc_length,
        )

    @_check_shape(1)
    def interpolate_from_length(self, X: np.ndarray) -> Projection:
        X = np.clip(X, 0.0, self._arc_length)
        idxs = np.clip(
            np.searchsorted(self._cumulative_lengths, X, side="right") - 1,
            0, len(self._segment_lengths) - 1,
        )
        local_t = np.clip(
            (X - self._cumulative_lengths[idxs]) / self._segment_lengths[idxs], 0.0, 1.0
        )
        a = self._control_points[idxs]
        b = self._control_points[idxs + 1]
        curve_points = a + local_t[:, None] * (b - a)
        return Projection(
            points=curve_points,
            arc_lengths=X,
            unit_lengths=X / self._arc_length,
        )

    @_check_shape(1)
    def interpolate_from_unit(self, X: np.ndarray) -> Projection:
        arc_lengths = np.clip(X, 0.0, 1.0) * self._arc_length
        idxs = np.clip(
            np.searchsorted(self._cumulative_lengths, arc_lengths, side="right") - 1,
            0, len(self._segment_lengths) - 1,
        )
        local_t = np.clip(
            (arc_lengths - self._cumulative_lengths[idxs]) / self._segment_lengths[idxs], 0.0, 1.0
        )
        a = self._control_points[idxs]
        b = self._control_points[idxs + 1]
        curve_points = a + local_t[:, None] * (b - a)
        return Projection(
            points=curve_points,
            arc_lengths=arc_lengths,
            unit_lengths=X,
        )




class _SplineCurve(PrincipalCurve):
    """
    A class representing a spline curve.

    Args:
        control_points (np.ndarray): The control points of the spline curve.
    """
    def __init__(self, control_points: np.ndarray):
        self._spline = si.make_splprep(control_points.T, s=0)[0]
        self._arc_length = self._calculate_arc_length()
        self._control_points = control_points

    def control_points(self) -> np.ndarray:
        return self._control_points

    @_check_shape(2)
    def project(self, X: np.ndarray, sample_resolution: int = 500) -> Projection:
        X = np.ascontiguousarray(X, dtype=np.float32)
        samples = np.linspace(0, 1, sample_resolution)
        curve_points = np.array(self._spline(samples)).T
        curve_points = np.ascontiguousarray(curve_points, dtype=np.float32)
        point_idxs = find_nearest_points(X, curve_points)

        # Get points and arc-lengths
        points = curve_points[point_idxs]
        lengths = samples[point_idxs] * self.length()
        unit_lengths = samples[point_idxs]

        return Projection(points=points, arc_lengths=lengths, unit_lengths=unit_lengths)

    @_check_shape(1)
    def interpolate_from_length(self, X: np.ndarray) -> Projection:
        """Return the projection of the input points onto the curve.

        Args:
            X (np.ndarray): A 1D array of arc-lengths.

        Returns:
            Projection: A Projection object containing the projected points and their corresponding
                arc-lengths.
        """
        curve_points = np.array(self._spline(X / self.length())).T
        return Projection(
            points=curve_points, arc_lengths=X, unit_lengths=X / self.length()
        )

    @_check_shape(1)    
    def interpolate_from_unit(self, X: np.ndarray) -> Projection:
        """Return the projection of the input points onto the curve.

        Args:
            X (np.ndarray): A 1D array of unit-lengths.

        Returns:
            Projection: A Projection object containing the projected points and their corresponding
                unit-lengths.
        """
        curve_points = np.array(self._spline(X)).T
        return Projection(
            points=curve_points, arc_lengths=X * self.length(), unit_lengths=X
        )

    def length(self) -> float:
        return self._arc_length

    def _calculate_arc_length(self) -> float:
        """Calculate the arc length of the spline curve using numerical integration.

        Args:
            samples (np.ndarray): Sample points along the curve.
                Has shape (n_samples, n_dimensions).

        Returns:
            float: The arc length of the curve.
        """

        def integrand(t):
            dx_dt = self._spline(t, nu=1)
            return np.linalg.norm(dx_dt)

        k = self._spline.k
        interior_knots = self._spline.t[k + 1 : -(k + 1)]
        interior_knots = interior_knots[(interior_knots > 0.0) & (interior_knots < 1.0)]
        arc_length, _ = quad(integrand, 0, 1, points=interior_knots)
        return arc_length
