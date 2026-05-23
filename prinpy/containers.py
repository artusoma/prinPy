from prinpy.interfaces import ICurve, Projection
import numpy as np
import scipy.interpolate as si
from prinpy._rs import find_nearest_points
from scipy.integrate import quad


class SplineCurve(ICurve):
    def __init__(self, control_points: np.ndarray):
        self._spline = si.make_splprep(control_points.T, s=0)[0]
        self._arc_length = self._calculate_arc_length()
        self._control_points = control_points

    def get_control_points(self) -> np.ndarray:
        return self._control_points

    def project(self, X: np.ndarray, sample_resolution: int = 500) -> Projection:
        samples = np.linspace(0, 1, sample_resolution)
        curve_points = np.array(self._spline(samples)).T
        point_idxs = find_nearest_points(X, curve_points)

        # Get points and arc-lengths
        points = curve_points[point_idxs]
        lengths = samples[point_idxs] * self.get_length()
        unit_lengths = samples[point_idxs]

        return Projection(points=points, arc_lengths=lengths, unit_lengths=unit_lengths)

    def interpolate_from_length(self, /, X: np.ndarray) -> Projection:
        """Return the projection of the input points onto the curve.

        Args:
            X (np.ndarray): A 1D array of arc-lengths.

        Returns:
            Projection: A Projection object containing the projected points and their corresponding
                arc-lengths.
        """
        curve_points = np.array(self._spline(X / self.get_length())).T
        return Projection(
            points=curve_points, arc_lengths=X, unit_lengths=X / self.get_length()
        )

    def interpolate_from_unit(self, /, X: np.ndarray) -> Projection:
        """Return the projection of the input points onto the curve.

        Args:
            X (np.ndarray): A 1D array of unit-lengths.

        Returns:
            Projection: A Projection object containing the projected points and their corresponding
                unit-lengths.
        """
        curve_points = np.array(self._spline(X)).T
        return Projection(
            points=curve_points, arc_lengths=X * self.get_length(), unit_lengths=X
        )

    def get_length(self) -> float:
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

        arc_length, _ = quad(integrand, 0, 1)
        return arc_length
