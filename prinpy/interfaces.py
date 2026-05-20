import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class Projection:
    arc_lengths: np.ndarray
    unit_lengths: np.ndarray
    points: np.ndarray


class ICurve:
    def project(self, X: np.ndarray) -> Projection:
        raise NotImplementedError("Subclasses must implement this method")

    def interpolate_from_length(self, X: np.ndarray) -> Projection:
        raise NotImplementedError("Subclasses must implement this method")

    def interpolate_from_unit(self, X: np.ndarray) -> Projection:
        raise NotImplementedError("Subclasses must implement this method")

    def get_length(self) -> float:
        raise NotImplementedError("Subclasses must implement this method")


class ICurveFitter:
    def fit(self, data: np.ndarray) -> ICurve:
        raise NotImplementedError("Subclasses must implement this method")

    def update(self, data: np.ndarray) -> ICurve:
        """Most curve calculators will not need to implement this method, but it is here
        for neural network based curve calculators that may want to update the model with new data.
        By default, this method will raise a NotImplementedError"""
        raise NotImplementedError("Subclasses must implement this method")
