import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class Projections:
    arc_lengths: np.ndarray[np.float64]
    points: np.ndarray[np.float64]

class ICurve:
    def project(self, /, X: np.ndarray[np.float64]) -> Projections:
        raise NotImplementedError("Subclasses must implement this method")
    
    def interpolate(self, /, X: np.ndarray[np.float64]) -> Projections:
        raise NotImplementedError("Subclasses must implement this method")
    
    def get_length(self) -> float:
        raise NotImplementedError("Subclasses must implement this method")
    

class ICurveCalculator:
    def fit(self, data: np.ndarray[np.float64]) -> ICurve:
        raise NotImplementedError("Subclasses must implement this method")

    def update(self, data: np.ndarray[np.float64]) -> ICurve:
        """Most curve calculators will not need to implement this method, but it is here
        for neural network based curve calculators that may want to update the model with new data.
        By default, this method will raise a NotImplementedError"""
        raise NotImplementedError("Subclasses must implement this method")
