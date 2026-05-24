import pytest
import numpy as np

pytest.importorskip("torch")
from prinpy.global_curves import _NetworkCurve, NetworkFitter

def test_fitter_raise_shape_error():
    fitter = NetworkFitter(dim=2, n_hidden=10, lr=0.01, epochs=10)
    with pytest.raises(ValueError):
        fitter.fit(np.random.rand(10))
