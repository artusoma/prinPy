from prinpy.global_curves import NetworkFitter, TrainingCallback
from prinpy.local_curves import ConstrainedFitter, GreedyFit, SVDFit
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray

from _test_data import create_parabola, create_3d_parabola, create_spiral

def network_fitter_factory(shape: int) -> NetworkFitter:
    callback = TrainingCallback(print_progress=True, every_n_epochs=500)
    fitter = NetworkFitter(
        dim=shape, n_hidden=32, lr=1e-3, epochs=5000, callback=callback
    )
    return fitter

def constrained_fitter_factory_greedy(_) -> ConstrainedFitter:
    fitter = ConstrainedFitter(algorithm=GreedyFit(inner_radius=0.9), tolerance=0.1)
    return fitter

def main():
    datum = (create_spiral(), create_parabola(), create_3d_parabola())
    fitters = (
        network_fitter_factory,
        constrained_fitter_factory_greedy,
    )
    for data in datum:
        for fitter in fitters:
            fitter = fitter(data.shape[1])
            curve = fitter.fit(data)

            # Plot original data vs reconstruction
            reconstructed = curve.project(data).points

            fig, ax = plt.subplots(figsize=(6, 5))
            ax.scatter(data[:, 0], data[:, 1], s=10, alpha=0.5, label="Input data")
            ax.scatter(
                reconstructed[:, 0],
                reconstructed[:, 1],
                s=10,
                alpha=0.5,
                label="Reconstructed",
            )
            ax.set_title(f"NetworkFitter — autoencoder reconstruction")
            ax.legend()
            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    main()
