from pathlib import Path
from prinpy.local_curves import ConstrainedFitter, GreedyFit, SVDFit
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from _test_data import *

_FIGURES = Path(__file__).parent / Path("figures")

def main():
    data_generators = {
        "Spiral": create_spiral,
        "Parabola": create_parabola,
    }
    for name, generator in data_generators.items():
        data = generator()
        fig, ax = plt.subplots(2, 3, figsize=(9, 6))
        for idx, tolerance in enumerate([0.03, .075, .25]):
            curve = ConstrainedFitter(
                algorithm=GreedyFit(inner_radius=0.9),
                tolerance=tolerance,
            ).fit(data)
            plot_2d_data(
                ax[0, idx],
                data,
                curve,
                title=f"GreedyFit — tolerance={tolerance}",
            )

            curve = ConstrainedFitter(
                algorithm=SVDFit(),
                tolerance=tolerance,
            ).fit(data)
            plot_2d_data(
                ax[1, idx],
                data,
                curve,
                title=f"SVDFit — tolerance={tolerance}",
            )
        fig.tight_layout()
        fig.savefig(_FIGURES / f"{name}_curve_fitting.png")

    data_generators_3d = {
        "3D Spiral": create_3d_spiral,
        "3D Parabola": create_3d_parabola,
    }
    algorithms = {
        "GreedyFit": lambda: GreedyFit(inner_radius=0.9),
        "SVDFit": SVDFit,
    }
    tolerance = 0.05

    fig = plt.figure(figsize=(10, 8))
    for col, (shape_name, generator) in enumerate(data_generators_3d.items()):
        data = generator()
        for row, (algo_name, algo_cls) in enumerate(algorithms.items()):
            ax3d = fig.add_subplot(2, 2, row * 2 + col + 1, projection="3d")
            curve = ConstrainedFitter(algorithm=algo_cls(), tolerance=tolerance).fit(data)
            plot_3d_data(ax3d, data, curve, title=f"{shape_name} — {algo_name}")
    fig.tight_layout()
    fig.savefig(_FIGURES / "3d_curve_fitting.png")


if __name__ == "__main__":
    main()
