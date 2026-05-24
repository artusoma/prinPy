from pathlib import Path

from prinpy.global_curves import NetworkFitter, TrainingCallback
import matplotlib.pyplot as plt

from _test_data import *

_FIGURES = Path(__file__).parent / "figures"


def main():
    _FIGURES.mkdir(exist_ok=True)

    data_generators_2d = {
        "Spiral": create_spiral,
        "Parabola": create_parabola,
    }
    n_hidden_values = [8, 16, 32]

    for name, generator in data_generators_2d.items():
        data = generator()
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        for idx, n_hidden in enumerate(n_hidden_values):
            curve = NetworkFitter(
                dim=2,
                n_hidden=n_hidden,
                lr=0.01,
                epochs=500,
                callback=TrainingCallback(print_progress=False),
            ).fit(data)
            plot_2d_data(
                ax[idx],
                data,
                curve,
                title=f"NetworkFitter — n_hidden={n_hidden}",
            )
        fig.tight_layout()
        fig.savefig(_FIGURES / f"{name}_network_fitting.png")

    data_generators_3d = {
        "3D Spiral": create_3d_spiral,
        "3D Parabola": create_3d_parabola,
    }
    n_hidden_values_3d = [16, 32]

    fig = plt.figure(figsize=(10, 8))
    for col, n_hidden in enumerate(n_hidden_values_3d):
        for row, (shape_name, generator) in enumerate(data_generators_3d.items()):
            data = generator()
            ax3d = fig.add_subplot(2, 2, row * 2 + col + 1, projection="3d")
            curve = NetworkFitter(
                dim=3,
                n_hidden=n_hidden,
                lr=0.01,
                epochs=500,
                callback=TrainingCallback(print_progress=False),
            ).fit(data)
            plot_3d_data(ax3d, data, curve, title=f"{shape_name} — n_hidden={n_hidden}")
    fig.tight_layout()
    fig.savefig(_FIGURES / "3d_network_fitting.png")


if __name__ == "__main__":
    main()
