from prinpy.global_curves import NetworkFitter, TrainingCallback
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray


def create_parabola(n_points: int = 200, noise: float = 0.05) -> NDArray:
    t = np.linspace(-1, 1, n_points)
    x = t
    y = t ** 2
    pts = np.column_stack([x, y])
    pts += np.random.default_rng(0).normal(scale=noise, size=pts.shape)
    return pts


def create_3d_parabola(n_points: int = 200, noise: float = 0.05) -> NDArray:
    t = np.linspace(-1, 1, n_points)
    x = t
    y = t ** 2
    z = t ** 3
    pts = np.column_stack([x, y, z])
    pts += np.random.default_rng(0).normal(scale=noise, size=pts.shape)
    return pts


def main():
    data = create_parabola()

    callback = TrainingCallback(print_progress=True, every_n_epochs=10)
    fitter = NetworkFitter(dim=2, n_hidden=32, lr=1e-3, epochs=1000, callback=callback)
    curve = fitter.fit(data)

    # Plot original data vs reconstruction
    import torch
    with torch.no_grad():
        data_t = torch.tensor(data, dtype=torch.float32)
        reconstructed = curve.model(data_t).numpy()

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(data[:, 0], data[:, 1], s=10, alpha=0.5, label="Input data")
    ax.scatter(reconstructed[:, 0], reconstructed[:, 1], s=10, alpha=0.5, label="Reconstructed")
    ax.set_title("NetworkFitter — autoencoder reconstruction (2D)")
    ax.legend()
    plt.tight_layout()
    plt.show()

    # 3D parabola example
    data3d = create_3d_parabola()

    callback3d = TrainingCallback(print_progress=True, every_n_epochs=10)
    fitter3d = NetworkFitter(dim=3, n_hidden=32, lr=1e-3, epochs=1000, callback=callback3d)
    curve3d = fitter3d.fit(data3d)

    with torch.no_grad():
        data3d_t = torch.tensor(data3d, dtype=torch.float32)
        reconstructed3d = curve3d.model(data3d_t).numpy()

    fig3d = plt.figure(figsize=(7, 5))
    ax3d = fig3d.add_subplot(projection="3d")
    ax3d.scatter(data3d[:, 0], data3d[:, 1], data3d[:, 2], s=10, alpha=0.5, label="Input data")
    ax3d.scatter(reconstructed3d[:, 0], reconstructed3d[:, 1], reconstructed3d[:, 2], s=10, alpha=0.5, label="Reconstructed")
    ax3d.set_title("NetworkFitter — autoencoder reconstruction (3D)")
    ax3d.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()


