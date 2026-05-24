import numpy as np
from numpy.typing import NDArray
from prinpy.interfaces import PrincipalCurve
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.axes import Axes
import seaborn as sns

sns.set_style("whitegrid")


def plot_2d_data(
    axes: Axes,
    data: NDArray,
    curve: PrincipalCurve,
    title: str = "2D Curve Fitting",
):
    reconstructed = curve.interpolate_from_unit(np.linspace(0, 1, 100)).points

    axes.scatter(x=data[:, 0], y=data[:, 1], s=10, alpha=0.5)
    axes.plot(reconstructed[:, 0], reconstructed[:, 1], alpha=0.75, c="C1")
    axes.scatter(
        x=curve.control_points()[:, 0],
        y=curve.control_points()[:, 1],
        s=20,
        alpha=0.5,
        c="C1",
    )
    axes.set_title(title)
    axes.grid(False)


def plot_3d_data(
    ax3d: Axes3D,
    data: NDArray,
    curve: PrincipalCurve,
    title: str = "3D Curve Fitting",
):
    reconstructed = curve.interpolate_from_unit(np.linspace(0, 1, 100)).points

    ax3d.scatter(
        xs=data[:, 0],
        ys=data[:, 1],
        zs=data[:, 2],  # type: ignore
        s=10,
        alpha=0.5,
        c="C0",
    )
    ax3d.plot(
        reconstructed[:, 0],
        reconstructed[:, 1],
        reconstructed[:, 2],  # type: ignore
        alpha=0.5,
        c="C1",
    )
    ax3d.scatter(
        xs=curve.control_points()[:, 0],
        ys=curve.control_points()[:, 1],
        zs=curve.control_points()[:, 2],  # type: ignore
        s=20,
        alpha=0.5,
        c="C1",
    )
    ax3d.set_title(title)


def create_spiral(n_points: int = 500, noise: float = 0.02) -> NDArray:
    theta = np.linspace(0, np.pi * 3, n_points)
    r = np.linspace(0, 1, n_points) ** 0.5

    x_data = r * np.cos(theta) + np.random.normal(scale=noise, size=n_points)
    y_data = r * np.sin(theta) + np.random.normal(scale=noise, size=n_points)
    data = np.column_stack([x_data, y_data])
    return data


def create_parabola(n_points: int = 500, noise: float = 0.025) -> NDArray:
    t = np.linspace(-1, 1, n_points)
    x = t
    y = t**2
    pts = np.column_stack([x, y])
    pts += np.random.default_rng(0).normal(scale=noise, size=pts.shape)
    return pts


def create_3d_parabola(n_points: int = 300, noise: float = 0.025) -> NDArray:
    t = np.linspace(-1, 1, n_points)
    x = t
    y = t**2
    z = t**3
    pts = np.column_stack([x, y, z])
    pts += np.random.default_rng(0).normal(scale=noise, size=pts.shape)
    return pts


def create_3d_spiral(n_points: int = 300, noise: float = 0.025) -> NDArray:
    theta = np.linspace(0, np.pi * 3, n_points)
    r = np.linspace(0, 1, n_points) ** 0.5

    x_data = r * np.cos(theta) + np.random.normal(scale=noise, size=n_points)
    y_data = r * np.sin(theta) + np.random.normal(scale=noise, size=n_points)
    z_data = r + np.random.normal(scale=noise, size=n_points)

    data = np.column_stack([x_data, y_data, z_data])
    return data
