import numpy as np
from prinpy.local_curves import ConstrainedFitter, GreedyFit

# Noisy 2D spiral
theta = np.linspace(0, 3 * np.pi, 400)
r = np.linspace(0, 1, 400) ** 0.5
data = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
data += np.random.normal(scale=0.02, size=data.shape)

# Fit a principal curve
curve = ConstrainedFitter(algorithm=GreedyFit(), tolerance=0.05).fit(data)

# Project data onto the curve — returns arc lengths, unit positions, and coordinates
projection = curve.project(data)
print(projection.arc_lengths)   # distance along the curve for each point
print(projection.unit_lengths)  # normalised position in [0, 1]
print(projection.points)        # nearest point on the curve

# Reconstruct 100 evenly-spaced points along the curve
reconstructed = curve.interpolate_from_unit(np.linspace(0, 1, 100)).points

# Plotting
import matplotlib.pyplot as plt
plt.scatter(data[:, 0], data[:, 1], s=10, alpha=0.5, c="C0", label="Data")
plt.plot(reconstructed[:, 0], reconstructed[:, 1], alpha=0.5, c="C1", label="Curve")
plt.scatter(
    x=curve.control_points()[:, 0],
    y=curve.control_points()[:, 1],
    s=20,
    alpha=0.5,
    c="C1",
    label="Control Points",
)
plt.legend()
plt.show()
input("Press Enter to exit...")