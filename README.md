[![Downloads](https://pepy.tech/badge/prinpy)](https://pepy.tech/project/prinpy)
[![PyPI](https://img.shields.io/pypi/v/prinpy)](https://pypi.org/project/prinpy/)
[![Python](https://img.shields.io/pypi/pyversions/prinpy)](https://pypi.org/project/prinpy/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

# prinPy

A high-performance Python library for fitting **principal curves** to n-dimensional data, with core algorithms implemented in Rust for speed.

Inspired by the [princurve R package](https://github.com/rcannood/princurve).

---

## Installation

```bash
pip install prinpy
```

For neural-network-based fitting (requires PyTorch):

```bash
pip install "prinpy[neural]"
```

**Requirements:** Python ≥ 3.9, NumPy ≥ 1.20

---

## Quick Start

```python
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
```

---

## What is a Principal Curve?

A principal curve is a smooth, one-dimensional manifold that passes through the middle of a dataset. It is the nonlinear generalisation of a principal component — instead of a straight line of best fit, it is a curve of best fit.

Principal curves are used in GPS track smoothing, bioinformatics, image processing, and anywhere a dataset has an intrinsic one-dimensional structure.

---

## Algorithms

| Class | Module | Strategy | Best for |
|---|---|---|---|
| `GreedyFit` | `prinpy.local_curves` | CLPC-g (greedy) | Fast fitting, simple or tightly-bunched curves |
| `SVDFit` | `prinpy.local_curves` | CLPC-s (truncated SVD) | Higher accuracy on complex curves |
| `NetworkFitter` | `prinpy.global_curves` | NLPCA (autoencoder) | Sparse data or diffuse point clouds |

All algorithms return a `PrincipalCurve` with the same interface — your downstream code never depends on which algorithm was used.

---

## Local Algorithms

Local algorithms grow the curve one segment at a time, marching from one end of the data to the other. They are fast and work well for tightly structured data.

Both are accessed through `ConstrainedFitter`, which wraps the chosen segment-finding strategy and fits a smooth spline through the resulting vertices.

```python
from prinpy.local_curves import ConstrainedFitter, GreedyFit, SVDFit

# Greedy — faster, good for most use cases
curve = ConstrainedFitter(algorithm=GreedyFit(inner_radius=0.9), tolerance=0.05).fit(data)

# SVD — more accurate for complex or curved shapes
curve = ConstrainedFitter(algorithm=SVDFit(), tolerance=0.05).fit(data)
```

**`tolerance`** controls the maximum allowed local fitting error per segment. Lower values produce more control points and a tighter fit; higher values produce a coarser, smoother curve.

---

## Global Algorithm (Neural Network)

The global algorithm fits an autoassociative neural network (NLPCA) whose bottleneck layer encodes the one-dimensional position along the curve. It is better suited to sparse or cloud-like data where local structure is not well-defined.

```python
from prinpy.global_curves import NetworkFitter, TrainingCallback

curve = NetworkFitter(
    dim=2,          # dimensionality of your data
    n_hidden=16,    # hidden layer size
    lr=0.01,        # learning rate
    epochs=500,
    callback=TrainingCallback(print_progress=True, every_n_epochs=50),
).fit(data)
```

Requires `pip install "prinpy[neural]"`.

---

## Working with a Fitted Curve

Every algorithm returns a `PrincipalCurve` with the same interface:

```python
# Total arc length of the curve
total_length = curve.length()

# Project arbitrary points onto the curve
proj = curve.project(new_data)
proj.points        # (n, d) — nearest points on the curve
proj.arc_lengths   # (n,)   — distance from the start of the curve
proj.unit_lengths  # (n,)   — normalised position in [0, 1]

# Interpolate from arc length
proj = curve.interpolate_from_length(np.array([0.0, 0.5, 1.2]))

# Interpolate from normalised position
proj = curve.interpolate_from_unit(np.linspace(0, 1, 200))

# Control points that define the curve's shape
pts = curve.control_points()  # (k, d)
```

---

## Development

prinPy uses [maturin](https://github.com/PyO3/maturin) to build the Rust extension.

```bash
# Clone and set up
git clone https://github.com/artusoma/prinpy
cd prinpy

# Install maturin and build the Rust extension in-place
pip install maturin
maturin develop

# Install Python dependencies (add [neural] for PyTorch support)
pip install -e ".[neural]"

# Run tests
python -m pytest tests/
```

---

## Migrating from v0.x

v1.0.0 is **not backwards-compatible**. Key changes:

- A standard `PrincipalCurve` / `CurveFitter` interface now exists — v0.x had no common API
- Core algorithms rewritten in Rust (~70× faster)
- PyTorch replaces Keras/TensorFlow for the neural fitter
- SVDFit replaces the old one-dimensional search algorithm
- All fitters now return a standard `PrincipalCurve` with a unified projection and interpolation API

---

## References

[1] Dewang Chen, Jiateng Yin, Shiying Yang, Lingxi Li, Peter Pudney,
*Constraint local principal curve: Concept, algorithms and applications*,
Journal of Computational and Applied Mathematics, Volume 298, 2016, Pages 222–235.
https://doi.org/10.1016/j.cam.2015.11.041

[2] Mark Kramer, *Nonlinear Principal Component Analysis Using Autoassociative Neural Networks*, AIChE Journal, 1991.

---

## License

MIT © [Matthew Artuso](https://github.com/artusoma). See [LICENSE](LICENSE) for details.
 
