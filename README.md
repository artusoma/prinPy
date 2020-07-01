# prinPy

Inspired by [this R package](https://github.com/rcannood/princurve), prinPy brings principal curves to Python. 

### What prinPy does
Currently, prinPy has implemented two local ("bottom-up") algorithms from [this paper](https://www.sciencedirect.com/science/article/pii/S0377042715005956). As of now, these only work in 2-dimensional space. 

1. CLPC-g (Greedy Constraint Local Principal Curve)
2. CLPC-s (One-Dimensional Search Constraint Local Principal Curve)

CLPC-g will be faster and is fine for simpler curves. CLPS-s has the potential to be much more accurate at the expense of speed for more difficult curves. After fitting a curve, prinPy has the ability to project to the curve.

### What is a Principal Curve?
A principal curve, simply put, is a smooth line that passes through the middle of a dataset. It then is a one-dimensional summary of a data.

## Quick-Start
View the quickstart notebook [here](). Docs will be coming soon!

## Future:
1. Add global algorithms, and expand to 3-dimensions+
2. Move some code to C++

## References
\[1\] Dewang Chen, Jiateng Yin, Shiying Yang, Lingxi Li, Peter Pudney,
Constraint local principal curve: Concept, algorithms and applications,
Journal of Computational and Applied Mathematics,
Volume 298,
2016,
Pages 222-235,
ISSN 0377-0427,
https://doi.org/10.1016/j.cam.2015.11.041.
