# prinPy

Inspired by [this R package](https://github.com/rcannood/princurve), prinPy brings principal curves to Python. 

### What prinPy does
Currently, prinPy has implemented two algorithms from [this paper](https://www.sciencedirect.com/science/article/pii/S0377042715005956). As of now, these only work in 2-dimensional space. 

1. CLPC-g (Greedy Constraint Local Principal Curve)
2. CLPC-s (One-Dimensional Constraint Local Principal Curve)

CLPC-g, as the name implies, is a greedy algorithm and will be faster. This is fine for simpler curves. CLPS-s has the potential to be much more accurate at the expense of speed for more difficult curves. After fitting a curve, prinPy has the ability to project to the curve.

### Quick-Start
View the quickstart notebook [here](). Docs will be coming soon!

### Future:
1. Add more PC algorithms, and expand to 3-dimensions+
2. Move some code to C++

### Referebces
\[1\] Dewang Chen, Jiateng Yin, Shiying Yang, Lingxi Li, Peter Pudney,
Constraint local principal curve: Concept, algorithms and applications,
Journal of Computational and Applied Mathematics,
Volume 298,
2016,
Pages 222-235,
ISSN 0377-0427,
https://doi.org/10.1016/j.cam.2015.11.041.
