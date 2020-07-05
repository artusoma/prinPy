# prinPy
Installation: `pip install prinpy`

Inspired by [this R package](https://github.com/rcannood/princurve), prinPy brings principal curves to Python. 

### What prinPy does
PrinPy has two local and one global algorithm.

**Local**
1. CLPC-g (Greedy Constraint Local Principal Curve)<sup>1</sup>
2. CLPC-s (One-Dimensional Search Constraint Local Principal Curve)<sup>1</sup>

CLPC-g will be faster and is fine for simpler curves. CLPS-s has the potential to be much more accurate at the expense of speed for more difficult curves. After fitting a curve, prinPy has the ability to project to the curve.

**Global**
The sole global algorithm is not necessarily a principal curve, but a nonlinear principal component analysis. However, a PC and NLPCA end up doing the same thing. The global algorithm, called NLPCA, is a neural network implementation.<sup>2</sup>

**Which one should I use?**
The local algorithms will be better for tightly bunched data, such as digit recogniition or GPS data. The global algorithm is better suited for "clouds" of data or sparsely represented data.


### What is a Principal Curve?
A principal curve, simply put, is a smooth line that passes through the middle of a dataset.

## Quick-Start
View the quickstart notebook [here](https://github.com/artusoma/prinPy/blob/master/prinPy%20quickstart.ipynb). Docs will be coming soon!

```
cl = CLPCG() # Create solver

# CLPCG.fit() fits the principal curve. takes x_data, y_data,
# and the min allowed error for each step. e_min is acheived 
# through trial and error, but 1/4 to 1/2 data error is what authors
# recommend.
cl.fit(xdata, ydata, e_max = .1) 
cl.plot()       # plots curve, optional axes can be passed

# Reconstruct curve
tcks = cl.spline_ticks    # get spline ticks
xy = scipy.interpolate.splev(np.linspace(0,1,100), self.spline_ticks)
```

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

\[2\] Mark Kramer, Nonlinear Principal Component Analysis Using
Autoassociative Neural Networks 
