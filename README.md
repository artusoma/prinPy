[![Downloads](https://pepy.tech/badge/prinpy)](https://pepy.tech/project/prinpy)
# prinPy
`pip install prinpy`
<br>
<br>
Inspired by [this R package](https://github.com/rcannood/princurve), prinPy brings principal curves to Python. 

## prinPy version v1.0.0 is here! 🎉
**v1.0.0 introduces breaking changes and is not backwards-compatible.**
<br>
<br>
Key changes:
- Codebase has been refactored to be more modular and maintainable. 
- Key algorithms and functions are now implemented in Rust for speed: the CLPC-g algorithm now runs ~70x faster than the previous Python implementation.
- The API is more consistent and easier to use. All principal curve algorithms are produced by an `ICurveFitter`  and return a standard `ICurve` interface for projecting and interpolating.
- PyTorch is now used instead of Keras / TF, and is included with the optional `Neural` flag.
- The constained search algorithm has been replaced by a more efficient version that uses truncated SVD to find the optimal segment direction instead of searching over $\theta$.

## What prinPy does
PrinPy has local and global algorithms for computing principal curves. 

## What is a Principal Curve?
A principal curve is a smooth n-dimensional curve that passes through the middle of a dataset. Principal curves are a dimensionality reduction tool analogous to a nonlinear principal component. PCs have uses in GPS data, image recognition, bioinformatics, and so much more. 

### Local Algorithms
Local algorithms work on a step-by-step basis. Starting at one end of the curve, it will attempt to make segments that meet an acceptable error threshold as it moves from one end of the curve to the other. Once the algorithm can connect the current point to the end point, the algorithm terminates and a curve is interpolated through the segments. PrinPy currently has two local algorithms:

1. CLPC-g (Greedy Constraint Local Principal Curve)<sup>1</sup>
2. CLPC-s (One-Dimensional Search Constraint Local Principal Curve)<sup>1</sup>

CLPC-g will be faster and is fine for simpler curves. CLPS-s has the potential to be much more accurate at the expense of speed for more difficult curves. After fitting a curve, prinPy has the ability to project to the curve.

### Global Algorithms
Global algorithms, unlike local algorithms, are more like minimization problems. Given a dataset, a global algorithm might make an initial guess at a principal curve and adjust it from there. 

The sole global algorithm as of now performs nonlinear principal component analysis. The global algorithm, called NLPCA in this package, is a neural network implementation.<sup>2</sup> This algorithm works by creating an autoassociative neural network with a "bottle-neck" layer which forces the network to learn the most important features of the data. 

**Which one should I use?** <br>
The local algorithms will be better for tightly bunched data, such as digit recogniition or GPS data. The global algorithm is better suited for "clouds" of data or sparsely represented data.

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
