# prinPy

Inspired by [this R package](https://github.com/rcannood/princurve), prinPy is an implementation of principal curves in Python, based rougly on [this](https://web.stanford.edu/~hastie/Papers/Principal_Curves.pdf) paper and [this](http://www.lpsm.paris/pageperso/biau/BIAU/bf.pdf) paper. 

### Features:
- 2-Dimensional curve fitting
- Projecting points to curve

### Future:
1. Currently, only simple curves are supported. For example, x must be strictly increasing. I would like to change this eventually.
2. Move to C++. Right now it is completely written in Python with a lot of support from sciPy. I hope to use C++ to speed up calculations.
