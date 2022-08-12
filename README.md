## xsampletests

[![tests](https://github.com/dougiesquire/xsampletests/actions/workflows/tests.yml/badge.svg)](https://github.com/dougiesquire/xsampletests/actions/workflows/tests.yml)
[![pre-commit](https://github.com/dougiesquire/xsampletests/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/dougiesquire/xsampletests/actions/workflows/pre-commit.yml)
[![codecov](https://codecov.io/gh/dougiesquire/xsampletests/branch/main/graph/badge.svg?token=DBGC0FIRLA)](https://codecov.io/gh/dougiesquire/xsampletests)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/dougiesquire/xsampletests/blob/master/LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)

Sample testing for xarray objects. Currently includes the following distribution tests:
- One-dimensional Kolmogorov-Smirnov test on two samples
- Two-dimensional Kolmogorov-Smirnov test on two samples
- Anderson-Darling test on N samples
This package was originally called `xks` but was renamed when tests other that the KS test were added.

### Installation
To install this package from PyPI:
```
#TODO
```

### References

Press, W.H. et al. 2007, Numerical Recipes, section 14.8

Peacock, J.A. 1983, Two-Dimensional Goodness-of-Fit Testing in Astronomy, Monthly Notices of the Royal Astronomical Society, vol. 202, pp. 615-627

Fasano, G. and Franceschini, A. 1987, A Multidimensional Version of the Kolmogorov-Smirnov Test, Monthly Notices of the Royal Astronomical Society, vol. 225, pp. 155-170
