## xstatstests

[![PyPI](https://img.shields.io/pypi/v/xstatstests)](https://pypi.org/project/xstatstests)
[![tests](https://github.com/dougiesquire/xstatstests/actions/workflows/tests.yml/badge.svg)](https://github.com/dougiesquire/xstatstests/actions/workflows/tests.yml)
[![pre-commit](https://github.com/dougiesquire/xstatstests/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/dougiesquire/xstatstests/actions/workflows/pre-commit.yml)
[![codecov](https://codecov.io/gh/dougiesquire/xstatstests/branch/main/graph/badge.svg?token=DBGC0FIRLA)](https://codecov.io/gh/dougiesquire/xstatstests)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/dougiesquire/xstatstests/blob/master/LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)

Statistical tests on xarray objects. Currently includes the following statistical tests (most are currently simple, often vectorized, wrappers of [`scipy.stats`](https://docs.scipy.org/doc/scipy/reference/stats.html) functions):

#### Distribution tests
- `xstatstests.ks_2samp_1d`: one-dimensional Kolmogorov-Smirnov test on two independent samples
- `xstatstests.ks_2samp_2d`: two-dimensional Kolmogorov-Smirnov test on two independent samples
- `xstatstests.anderson_ksamp`: Anderson-Darling test on K independent samples
- `xstatstests.cramervonmises_2samp` : Cramér-von Mises test on two independent samples
- `xstatstests.epps_singleton_2samp` : Epps-Singleton test on two independent samples
- `xstatstests.ansari` : Ansari-Bradley test for equal distribution scale parameters from two independent samples
- `xstatstests.mood` : Mood test for equal distribution scale parameters from two independent samples

#### Parameter tests
- `xstatstests.ttest_ind` : t-test for the means of two independent samples
- `xstatstests.ttest_rel` : t-test for the means of two related samples
- `xstatstests.bartlett` : Bartlett test for the variances of K independent samples
- `xstatstests.levene` : Levene test for the variances of K independent samples
- `xstatstests.fligner` : Fligner-Killeen test for the variances of K independent samples
- `xstatstests.median_test` : Mood test for the medians of K independent samples

#### Normality tests
- `xstatstests.skewtest` : D’Agostino test for normal sample skewness
- `xstatstests.kurtosistest` : Anscombe test for normal sample kurtosis
- `xstatstests.normaltest`: D’Agostino/Pearson test for normal sample skewness and kurtosis
- `xstatstests.jarque_bera` : Jarque-Bera test for normal sample skewness and kurtosis

#### Other tests
- `xstatstests.mannwhitneyu` : Mann-Whitney U rank test on two independent samples
- `xstatstests.ranksums` : Wilcoxon rank-sum statistic on two independent samples
- `xstatstests.kruskal` : Kruskal-Wallis H-test on K independent samples
- `xstatstests.friedmanchisquare` : Friedman chi-squared test on K repeated samples
- `xstatstests.brunnermunzel` : Brunner-Munzel test on two independent samples

This package was originally called `xks` but was renamed when tests additional to the KS test were added.

### Installation
To install this package from PyPI:
```
pip install xstatstests
```

### Contributing
Contributions are very welcome, particularly in the form of reporting bugs and writing tests. Please open an issue and check out the [contributor guide](CONTRIBUTING.md).

### References

Press, W.H. et al. 2007, Numerical Recipes, section 14.8

Peacock, J.A. 1983, Two-Dimensional Goodness-of-Fit Testing in Astronomy, Monthly Notices of the Royal Astronomical Society, vol. 202, pp. 615-627

Fasano, G. and Franceschini, A. 1987, A Multidimensional Version of the Kolmogorov-Smirnov Test, Monthly Notices of the Royal Astronomical Society, vol. 225, pp. 155-170
