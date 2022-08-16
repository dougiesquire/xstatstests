import inspect

import numpy as np
import xarray as xr
import scipy.stats

SAMPLE_DIM = "xsampletest_sample_dim"

# Summary of how to wrap scipy.stats funcs
# Enter outputs so that first output is the test statistic and the second
# output is the p-value
# n_args = -1 means no limit to the number of args
scipy_function_info = {
    "ks_2samp_1d": {
        "name": "ks_2samp",
        "stack_args": False,
        "same_sample_sizes": False,
        "remove_nans": True,
        "disallowed_kwargs": [],
        "outputs": [0, 1],
    },
    "anderson_ksamp": {
        "name": "anderson_ksamp",
        "stack_args": True,
        "same_sample_sizes": False,
        "remove_nans": True,
        "disallowed_kwargs": [],
        "outputs": ["statistic", "significance_level"],
    },
    "ttest_ind": {
        "name": "ttest_ind",
        "stack_args": False,
        "same_sample_sizes": False,
        "remove_nans": False,
        "disallowed_kwargs": ["axis"],
        "outputs": [0, 1],
    },
    "ttest_rel": {
        "name": "ttest_rel",
        "stack_args": False,
        "same_sample_sizes": True,
        "remove_nans": False,
        "disallowed_kwargs": ["axis"],
        "outputs": [0, 1],
    },
    "cramervonmises_2samp": {
        "name": "cramervonmises_2samp",
        "stack_args": False,
        "same_sample_sizes": False,
        "remove_nans": True,
        "disallowed_kwargs": [],
        "outputs": ["statistic", "pvalue"],
    },
    "epps_singleton_2samp": {
        "name": "epps_singleton_2samp",
        "stack_args": False,
        "same_sample_sizes": False,
        "remove_nans": True,
        "disallowed_kwargs": [],
        "outputs": [0, 1],
    },
    "mannwhitneyu": {
        "name": "mannwhitneyu",
        "stack_args": False,
        "same_sample_sizes": False,
        "remove_nans": False,
        "disallowed_kwargs": ["axis", "keepdims"],
        "outputs": ["statistic", "pvalue"],
    },
    "ranksums": {
        "name": "ranksums",
        "stack_args": False,
        "same_sample_sizes": False,
        "remove_nans": False,
        "disallowed_kwargs": ["axis", "keepdims"],
        "outputs": [0, 1],
    },
    "kruskal": {
        "name": "kruskal",
        "stack_args": False,
        "same_sample_sizes": False,
        "remove_nans": False,
        "disallowed_kwargs": ["axis", "keepdims"],
        "outputs": [0, 1],
    },
    "friedmanchisquare": {
        "name": "friedmanchisquare",
        "stack_args": False,
        "same_sample_sizes": True,
        "remove_nans": True,
        "disallowed_kwargs": [],
        "outputs": [0, 1],
    },
    "brunnermunzel": {
        "name": "brunnermunzel",
        "stack_args": False,
        "same_sample_sizes": False,
        "remove_nans": False,
        "disallowed_kwargs": [],
        "outputs": [0, 1],
    },
    "ansari": {
        "name": "ansari",
        "stack_args": False,
        "same_sample_sizes": False,
        "remove_nans": True,
        "disallowed_kwargs": [],
        "outputs": [0, 1],
    },
    "bartlett": {
        "name": "bartlett",
        "stack_args": False,
        "same_sample_sizes": False,
        "remove_nans": True,
        "disallowed_kwargs": [],
        "outputs": [0, 1],
    },
    "levene": {
        "name": "levene",
        "stack_args": False,
        "same_sample_sizes": False,
        "remove_nans": True,
        "disallowed_kwargs": [],
        "outputs": [0, 1],
    },
    "fligner": {
        "name": "fligner",
        "stack_args": False,
        "same_sample_sizes": False,
        "remove_nans": True,
        "disallowed_kwargs": [],
        "outputs": [0, 1],
    },
    "median_test": {
        "name": "median_test",
        "stack_args": False,
        "same_sample_sizes": False,
        "remove_nans": False,
        "disallowed_kwargs": [],
        "outputs": [0, 1],
    },
    "mood": {
        "name": "mood",
        "stack_args": False,
        "same_sample_sizes": False,
        "remove_nans": False,
        "disallowed_kwargs": ["axis"],
        "outputs": [0, 1],
    },
}


def _prep_data(*args, dim, nd):
    """Prepare data for 2D tests"""
    if isinstance(dim, str):
        dim = [dim]

    if any([(not isinstance(ds, xr.Dataset)) for ds in args]):
        raise TypeError(
            f"Input arrays must be xarray Datasets with {nd} variable(s) each"
        )

    args = xr.broadcast(*[ds.copy() for ds in args], exclude=dim)

    if len(dim) == 1:
        args = [ds.rename({dim[0]: SAMPLE_DIM}) for ds in args]
    else:
        args = [ds.stack({SAMPLE_DIM: dim}) for ds in args]

    args = [ds.drop_vars({SAMPLE_DIM, *dim}, errors="ignore") for ds in args]
    args = [ds.assign_coords({SAMPLE_DIM: range(ds.sizes[SAMPLE_DIM])}) for ds in args]

    assert all(
        [len(ds.data_vars) == nd for ds in args]
    ), f"Input Datasets must have {nd} variables each"
    assert all(
        [list(args[0].data_vars) == list(ds.data_vars) for ds in args]
    ), "Variables of all input Datasets must have the same name(s)"

    # Need to rename sample dim otherwise apply_ufunc tries to align
    # Expand into list of single variable Datasets
    args_prepped = []
    input_core_dims = []
    for ind, ds in enumerate(args):
        for var in ds.data_vars:
            sample_dim = f"{SAMPLE_DIM}{ind+1}"
            input_core_dims.append([sample_dim])
            args_prepped.append(ds[var].rename({SAMPLE_DIM: sample_dim}))

    return args_prepped, input_core_dims


def _wrap_scipy(func, args, dim, kwargs):
    """Generic xarray wrapper for subset of scipy stats functions"""

    def _wrap_scipy_func(*args, scipy_func_info, scipy_kwargs):
        """Parse scipy_function_info and apply scipy function"""
        func = getattr(scipy.stats, scipy_func_info["name"])
        getter = scipy_func_info["outputs"]

        if scipy_func_info["remove_nans"]:
            args = [arg[~np.isnan(arg)] for arg in args]
        if scipy_func_info["stack_args"]:
            outputs = func(args, **scipy_kwargs)
        else:
            outputs = func(*args, **scipy_kwargs)

        return tuple(
            [getattr(outputs, g) if isinstance(g, str) else outputs[g] for g in getter]
        )

    info = scipy_function_info[func]

    args, input_core_dims = _prep_data(*args, dim=dim, nd=1)

    sample_sizes = [ds.sizes[dim[0]] for ds, dim in zip(args, input_core_dims)]
    same_sample_sizes = all([sample_sizes[0] == samp for samp in sample_sizes[1:]])
    if info["same_sample_sizes"] & (not same_sample_sizes):
        raise ValueError(
            f"`{func}` requires that the sample size is the same for all input arrays"
        )

    scipy_kwargs = kwargs.copy()

    # Simply vectorize if function does not allow axis argument
    vectorize = True
    for kw in info["disallowed_kwargs"]:
        if kw in scipy_kwargs.keys():
            raise ValueError(f"`{kw}` kwarg is disallowed by xstatstests")
        if kw == "axis":
            vectorize = False
            scipy_kwargs["axis"] = -1

    output_core_dims = [[]] * 2
    output_dtypes = ["float32"] * 2
    kwargs = dict(scipy_func_info=info, scipy_kwargs=scipy_kwargs)
    statistic, pvalue = xr.apply_ufunc(
        _wrap_scipy_func,
        *args,
        kwargs=kwargs,
        input_core_dims=input_core_dims,
        output_core_dims=output_core_dims,
        output_dtypes=output_dtypes,
        vectorize=vectorize,
        dask="parallelized",
    )

    return xr.merge([statistic.rename("statistic"), pvalue.rename("pvalue")])


def ks_2samp_1d(ds1, ds2, dim, kwargs={}):
    """
    The one-dimensional Kolmogorov-Smirnov test on two samples.

    This test compares the underlying continuous distributions ds1 and ds2 of
    two independent samples.

    Parameters
    ----------
    ds1 : xarray Dataset
        Sample 1 data. Nans are automatically removed prior to executing the test
    ds2 : xarray Dataset
        Sample 2 data. Nans are automatically removed prior to executing the test.
        The sizes of samples 1 and 2 along dim can be different
    dim : str
        The name of the sample dimension(s) in ds1 and ds2
    kwargs : dict
        Any other kwargs to pass to scipy.stats.ks_2samp

    Returns
    -------
    statistics : xarray Dataset
        Dataset with the following variables:
        - "statistic" : The KS statistic
        - "pvalue" : One-tailed or two-tailed p-value

    Notes
    -----
    This function is a simple wrapper on the scipy function scipy.stats.ks_2samp.
    Users are recommended to read the scipy documentation prior to using this
    function.

    See also
    --------
    scipy.stats.ks_2samp

    References
    ----------
    Hodges, J.L. Jr., “The Significance Probability of the Smirnov Two-Sample Test,”
        Arkiv fiur Matematik, 3, No. 43 (1958), 469-86.
    """

    return _wrap_scipy(inspect.stack()[0][3], [ds1, ds2], dim, kwargs)


def anderson_ksamp(*args, dim, kwargs={}):
    """
    The Anderson-Darling test for k independent samples.

    The k-sample Anderson-Darling test is a modification of the one-sample
    Anderson-Darling test. It tests the null hypothesis that k-samples are drawn
    from the same population without having to specify the distribution function
    of that population.

    Parameters
    ----------
    args : xarray Datasets
        The k samples of data. Nans are automatically removed prior to executing the test.
        The sizes of the samples along dim can be different
    dim : str
        The name of the sample dimension(s) in args
    kwargs : dict
        Any other kwargs to pass to scipy.stats.ad_ksamp

    Returns
    -------
    statistics : xarray Dataset
        Dataset with the following variables:
        - "statistic" : Normalized k-sample Anderson-Darling test statistic
        - "pvalue" : An approximate significance level at which the null hypothesis
            for the provided samples can be rejected. The value is floored / capped
            at 0.1% / 25%

    Notes
    -----
    This function is a simple wrapper on the scipy function scipy.stats.anderson_ksamp.
    Users are recommended to read the scipy documentation prior to using this
    function.
    """

    return _wrap_scipy(inspect.stack()[0][3], args, dim, kwargs)


def ttest_ind(ds1, ds2, dim, kwargs={}):
    """
    The T-test for the means of two independent samples.

    This is a test for the null hypothesis that 2 independent samples have identical
    average (expected) values. This test assumes that the populations have identical
    variances by default.

    Parameters
    ----------
    ds1 : xarray Dataset
        Sample 1 data.
    ds2 : xarray Dataset
        Sample 2 data. The sizes of samples 1 and 2 along dim can be different
    dim : str
        The name of the sample dimension(s) in ds1 and ds2
    kwargs : dict
        Any other kwargs to pass to scipy.stats.ttest_ind

    Returns
    -------
    statistics : xarray Dataset
        Dataset with the following variables:
        - "statistic" : The t-statistic
        - "pvalue" : The p-value

    Notes
    -----
    This function is a simple wrapper on the scipy function scipy.stats.ttest_ind.
    Users are recommended to read the scipy documentation prior to using this
    function.
    """

    return _wrap_scipy(inspect.stack()[0][3], [ds1, ds2], dim, kwargs)


def ttest_rel(ds1, ds2, dim, kwargs={}):
    """
    The T-test for the means of two related samples.

    This is a test for the null hypothesis that two related or repeated samples
    have identical average (expected) values.

    Parameters
    ----------
    ds1 : xarray Dataset
        Sample 1 data.
    ds2 : xarray Dataset
        Sample 2 data. The sizes of samples 1 and 2 along dim must be the same
    dim : str
        The name of the sample dimension(s) in args
    kwargs : dict
        Any other kwargs to pass to scipy.stats.ttest_rel

    Returns
    -------
    statistics : xarray Dataset
        Dataset with the following variables:
        - "statistic" : The t-statistic
        - "pvalue" : The p-value

    Notes
    -----
    This function is a simple wrapper on the scipy function scipy.stats.ttest_rel.
    Users are recommended to read the scipy documentation prior to using this
    function.
    """

    return _wrap_scipy(inspect.stack()[0][3], [ds1, ds2], dim, kwargs)


def cramervonmises_2samp(ds1, ds2, dim, kwargs={}):
    """
    The two-sample Cramér-von Mises test for goodness of fit.

    This is the two-sample version of the Cramér-von Mises test: for two independent
    samples,the null hypothesis is that the samples come from the same (unspecified)
    continuous distribution.

    Parameters
    ----------
    ds1 : xarray Dataset
        Sample 1 data. Nans are automatically removed prior to executing the test
    ds2 : xarray Dataset
        Sample 2 data. Nans are automatically removed prior to executing the test.
        The sizes of samples 1 and 2 along dim can be different
    dim : str
        The name of the sample dimension(s) in ds1 and ds2
    kwargs : dict
        Any other kwargs to pass to scipy.stats.cramervonmises_2samp

    Returns
    -------
    statistics : xarray Dataset
        Dataset with the following variables:
        - "statistic" : The Cramér-von Mises statistic
        - "pvalue" : The p-value

    Notes
    -----
    This function is a simple wrapper on the scipy function scipy.stats.cramervonmises_2samp.
    Users are recommended to read the scipy documentation prior to using this
    function.
    """

    return _wrap_scipy(inspect.stack()[0][3], [ds1, ds2], dim, kwargs)


def epps_singleton_2samp(ds1, ds2, dim, kwargs={}):
    """
    The Epps-Singleton (ES) test on two samples.

    This tests the null hypothesis that two samples have the same underlying probability
    distribution.

    Parameters
    ----------
    ds1 : xarray Dataset
        Sample 1 data. Nans are automatically removed prior to executing the test
    ds2 : xarray Dataset
        Sample 2 data. Nans are automatically removed prior to executing the test.
        The sizes of samples 1 and 2 along dim can be different
    dim : str
        The name of the sample dimension(s) in ds1 and ds2
    kwargs : dict
        Any other kwargs to pass to scipy.stats.epps_singleton_2samp

    Returns
    -------
    statistics : xarray Dataset
        Dataset with the following variables:
        - "statistic" : The Epps-Singleton statistic
        - "pvalue" : The p-value based on the asymptotic chi2-distribution

    Notes
    -----
    This function is a simple wrapper on the scipy function scipy.stats.epps_singleton_2samp.
    Users are recommended to read the scipy documentation prior to using this
    function.
    """

    return _wrap_scipy(inspect.stack()[0][3], [ds1, ds2], dim, kwargs)


def mannwhitneyu(ds1, ds2, dim, kwargs={}):
    """
    The Mann-Whitney U rank test on two independent samples.

    The Mann-Whitney U test is a nonparametric test of the null hypothesis that the
    distribution underlying sample ds1 is the same as the distribution underlying
    sample ds2. It is often used as a test of difference in location between distributions.

    Parameters
    ----------
    ds1 : xarray Dataset
        Sample 1 data.
    ds2 : xarray Dataset
        Sample 2 data. The sizes of samples 1 and 2 along dim can be different
    dim : str
        The name of the sample dimension(s) in ds1 and ds2
    kwargs : dict
        Any other kwargs to pass to scipy.stats.mannwhitneyu

    Returns
    -------
    statistics : xarray Dataset
        Dataset with the following variables:
        - "statistic" : The Mann-Whitney U statistic corresponding with sample ds1. See the
            scipy.stats.epps_singleton_2samp documentation for how to calculate the U
            statistic corresponding with sample ds2
        - "pvalue" : The p-value

    Notes
    -----
    This function is a simple wrapper on the scipy function scipy.stats.mannwhitneyu.
    Users are recommended to read the scipy documentation prior to using this
    function.
    """

    return _wrap_scipy(inspect.stack()[0][3], [ds1, ds2], dim, kwargs)


def ranksums(ds1, ds2, dim, kwargs={}):
    """
    The Wilcoxon rank-sum statistic for two independent samples.

    The Wilcoxon rank-sum test tests the null hypothesis that two sets of measurements are drawn
    from the same distribution. The alternative hypothesis is that values in one sample are more
    likely to be larger than the values in the other sample. This test should be used to compare
    two samples from continuous distributions. It does not handle ties between ds1 and ds2.

    Parameters
    ----------
    ds1 : xarray Dataset
        Sample 1 data.
    ds2 : xarray Dataset
        Sample 2 data. The sizes of samples 1 and 2 along dim can be different
    dim : str
        The name of the sample dimension(s) in ds1 and ds2
    kwargs : dict
        Any other kwargs to pass to scipy.stats.ranksums

    Returns
    -------
    statistics : xarray Dataset
        Dataset with the following variables:
        - "statistic" : The Wilcoxon rank-sum statistic under the large-sample approximation that
            the rank sum statistic is normally distributed.
        - "pvalue" : The p-value

    Notes
    -----
    This function is a simple wrapper on the scipy function scipy.stats.ranksums.
    Users are recommended to read the scipy documentation prior to using this
    function.
    """

    return _wrap_scipy(inspect.stack()[0][3], [ds1, ds2], dim, kwargs)


def kruskal(*args, dim, kwargs={}):
    """
    The Kruskal-Wallis H-test for k independent samples.

    The Kruskal-Wallis H-test tests the null hypothesis that the population median of all of the
    groups are equal. It is a non-parametric version of ANOVA.

    Parameters
    ----------
    args : xarray Datasets
        The k samples of data. The sizes of the samples along dim can be different
    dim : str
        The name of the sample dimension(s) in args
    kwargs : dict
        Any other kwargs to pass to scipy.stats.kruskal

    Returns
    -------
    statistics : xarray Dataset
        Dataset with the following variables:
        - "statistic" : The Kruskal-Wallis H statistic, corrected for ties.
        - "pvalue" : The p-value for the test using the assumption that H has a chi square
            distribution. The p-value returned is the survival function of the chi square
            distribution evaluated at H.

    Notes
    -----
    This function is a simple wrapper on the scipy function scipy.stats.kruskal.
    Users are recommended to read the scipy documentation prior to using this
    function.
    """

    return _wrap_scipy(inspect.stack()[0][3], args, dim, kwargs)


def friedmanchisquare(*args, dim, kwargs={}):
    """
    The Friedman test for k repeated samples.

    The Friedman test tests the null hypothesis that repeated samples of the same
    individuals have the same distribution. It is often used to test for consistency
    among samples obtained in different ways. For example, if two sampling techniques
    are used on the same set of individuals, the Friedman test can be used to determine
    if the two sampling techniques are consistent.

    Parameters
    ----------
    args : xarray Datasets
        The k samples of data. Nans are automatically removed prior to executing the test.
        The sizes of the samples along dim can be different
    dim : str
        The name of the sample dimension(s) in args
    kwargs : dict
        Any other kwargs to pass to scipy.stats.friedmanchisquare

    Returns
    -------
    statistics : xarray Dataset
        Dataset with the following variables:
        - "statistic" : The test statistic, correcting for ties.
        - "pvalue" : The p-value assuming that the test statistic has a chi squared
            distribution.

    Notes
    -----
    This function is a simple wrapper on the scipy function scipy.stats.kruskal.
    Users are recommended to read the scipy documentation prior to using this
    function.
    """

    return _wrap_scipy(inspect.stack()[0][3], args, dim, kwargs)


def brunnermunzel(ds1, ds2, dim, kwargs={}):
    """
    The Brunner-Munzel test on two independent samples.

    The Brunner-Munzel test is a nonparametric test of the null hypothesis that when values
    are taken one by one from each group, the probabilities of getting large values in both
    groups are equal. Unlike the Wilcoxon-Mann-Whitney’s U test, this does not require the
    assumption of equivariance of two groups. Note that this does not assume the
    distributions are same.

    Parameters
    ----------
    ds1 : xarray Dataset
        Sample 1 data.
    ds2 : xarray Dataset
        Sample 2 data. The sizes of samples 1 and 2 along dim can be different
    dim : str
        The name of the sample dimension(s) in ds1 and ds2
    kwargs : dict
        Any other kwargs to pass to scipy.stats.brunnermunzel

    Returns
    -------
    statistics : xarray Dataset
        Dataset with the following variables:
        - "statistic" : The Brunner-Munzer W statistic.
        - "pvalue" : The one-sided or two-sided p-value assuming an t distribution.

    Notes
    -----
    This function is a simple wrapper on the scipy function scipy.stats.brunnermunzel.
    Users are recommended to read the scipy documentation prior to using this
    function.
    """

    return _wrap_scipy(inspect.stack()[0][3], [ds1, ds2], dim, kwargs)


def ansari(ds1, ds2, dim, kwargs={}):
    """
    The Ansari-Bradley test for equal distribution scale parameters from two independent samples.

    The Ansari-Bradley test is a non-parametric test for the equality of the scale parameter
    of the distributions from which two samples were drawn. The null hypothesis states that
    the ratio of the scale of the distribution underlying ds1 to the scale of the distribution
    underlying ds2 is 1.

    Parameters
    ----------
    ds1 : xarray Dataset
        Sample 1 data. Nans are automatically removed prior to executing the test
    ds2 : xarray Dataset
        Sample 2 data. Nans are automatically removed prior to executing the test.
        The sizes of samples 1 and 2 along dim can be different
    dim : str
        The name of the sample dimension(s) in ds1 and ds2
    kwargs : dict
        Any other kwargs to pass to scipy.stats.ansari

    Returns
    -------
    statistics : xarray Dataset
        Dataset with the following variables:
        - "statistic" : The Ansari-Bradley test statistic
        - "pvalue" : The p-value

    Notes
    -----
    This function is a simple wrapper on the scipy function scipy.stats.ansari.
    Users are recommended to read the scipy documentation prior to using this
    function.
    """

    return _wrap_scipy(inspect.stack()[0][3], [ds1, ds2], dim, kwargs)


def bartlett(*args, dim, kwargs={}):
    """
    The Bartlett test for the variances of k independent samples.

    Bartlett’s test tests the null hypothesis that all input samples are from populations
    with equal variances. For samples from significantly non-normal populations, Levene’s
    test levene is more robust.

    Parameters
    ----------
    args : xarray Datasets
        The k samples of data. Nans are automatically removed prior to executing the test.
        The sizes of the samples along dim can be different
    dim : str
        The name of the sample dimension(s) in args
    kwargs : dict
        Any other kwargs to pass to scipy.stats.bartlett

    Returns
    -------
    statistics : xarray Dataset
        Dataset with the following variables:
        - "statistic" : The Bartlett test statistic
        - "pvalue" : The p-value

    Notes
    -----
    This function is a simple wrapper on the scipy function scipy.stats.bartlett.
    Users are recommended to read the scipy documentation prior to using this
    function.
    """

    return _wrap_scipy(inspect.stack()[0][3], args, dim, kwargs)


def levene(*args, dim, kwargs={}):
    """
    The Levene test for the variances of k independent samples.

    The Levene test tests the null hypothesis that all input samples are from populations
    with equal variances. Levene’s test is an alternative to Bartlett’s test in the case
    where there are significant deviations from normality.

    Parameters
    ----------
    args : xarray Datasets
        The k samples of data. Nans are automatically removed prior to executing the test.
        The sizes of the samples along dim can be different
    dim : str
        The name of the sample dimension(s) in args
    kwargs : dict
        Any other kwargs to pass to scipy.stats.levene

    Returns
    -------
    statistics : xarray Dataset
        Dataset with the following variables:
        - "statistic" : The Levene test statistic
        - "pvalue" : The p-value

    Notes
    -----
    This function is a simple wrapper on the scipy function scipy.stats.levene.
    Users are recommended to read the scipy documentation prior to using this
    function.
    """

    return _wrap_scipy(inspect.stack()[0][3], args, dim, kwargs)


def fligner(*args, dim, kwargs={}):
    """
    The Fligner-Killeen test for the variances of k independent samples.

    Fligner’s test tests the null hypothesis that all input samples are from populations
    with equal variances. Fligner-Killeen’s test is distribution free when populations are
    identical.

    Parameters
    ----------
    args : xarray Datasets
        The k samples of data. Nans are automatically removed prior to executing the test.
        The sizes of the samples along dim can be different
    dim : str
        The name of the sample dimension(s) in args
    kwargs : dict
        Any other kwargs to pass to scipy.stats.fligner

    Returns
    -------
    statistics : xarray Dataset
        Dataset with the following variables:
        - "statistic" : The Fligner-Killeen test statistic
        - "pvalue" : The p-value

    Notes
    -----
    This function is a simple wrapper on the scipy function scipy.stats.fligner.
    Users are recommended to read the scipy documentation prior to using this
    function.
    """

    return _wrap_scipy(inspect.stack()[0][3], args, dim, kwargs)


def median_test(*args, dim, kwargs={}):
    """
    The Mood test for the medians of k independent samples.

    Mood's test tests that two or more samples come from populations with the same median.

    Parameters
    ----------
    args : xarray Datasets
        The k samples of data. The sizes of the samples along dim can be different
    dim : str
        The name of the sample dimension(s) in args
    kwargs : dict
        Any other kwargs to pass to scipy.stats.median_test

    Returns
    -------
    statistics : xarray Dataset
        Dataset with the following variables:
        - "statistic" : The test statistic
        - "pvalue" : The p-value

    Notes
    -----
    This function is a simple wrapper on the scipy function scipy.stats.median_test.
    Users are recommended to read the scipy documentation prior to using this
    function.
    """

    return _wrap_scipy(inspect.stack()[0][3], args, dim, kwargs)


def mood(ds1, ds2, dim, kwargs={}):
    """
    The Mood test for equal distribution scale parameters from two independent samples.

    Mood’s two-sample test for scale parameters is a non-parametric test for the null
    hypothesis that two samples are drawn from the same distribution with the same scale
    parameter.

    Parameters
    ----------
    ds1 : xarray Dataset
        Sample 1 data.
    ds2 : xarray Dataset
        Sample 2 data. The sizes of samples 1 and 2 along dim can be different
    dim : str
        The name of the sample dimension(s) in ds1 and ds2
    kwargs : dict
        Any other kwargs to pass to scipy.stats.mood

    Returns
    -------
    statistics : xarray Dataset
        Dataset with the following variables:
        - "statistic" : The z-score for the hypothesis test
        - "pvalue" : The p-value

    Notes
    -----
    This function is a simple wrapper on the scipy function scipy.stats.mood.
    Users are recommended to read the scipy documentation prior to using this
    function.
    """

    return _wrap_scipy(inspect.stack()[0][3], [ds1, ds2], dim, kwargs)


def ks_2samp_2d_np(x1, y1, x2, y2):
    """
    Two-dimensional Kolmogorov-Smirnov test on two samples. For now, returns only
    the KS statistic.

    Parameters
    ----------
    x1, y1 : ndarray, shape (..., n1)
        Data of sample 1, where n1 is the sample size. Dimensions preceding the last
        dimension are broadcast
    x2, y2 : ndarray, shape (..., n2)
        Data of sample 2, where n2 is the sample size. Size of two samples can be different.

    Returns
    -------
    D : float, optional
        KS statistic estimating the max difference between the join distributions

    References
    ----------
    Press, W.H. et al. 2007, Numerical Recipes, section 14.8
    Peacock, J.A. 1983, Two-Dimensional Goodness-of-Fit Testing in Astronomy, Monthly Notices of
        the Royal Astronomical Society, vol. 202, pp. 615-627
    Fasano, G. and Franceschini, A. 1987, A Multidimensional Version of the Kolmogorov-Smirnov
        Test, Monthly Notices of the Royal Astronomical Society, vol. 225, pp. 155-170
    """

    def _quadct(x, y, xx, yy):
        """
        Given an origin (x,y) and an array of NN points with coordinates XX and YY, count
        how many of them are in each quadrant around the origin, and return the normalised
        fractions.
        """
        n = xx.shape[-1]
        ix1, ix2 = xx > x, yy > y
        a = np.sum(ix1 & ix2, axis=-1) / n
        b = np.sum(~ix1 & ix2, axis=-1) / n
        c = np.sum(~ix1 & ~ix2, axis=-1) / n
        d = np.sum(ix1 & ~ix2, axis=-1) / n
        np.testing.assert_almost_equal(1, a + b + c + d)
        return a, b, c, d

    def _maxdist(x1, y1, x2, y2):
        """
        Return the max distance ranging over data points and quadrants of the integrated
        probabilities
        """
        n1 = x1.shape[-1]
        D = np.empty((*x1.shape[:-1], 4, x1.shape[-1]))
        for i in range(n1):
            a1, b1, c1, d1 = _quadct(
                np.expand_dims(x1[..., i], axis=-1),
                np.expand_dims(y1[..., i], axis=-1),
                x1,
                y1,
            )
            a2, b2, c2, d2 = _quadct(
                np.expand_dims(x1[..., i], axis=-1),
                np.expand_dims(y1[..., i], axis=-1),
                x2,
                y2,
            )
            D[..., :, i] = np.stack(
                [a1 - a2, b1 - b2, c1 - c2, d1 - d2], axis=-1
            )  # differences in each quadrant

        # re-assign the point to maximize difference,
        # the discrepancy is significant for N < ~50
        #     D[:, 0] -= 1 / n1
        #     dmin, dmax = -D.min(), D.max() + 1 / n1
        #     return max(dmin, dmax)

        return np.max(abs(D), axis=(-2, -1))  # Find max over all points and quadrants

    # Remove any nans along the sample dimension that were added by broadcasting sample_1 and sample_2
    x1 = x1[
        ..., ~np.apply_over_axes(np.all, np.isnan(x1), range(x1.ndim - 1)).squeeze()
    ]
    y1 = y1[
        ..., ~np.apply_over_axes(np.all, np.isnan(y1), range(y1.ndim - 1)).squeeze()
    ]
    x2 = x2[
        ..., ~np.apply_over_axes(np.all, np.isnan(x2), range(x2.ndim - 1)).squeeze()
    ]
    y2 = y2[
        ..., ~np.apply_over_axes(np.all, np.isnan(y2), range(y2.ndim - 1)).squeeze()
    ]
    assert (x1.shape[-1] == y1.shape[-1]) and (x2.shape[-1] == y2.shape[-1])
    assert (x1.shape[:-1] == x2.shape[:-1]) and (y1.shape[:-1] == y2.shape[:-1])

    D1 = _maxdist(x1, y1, x2, y2)
    D2 = _maxdist(x2, y2, x1, y1)
    return (D1 + D2) / 2


def ks_2samp_2d(ds1, ds2, dim):
    """
    Two-dimensional Kolmogorov-Smirnov test on two samples. For now, returns only
    the KS statistic with the expectation that confidence is assigned via resampling.

    This test compares the underlying continuous distributions ds1 and ds2 of two
    two-dimensional independent samples.

    Parameters
    ----------
    ds1 : xarray Dataset
        Sample 1 data containing two variables
    ds2 : xarray Dataset
        Sample 2 data containing two variables. The sizes of samples 1 and 2 along dim
        can be different
    dim : str
        The name of the sample dimension(s) in ds1 and ds2

    Returns
    -------
    statistic : xarray Dataset
        KS statistic estimating the max difference between the joint distributions

    References
    ----------
    Press, W.H. et al. 2007, Numerical Recipes, section 14.8
    Peacock, J.A. 1983, Two-Dimensional Goodness-of-Fit Testing in Astronomy, Monthly Notices of
        the Royal Astronomical Society, vol. 202, pp. 615-627
    Fasano, G. and Franceschini, A. 1987, A Multidimensional Version of the Kolmogorov-Smirnov
        Test, Monthly Notices of the Royal Astronomical Society, vol. 225, pp. 155-170
    """

    args, input_core_dims = _prep_data(ds1, ds2, dim=dim, nd=2)

    res = xr.apply_ufunc(
        ks_2samp_2d_np,
        *args,
        input_core_dims=input_core_dims,
        output_core_dims=[[]],
        output_dtypes=["float32"],
        dask="parallelized",
    )

    return res.to_dataset(name="statistic")
