import inspect

import numpy as np
import xarray as xr
import scipy.stats

SAMPLE_DIM = "xsampletest_sample_dim"

# outputs[0] -> test statistic, outputs[1] -> p-value
# n_args = -1 means no limit to the number of args
scipy_function_info = {
    "ks_1d_2samp": {
        "name": "ks_2samp",
        "n_args": 2,
        "stack_args": False,
        "remove_nans": True,
        "outputs": [0, 1],
    },
    "ad_ksamp": {
        "name": "anderson_ksamp",
        "n_args": -1,
        "stack_args": True,
        "remove_nans": True,
        "outputs": ["statistic", "significance_level"],
    },
}


# 1-dimensional tests
# -------------------
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

    if isinstance(dim, str):
        dim = [dim]

    args = _prep_data_1d(*args, dim=dim)
    input_core_dims = [[f"{SAMPLE_DIM}{ind+1}"] for ind in range(len(args))]
    output_core_dims = [[]] * 2
    output_dtypes = ["float32"] * 2
    kwargs = dict(scipy_func_info=scipy_function_info[func], scipy_kwargs=kwargs)
    return xr.apply_ufunc(
        _wrap_scipy_func,
        *args,
        kwargs=kwargs,
        input_core_dims=input_core_dims,
        output_core_dims=output_core_dims,
        output_dtypes=output_dtypes,
        vectorize=True,
        dask="parallelized",
    )


def ks_1d_2samp(ds1, ds2, dim, kwargs={}):
    """One-dimensional Kolmogorov-Smirnov test on two samples, ds1 and ds2.

    Parameters
    ----------
    ds1 : xarray Dataset
        Sample 1 data
    ds2 : xarray Dataset
        Sample 2 data. Size of two samples can be different
    dim : str
        The name of the sample dimension(s) in ds1 and ds2
    kwargs : dict
        Any other kwargs to pass to scipy.stats.ks_2samp

    Returns
    -------
    statistic : xarray Dataset
        KS statistic estimating the max difference between the join distributions
    p-value : xarray Dataset
        The two-tailed p-value

    See also
    --------
    scipy.stats.ks_2samp

    References
    ----------
    Hodges, J.L. Jr., “The Significance Probability of the Smirnov Two-Sample Test,”
        Arkiv fiur Matematik, 3, No. 43 (1958), 469-86.
    """
    return _wrap_scipy(inspect.stack()[0][3], [ds1, ds2], dim, kwargs)


def ad_ksamp(*args, dim, kwargs={}):
    """Anderson-Darling test for k-samples.

    Parameters
    ----------
    args : xarray Datasets
        The k samples of data. Sizes of samples can be different
    dim : str
        The name of the sample dimension(s) in args
    kwargs : dict
        Any other kwargs to pass to scipy.stats.ad_ksamp

    Returns
    -------
    statistic : xarray Dataset
        Normalized k-sample Anderson-Darling test statistic.
    p-value : xarray Dataset
        An approximate significance level at which the null hypothesis for the provided samples
        can be rejected. The value is floored / capped at 0.1% / 25%.

    See also
    --------
    scipy.stats.anderson_ksamp
    """
    return _wrap_scipy(inspect.stack()[0][3], args, dim, kwargs)


# 2-dimensional tests
# -------------------
def ks_2d_2samp_np(x1, y1, x2, y2):
    """Two-dimensional Kolmogorov-Smirnov test on two samples. For now, returns only the KS statistic.
    Parameters
    ----------
    x1, y1 : ndarray, shape (..., n1)
        Data of sample 1, where n1 is the sample size. Dimensions preceding the last dimension are broadcast
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
        """Given an origin (x,y) and an array of NN points with coordinates XX and YY, count how may of them
        are in each quadrant around the origin, and return the normalised fractions.
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
        """Return the max distance ranging over data points and quadrants of the integrated probabilities"""
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
    # n1, n2 = x1.shape[-1], x2.shape[-1]
    D1 = _maxdist(x1, y1, x2, y2)
    D2 = _maxdist(x2, y2, x1, y1)
    return (D1 + D2) / 2


def ks_2d_2samp(ds1, ds2, dim):
    """xarray version of two-dimensional Kolmogorov-Smirnov test on two samples, ds1 and ds2.
    ds# should contain two variables corresponding to each dimension. For now, returns only the KS
    statistic with the expectation that confidence is assigned via resampling.

    Parameters
    ----------
    ds1 : xarray Dataset
        Sample 1 data
    ds2 : xarray Dataset
        Sample 2 data. Size of two samples can be different
    dim : str
        The name of the sample dimension in ds1 and ds2

    Returns
    -------
    D : xarray Dataset
        KS statistic estimating the max difference between the join distributions

    References
    ----------
    Press, W.H. et al. 2007, Numerical Recipes, section 14.8
    Peacock, J.A. 1983, Two-Dimensional Goodness-of-Fit Testing in Astronomy, Monthly Notices of
        the Royal Astronomical Society, vol. 202, pp. 615-627
    Fasano, G. and Franceschini, A. 1987, A Multidimensional Version of the Kolmogorov-Smirnov
        Test, Monthly Notices of the Royal Astronomical Society, vol. 225, pp. 155-170
    """
    if (not isinstance(ds1, xr.Dataset)) | (not isinstance(ds2, xr.Dataset)):
        raise TypeError("Input arrays must be xarray Datasets with 2 variables each")

    ds1, ds2 = xr.broadcast(ds1.copy(), ds2.copy(), exclude=[dim])
    ds1 = ds1.assign_coords({dim: range(len(ds1[dim]))})
    ds2 = ds2.assign_coords({dim: range(len(ds2[dim]))})

    ds1_vars = list(ds1.data_vars)
    ds2_vars = list(ds2.data_vars)
    assert len(ds1_vars) == 2
    assert ds1_vars == ds2_vars

    # Need to rename sample dim otherwise apply_ufunc tries to align
    ds1 = ds1.rename({dim: "s1"})
    ds2 = ds2.rename({dim: "s2"})

    return xr.apply_ufunc(
        ks_2d_2samp_np,
        ds1[ds1_vars[0]],
        ds1[ds1_vars[1]],
        ds2[ds1_vars[0]],
        ds2[ds1_vars[1]],
        input_core_dims=[["s1"], ["s1"], ["s2"], ["s2"]],
    )


def _prep_data_1d(*args, dim):
    """Prepare data for 1D tests"""
    if all([isinstance(ds, xr.Dataset) for ds in args]):
        args_vars = [list(ds.data_vars) for ds in args]
        assert all([args_vars[0] == ds_vars for ds_vars in args_vars[1:]])
    elif all([isinstance(ds, xr.DataArray) for ds in args]):
        if any([ds.name is not None for ds in args]):
            assert all(
                [args[0].name == ds.name for ds in args[1:]]
            ), "When named DataArrays are supplied, they must have the same name"
    else:
        raise TypeError("Input arrays must be all xarray Datasets or DataArrays")

    args = xr.broadcast(*[ds.copy() for ds in args], exclude=dim)

    if len(dim) == 1:
        args = [ds.rename({dim[0]: SAMPLE_DIM}) for ds in args]
    else:
        args = [ds.stack({SAMPLE_DIM: dim}) for ds in args]

    args = [ds.assign_coords({SAMPLE_DIM: range(ds.sizes[SAMPLE_DIM])}) for ds in args]

    # Need to rename sample dim otherwise apply_ufunc tries to align
    args = [
        ds.rename({SAMPLE_DIM: f"{SAMPLE_DIM}{ind+1}"}) for ind, ds in enumerate(args)
    ]
    return tuple(args)
