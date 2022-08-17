import numpy as np
import xarray as xr

from .utils import _prep_data


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
