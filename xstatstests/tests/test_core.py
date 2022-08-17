import pytest

import numpy.testing as npt

import xarray as xr

import xstatstests as xst
from .fixtures import ds_1var


@pytest.mark.parametrize("samples", [10, 50])
@pytest.mark.parametrize("shape", [(), (2,), (2, 3)])
@pytest.mark.parametrize("dask", [True, False])
def test_ks_2samp_2d_identical(samples, shape, dask):
    """Check that KS statistical is zero for identical arrays"""
    ds1_v1 = ds_1var((samples,) + shape, dask=dask).rename({"var": "var_1"})
    ds1_v2 = ds_1var((samples,) + shape, dask=dask).rename({"var": "var_2"})
    ds1 = xr.merge([ds1_v1, ds1_v2])
    ds2 = ds1.copy()
    D = xst.ks_2samp_2d(ds1, ds2, "sample")

    if dask is False:
        npt.assert_allclose(D["statistic"].values, 0.0)
