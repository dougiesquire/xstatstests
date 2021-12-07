import pytest

import numpy as np
import numpy.testing as npt

import xarray as xr

from xks import ks1d2s, ks2d2s
from .fixtures import ds_1var
from scipy.stats import ks_2samp


@pytest.mark.parametrize("ds1_samples", [1000])
@pytest.mark.parametrize("ds2_samples", [100, 1000])
@pytest.mark.parametrize("shape", [(), (2,), (2, 3)])
def test_ks1d2s(ds1_samples, ds2_samples, shape):
    """Test values of ks1d2s relative to scipy.stats.ks_2samp"""
    ds1 = ds_1var((ds1_samples,) + shape)
    ds2 = ds_1var((ds2_samples,) + shape)
    D_res, p_res = ks1d2s(ds1, ds2, sample_dim="sample")
    ds1_1d = np.reshape(ds1["var"].values, (ds1.sizes["sample"], -1))[:, 0]
    ds2_1d = np.reshape(ds2["var"].values, (ds2.sizes["sample"], -1))[:, 0]
    D_ver, p_ver = ks_2samp(ds1_1d, ds2_1d)

    npt.assert_allclose(D_res["var"].values, D_ver)
    npt.assert_allclose(p_res["var"].values, p_ver)


@pytest.mark.parametrize("samples", [100, 1000])
@pytest.mark.parametrize("shape", [(), (2,), (2, 3)])
def test_ks2d2s_identical(samples, shape):
    """Check that KS statistical is zero for identical arrays"""
    ds1_v1 = ds_1var((samples,) + shape).rename({"var": "var_1"})
    ds1_v2 = ds_1var((samples,) + shape).rename({"var": "var_2"})
    ds1 = xr.merge([ds1_v1, ds1_v2])
    ds2 = ds1.copy()
    D = ks2d2s(ds1, ds2, "sample")

    npt.assert_allclose(D.values, 0.0)
