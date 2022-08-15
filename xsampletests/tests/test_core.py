import pytest

import numpy as np
import numpy.testing as npt

import xarray as xr

import scipy.stats

import xsampletests as xst
from xsampletests.core import scipy_function_info
from .fixtures import ds_1var


def _test_vs_scipy_values(inputs, outputs, func_info, kwargs={}):
    """Test wrapped xsampletests func values relative to scipy"""
    func = getattr(scipy.stats, func_info["name"])
    getter = func_info["outputs"]

    inputs_np_1d = [
        np.reshape(inp["var"].values, (inp.sizes["sample"], -1))[:, 0] for inp in inputs
    ]
    outputs_np = [out["var"].values for out in outputs]

    if func_info["stack_args"]:
        scipy_outputs = func(inputs_np_1d, **kwargs)
    else:
        scipy_outputs = func(*inputs_np_1d, **kwargs)

    outputs_ver = [
        getattr(scipy_outputs, g) if isinstance(g, str) else scipy_outputs[g]
        for g in getter
    ]

    for res, ver in zip(outputs_np, outputs_ver):
        npt.assert_allclose(res, ver)


@pytest.mark.parametrize("func", ["ks_1d_2samp", "ad_ksamp"])
@pytest.mark.parametrize("k_samples", [2, 3, 5])
@pytest.mark.parametrize(
    "n_per_sample", [[10, 10, 10, 10, 10], [10, 20, 30, 40, 50], [50, 40, 30, 20, 10]]
)
@pytest.mark.parametrize("shape", [(), (2,), (2, 3)])
@pytest.mark.parametrize("dask", [True, False])
def test_scipy_func(func, k_samples, n_per_sample, shape, dask):
    """Test values relative to scipy function"""

    def _stack_sample_dim(ds):
        """Stack sample dim into two dimensions"""
        return (
            ds.assign_coords(
                sample_1=range(2), sample_2=range(int(ds.sizes["sample"] / 2))
            )
            .stack(dim=["sample_1", "sample_2"])
            .reset_index("sample", drop=True)
            .rename(sample="dim")
            .unstack("dim")
        )

    function_info = scipy_function_info[func]

    if (function_info["n_args"] != -1) & (k_samples > function_info["n_args"]):
        pass
    else:
        # Test with a single sample dim
        dss = [ds_1var((n,) + shape, dask) for n in n_per_sample[slice(k_samples)]]
        outputs = getattr(xst, func)(*dss, dim="sample")
        if dask is False:
            _test_vs_scipy_values(dss, outputs, function_info)

        # Test with multiple sample dims
        dss_stack = [_stack_sample_dim(ds) for ds in dss]
        outputs_stack = getattr(xst, func)(*dss_stack, dim=["sample_1", "sample_2"])
        if dask is False:
            _test_vs_scipy_values(dss, outputs_stack, function_info)


@pytest.mark.parametrize("samples", [100, 1000])
@pytest.mark.parametrize("shape", [(), (2,), (2, 3)])
def test_ks_2d_2samp_identical(samples, shape):
    """Check that KS statistical is zero for identical arrays"""
    ds1_v1 = ds_1var((samples,) + shape).rename({"var": "var_1"})
    ds1_v2 = ds_1var((samples,) + shape).rename({"var": "var_2"})
    ds1 = xr.merge([ds1_v1, ds1_v2])
    ds2 = ds1.copy()
    D = xst.ks_2d_2samp(ds1, ds2, "sample")

    npt.assert_allclose(D.values, 0.0)
