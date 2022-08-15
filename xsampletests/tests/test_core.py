import pytest

import numpy as np
import numpy.testing as npt

import xarray as xr

import scipy.stats

from xsampletests.core import scipy_function_info
import xsampletests as xst
from .fixtures import ds_1var


def check_vs_scipy_func(func, args, dask, kwargs={}):
    """Test values relative to scipy function"""

    def _stack_sample_dim(ds):
        """Stack sample dim into two dimensions"""
        new_length = int(ds.sizes["sample"] / 2)
        return (
            ds.assign_coords(sample_1=range(2), sample_2=range(new_length))
            .stack(dim=["sample_1", "sample_2"])
            .reset_index("sample", drop=True)
            .rename(sample="dim")
            .unstack("dim")
        )

    def _test_vs_scipy_values(inputs, outputs, func_info, kwargs={}):
        """Test wrapped xsampletests func values relative to scipy"""
        scipy_func = getattr(scipy.stats, func_info["name"])

        inputs_np_1d = [
            np.reshape(inp["var"].values, (inp.sizes["sample"], -1))[:, 0]
            for inp in inputs
        ]
        outputs_np = [out["var"].values for out in outputs]

        if func_info["stack_args"]:
            scipy_outputs = scipy_func(inputs_np_1d, **kwargs)
        else:
            scipy_outputs = scipy_func(*inputs_np_1d, **kwargs)

        getter = func_info["outputs"]
        outputs_ver = [
            getattr(scipy_outputs, g) if isinstance(g, str) else scipy_outputs[g]
            for g in getter
        ]

        for res, ver in zip(outputs_np, outputs_ver):
            npt.assert_allclose(res, ver)

    function_info = scipy_function_info[func]

    # Test with a single sample dim

    outputs = getattr(xst, func)(*args, dim="sample", kwargs=kwargs)
    if dask is False:
        _test_vs_scipy_values(args, outputs, function_info, kwargs=kwargs)

    # Test with multiple sample dims
    args_stack = [_stack_sample_dim(ds) for ds in args]
    outputs_stack = getattr(xst, func)(*args_stack, dim=["sample_1", "sample_2"])
    if dask is False:
        _test_vs_scipy_values(args, outputs_stack, function_info)


@pytest.mark.parametrize("ds1_n_per_sample", [10, 30])
@pytest.mark.parametrize("ds2_n_per_sample", [10, 20])
@pytest.mark.parametrize("shape", [(), (2,), (2, 3)])
@pytest.mark.parametrize("dask", [True, False])
@pytest.mark.parametrize("alternative", ["two-sided", "less", "greater"])
@pytest.mark.parametrize("method", ["auto", "exact", "asymp"])
def test_ks_1d_2samp(
    ds1_n_per_sample, ds2_n_per_sample, shape, dask, alternative, method
):
    args = [
        ds_1var((ds1_n_per_sample,) + shape, dask),
        ds_1var((ds2_n_per_sample,) + shape, dask),
    ]
    kwargs = dict(alternative=alternative, method=method)
    check_vs_scipy_func("ks_1d_2samp", args, dask, kwargs)


@pytest.mark.parametrize("k_samples", [2, 3, 5])
@pytest.mark.parametrize(
    "n_per_sample", [[10, 10, 10, 10, 10], [10, 20, 30, 40, 50], [50, 40, 30, 20, 10]]
)
@pytest.mark.parametrize("shape", [(), (2,), (2, 3)])
@pytest.mark.parametrize("dask", [True, False])
@pytest.mark.parametrize("midrank", [True, False])
def test_ad_ksamp(k_samples, n_per_sample, shape, dask, midrank):
    args = [ds_1var((n,) + shape, dask) for n in n_per_sample[slice(k_samples)]]
    kwargs = dict(midrank=midrank)
    check_vs_scipy_func("ad_ksamp", args, dask, kwargs)


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
