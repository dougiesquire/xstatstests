import pytest

import numpy as np
import numpy.testing as npt

import xarray as xr

import scipy.stats

from xsampletests.core import scipy_function_info
import xsampletests as xst
from .fixtures import ds_1var


def check_vs_scipy_func(func, args, kwargs={}):
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
        outputs_np = [outputs["statistic"].values, outputs["pvalue"].values]

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
    _test_vs_scipy_values(args, outputs, function_info, kwargs=kwargs)

    # Test with multiple sample dims
    args_stack = [_stack_sample_dim(ds) for ds in args]
    outputs_stack = getattr(xst, func)(
        *args_stack, dim=["sample_1", "sample_2"], kwargs=kwargs
    )
    _test_vs_scipy_values(args, outputs_stack, function_info, kwargs=kwargs)


@pytest.mark.parametrize("ds1_n_per_sample", [10, 30])
@pytest.mark.parametrize("ds2_n_per_sample", [10, 20])
@pytest.mark.parametrize("shape", [(), (2,), (2, 3)])
@pytest.mark.parametrize("alternative", ["two-sided", "less", "greater"])
@pytest.mark.parametrize("method", ["auto", "exact", "asymp"])
def test_ks_2samp_1d_values(
    ds1_n_per_sample, ds2_n_per_sample, shape, alternative, method
):
    """Check ks_2samp_1d relative to scipy func"""
    args = [
        ds_1var((ds1_n_per_sample,) + shape, add_nans=False, dask=False),
        ds_1var((ds2_n_per_sample,) + shape, add_nans=False, dask=False),
    ]
    kwargs = dict(alternative=alternative, method=method)
    check_vs_scipy_func("ks_2samp_1d", args, kwargs)


@pytest.mark.parametrize("k_samples", [2, 3, 5])
@pytest.mark.parametrize(
    "n_per_sample", [[10, 10, 10, 10, 10], [10, 20, 30, 40, 50], [50, 40, 30, 20, 10]]
)
@pytest.mark.parametrize("shape", [(), (2,), (2, 3)])
@pytest.mark.parametrize("midrank", [True, False])
def test_anderson_ksamp_values(k_samples, n_per_sample, shape, midrank):
    """Check anderson_ksamp relative to scipy func"""
    args = [
        ds_1var((n,) + shape, add_nans=False, dask=False)
        for n in n_per_sample[slice(k_samples)]
    ]
    kwargs = dict(midrank=midrank)
    check_vs_scipy_func("anderson_ksamp", args, kwargs)


@pytest.mark.parametrize("ds1_n_per_sample", [10, 30])
@pytest.mark.parametrize("ds2_n_per_sample", [10, 20])
@pytest.mark.parametrize("shape", [(), (2,), (2, 3)])
@pytest.mark.parametrize("add_nans", [True, False])
@pytest.mark.parametrize("equal_var", [True, False])
@pytest.mark.parametrize("nan_policy", ["propagate", "omit"])
@pytest.mark.parametrize("permutations", [0, 1000])
@pytest.mark.parametrize("alternative", ["two-sided", "less", "greater"])
@pytest.mark.parametrize("trim", [0, 0.4])
def test_ttest_ind_values(
    ds1_n_per_sample,
    ds2_n_per_sample,
    shape,
    add_nans,
    equal_var,
    nan_policy,
    permutations,
    alternative,
    trim,
):
    """Check ttest_ind relative to scipy func"""
    if (nan_policy == "omit") & ((permutations != 0) | (trim != 0)):
        pytest.skip(
            "nan_policy='omit' is currently not supported by permutation tests or trimmed tests."
        )
    elif (permutations != 0) & (trim != 0):
        pytest.skip("Permutations are currently not supported with trimming.")
    else:
        args = [
            ds_1var((ds1_n_per_sample,) + shape, add_nans=add_nans, dask=False),
            ds_1var((ds2_n_per_sample,) + shape, add_nans=add_nans, dask=False),
        ]
        kwargs = dict(
            equal_var=equal_var,
            nan_policy=nan_policy,
            permutations=permutations,
            random_state=0,
            alternative=alternative,
            trim=trim,
        )
        check_vs_scipy_func("ttest_ind", args, kwargs)


@pytest.mark.parametrize("shape", [(), (2,), (2, 3)])
@pytest.mark.parametrize("add_nans", [True, False])
@pytest.mark.parametrize("nan_policy", ["propagate", "omit"])
@pytest.mark.parametrize("alternative", ["two-sided", "less", "greater"])
def test_ttest_rel_values(
    shape,
    add_nans,
    nan_policy,
    alternative,
):
    """Check ttest_rel relative to scipy func"""
    ds1_n_per_sample = ds2_n_per_sample = 10
    args = [
        ds_1var((ds1_n_per_sample,) + shape, add_nans=add_nans, dask=False),
        ds_1var((ds2_n_per_sample,) + shape, add_nans=add_nans, dask=False),
    ]
    kwargs = dict(
        nan_policy=nan_policy,
        alternative=alternative,
    )
    check_vs_scipy_func("ttest_rel", args, kwargs)


@pytest.mark.parametrize("ds1_n_per_sample", [2, 30])
@pytest.mark.parametrize("ds2_n_per_sample", [2, 20])
@pytest.mark.parametrize("shape", [(), (2,), (2, 3)])
@pytest.mark.parametrize("method", ["auto", "exact", "asymptotic"])
def test_cramervonmises_2samp_values(ds1_n_per_sample, ds2_n_per_sample, shape, method):
    """Check cramervonmises_2samp relative to scipy func"""
    if (method == "exact") & ((ds1_n_per_sample > 2) | (ds2_n_per_sample > 2)):
        pytest.skip("method == 'exact' is very slow for large samples.")
    args = [
        ds_1var((ds1_n_per_sample,) + shape, add_nans=False, dask=False),
        ds_1var((ds2_n_per_sample,) + shape, add_nans=False, dask=False),
    ]
    kwargs = dict(method=method)
    check_vs_scipy_func("cramervonmises_2samp", args, kwargs)


@pytest.mark.parametrize("ds1_n_per_sample", [10, 30])
@pytest.mark.parametrize("ds2_n_per_sample", [10, 20])
@pytest.mark.parametrize("shape", [(), (2,), (2, 3)])
@pytest.mark.parametrize("t", [(0.4, 0.8), (0.25, 0.5, 0.75)])
def test_epps_singleton_2samp_values(ds1_n_per_sample, ds2_n_per_sample, shape, t):
    """Check epps_singleton_2samp relative to scipy func"""
    args = [
        ds_1var((ds1_n_per_sample,) + shape, add_nans=False, dask=False),
        ds_1var((ds2_n_per_sample,) + shape, add_nans=False, dask=False),
    ]
    kwargs = dict(t=t)
    check_vs_scipy_func("epps_singleton_2samp", args, kwargs)


@pytest.mark.parametrize("ds1_n_per_sample", [10, 30])
@pytest.mark.parametrize("ds2_n_per_sample", [10, 20])
@pytest.mark.parametrize("shape", [(), (2,), (2, 3)])
@pytest.mark.parametrize("add_nans", [True, False])
@pytest.mark.parametrize("use_continuity", [True, False])
@pytest.mark.parametrize("alternative", ["two-sided", "less", "greater"])
@pytest.mark.parametrize("method", ["auto", "exact", "asymptotic"])
@pytest.mark.parametrize("nan_policy", ["propagate", "omit"])
def test_mannwhitneyu_values(
    ds1_n_per_sample,
    ds2_n_per_sample,
    shape,
    add_nans,
    use_continuity,
    alternative,
    method,
    nan_policy,
):
    """Check mannwhitneyu relative to scipy func"""
    args = [
        ds_1var((ds1_n_per_sample,) + shape, add_nans=add_nans, dask=False),
        ds_1var((ds2_n_per_sample,) + shape, add_nans=add_nans, dask=False),
    ]
    kwargs = dict(
        use_continuity=use_continuity,
        alternative=alternative,
        method=method,
        nan_policy=nan_policy,
    )
    check_vs_scipy_func("mannwhitneyu", args, kwargs)


@pytest.mark.parametrize("ds1_n_per_sample", [10, 30])
@pytest.mark.parametrize("ds2_n_per_sample", [10, 20])
@pytest.mark.parametrize("shape", [(), (2,), (2, 3)])
@pytest.mark.parametrize("add_nans", [True, False])
@pytest.mark.parametrize("alternative", ["two-sided", "less", "greater"])
@pytest.mark.parametrize("nan_policy", ["propagate", "omit"])
def test_ranksums_values(
    ds1_n_per_sample,
    ds2_n_per_sample,
    shape,
    add_nans,
    alternative,
    nan_policy,
):
    """Check ranksums relative to scipy func"""
    args = [
        ds_1var((ds1_n_per_sample,) + shape, add_nans=add_nans, dask=False),
        ds_1var((ds2_n_per_sample,) + shape, add_nans=add_nans, dask=False),
    ]
    kwargs = dict(
        alternative=alternative,
        nan_policy=nan_policy,
    )
    check_vs_scipy_func("ranksums", args, kwargs)


@pytest.mark.parametrize("k_samples", [2, 3, 5])
@pytest.mark.parametrize(
    "n_per_sample", [[10, 10, 10, 10, 10], [10, 20, 30, 40, 50], [50, 40, 30, 20, 10]]
)
@pytest.mark.parametrize("shape", [(), (2,), (2, 3)])
@pytest.mark.parametrize("add_nans", [True, False])
@pytest.mark.parametrize("nan_policy", ["propagate", "omit"])
def test_kruskal_values(
    k_samples,
    n_per_sample,
    shape,
    add_nans,
    nan_policy,
):
    """Check kruskal relative to scipy func"""
    args = [
        ds_1var((n,) + shape, add_nans=add_nans, dask=False)
        for n in n_per_sample[slice(k_samples)]
    ]
    kwargs = dict(
        nan_policy=nan_policy,
    )
    check_vs_scipy_func("kruskal", args, kwargs)


@pytest.mark.parametrize("samples", [10, 50])
@pytest.mark.parametrize("shape", [(), (2,), (2, 3)])
def test_ks_2samp_2d_identical(samples, shape):
    """Check that KS statistical is zero for identical arrays"""
    ds1_v1 = ds_1var((samples,) + shape).rename({"var": "var_1"})
    ds1_v2 = ds_1var((samples,) + shape).rename({"var": "var_2"})
    ds1 = xr.merge([ds1_v1, ds1_v2])
    ds2 = ds1.copy()
    D = xst.ks_2samp_2d(ds1, ds2, "sample")

    npt.assert_allclose(D["statistic"].values, 0.0)


@pytest.mark.parametrize("func", ["ttest_ind", "ttest_rel", "mannwhitneyu"])
@pytest.mark.parametrize("dask", [True, False])
def test_disallowed_error(func, dask):
    """Check that error is thrown when a disallowed kwarg is provided to scipy funcs"""
    n_per_sample = [10, 10]
    shape = (2, 3)
    args = [ds_1var((n,) + shape, dask) for n in n_per_sample]
    kws = scipy_function_info[func]["disallowed_kwargs"]
    for kw in kws:
        with pytest.raises(ValueError):
            getattr(xst, func)(*args, dim="sample", kwargs={kw: None})


@pytest.mark.parametrize(
    "func", ["ks_2samp_1d", "anderson_ksamp", "ttest_ind", "ttest_rel"]
)
def test_dask_compute(func):
    """Check that functions run with dask arrays and don't compute"""
    n_per_sample = [10, 10]
    shape = (2, 3)
    args = [ds_1var((n,) + shape, True) for n in n_per_sample]
    getattr(xst, func)(*args, dim="sample")


@pytest.mark.parametrize("func", ["ttest_rel"])
@pytest.mark.parametrize("dask", [True, False])
def test_sample_size_error(func, dask):
    """Check that error is thrown when sample sizes differ for functions that don't allow this"""
    n_per_sample = [10, 20]
    shape = (2, 3)
    args = [ds_1var((n,) + shape, dask) for n in n_per_sample]
    with pytest.raises(ValueError):
        getattr(xst, func)(*args, dim="sample")
