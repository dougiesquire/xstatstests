import xarray as xr

SAMPLE_DIM = "xsampletest_sample_dim"


def _prep_data(*args, dim, nd):
    """Prepare data for tests"""
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
