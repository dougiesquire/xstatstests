import numpy as np
import xarray as xr

import dask
import dask.array as dsa


def empty_dask_array(shape, dtype=float, chunks=None):
    """A dask array that errors if you try to compute it
    Stolen from https://github.com/xgcm/xhistogram/blob/master/xhistogram/test/fixtures.py
    """

    def raise_if_computed():
        raise ValueError("Triggered forbidden computation on dask array")

    a = dsa.from_delayed(dask.delayed(raise_if_computed)(), shape, dtype)
    if chunks is not None:
        a = a.rechunk(chunks)
    return a


def ds_1var(shape, dask=False):
    """An example DataSet.
    The first dimension is named 'sample' and the data is identical over other dimensions
    """
    coords = [range(s) for s in shape]
    dims = ["sample"] + [f"dim_{i}" for i in range(1, len(shape))]
    if dask:
        data = empty_dask_array(shape)
    else:
        data = np.expand_dims(
            np.random.random(size=shape[0]), list(range(1, len(shape)))
        )
        data = np.tile(data, (*shape[1:],))
    return xr.DataArray(data, coords=coords, dims=dims).to_dataset(name="var")
