import numpy as np
import xarray as xr


def ds_1var(shape):
    """An example DataSet.
    The first dimension is named 'sample' and the data is identical over other dimensions
    """
    coords = [range(s) for s in shape]
    dims = ["sample"] + [f"dim_{i}" for i in range(1, len(shape))]
    data = np.expand_dims(np.random.random(size=shape[0]), list(range(1, len(shape))))
    data = np.tile(data, (*shape[1:],))
    return xr.DataArray(data, coords=coords, dims=dims).to_dataset(name="var")
