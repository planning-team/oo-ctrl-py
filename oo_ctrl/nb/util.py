import numpy as np

from numba import njit
from typing import Optional, Union, Tuple


@njit(parallel=True)
def wrap_angle(angle: float) -> float:
    return (angle + np.pi) % (2 * np.pi) - np.pi


@njit(parallel=True)
def extract_dims(vector_array: np.ndarray, 
                 dims: Optional[Union[int, Tuple[int, ...]]]) -> np.ndarray:
    if dims is not None:
        if isinstance(dims, int):
            return vector_array[..., :dims]
        else:
            # TODO: Fix or remove this case for Numba
            return vector_array[..., dims]
    else:
        return vector_array
