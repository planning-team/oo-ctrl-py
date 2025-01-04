import numpy as np

from typing import Optional, Union, Tuple


def wrap_angle(angle: Union[np.ndarray, float]) -> float:
    return (angle + np.pi) % (2 * np.pi) - np.pi


def extract_dims(vector_array: np.ndarray, 
                 dims: Optional[Union[int, Tuple[int, ...]]]) -> np.ndarray:
    if dims is not None:
        if isinstance(dims, int):
            return vector_array[..., :dims]
        else:
            return vector_array[..., dims]
    else:
        return vector_array
    
    
def vec_mat_vec(vec_l: np.ndarray,
                mat: Optional[np.ndarray],
                vec_r: np.ndarray) -> np.ndarray:
    if mat is not None:
        return np.einsum("...i,ij,...j->...", vec_l, mat, vec_r)
    else:
        return np.einsum("...i,...j->...", vec_l, vec_r)
