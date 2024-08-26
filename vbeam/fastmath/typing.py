"""Type hints for dynamic backend arrays.

The "base" type is a Numpy array. If you have JAX installed, its respective array type 
is added to the type definition. Same for PyTorch."""

from typing import Annotated, Union

import numpy as np

NumpyArray = np.ndarray

try:
    import jax

    JaxArray = jax.Array
except ImportError:
    JaxArray = np.ndarray

try:
    import torch

    TorchArray = torch.Tensor
except ImportError:
    TorchArray = np.ndarray

Array = Annotated[
    Union[NumpyArray, JaxArray, TorchArray],
    "__fastmath_array__",
]
