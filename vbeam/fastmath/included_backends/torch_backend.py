import functools
from typing import TYPE_CHECKING, Type

import numpy as np
import torch

from vbeam.fastmath.backend import Backend


# if TYPE_CHECKING:
#    from vbeam.module import Module
class Module:
    pass


def call_without_none_dim_kwarg(f, *args, **kwargs):
    """Call f with the arguments, but remove 'dim' from keywords if it exists and is
    None. The behavior of Numpy is to treat None as if the user did not supply that
    argument, but PyTorch crashes if dim is None.

    This function's purpose is purely for reducing boiler-plate."""
    if "dim" in kwargs and kwargs["dim"] is None:
        del kwargs["dim"]
    return f(*args, **kwargs)


def ensure_tensors(f):
    @functools.wraps(f)
    def wrapped(*args, **kwargs):
        args = [
            torch.from_numpy(arg) if isinstance(arg, np.ndarray) else arg
            for arg in args
        ]
        kwargs = {
            k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v
            for k, v in kwargs.items()
        }
        return f(*args, **kwargs)

    return wrapped


class TorchBackend(Backend):
    @property
    def ndarray(self):
        return torch.Tensor

    def zeros(self, shape, dtype=None):
        return torch.from_numpy(np.zeros(shape, dtype))

    def ones(self, shape, dtype=None):
        return torch.from_numpy(np.ones(shape, dtype))

    @property
    def pi(self):
        return torch.pi

    @ensure_tensors
    def abs(self, x):
        return torch.abs(x)

    @ensure_tensors
    def exp(self, x):
        return torch.exp(x)

    @ensure_tensors
    def log(self, x):
        return torch.log(x)

    @ensure_tensors
    def log10(self, x):
        return torch.log10(x)

    @ensure_tensors
    def sin(self, x):
        return torch.sin(x)

    @ensure_tensors
    def cos(self, x):
        return torch.cos(x)

    @ensure_tensors
    def tan(self, x):
        return torch.tan(x)

    @ensure_tensors
    def arcsin(self, x):
        return torch.arcsin(x)

    @ensure_tensors
    def arccos(self, x):
        return torch.arccos(x)

    @ensure_tensors
    def arctan2(self, y, x):
        return torch.arctan2(y, x)

    @ensure_tensors
    def sqrt(self, x):
        return torch.sqrt(x)

    @ensure_tensors
    def sign(self, x):
        return torch.sign(x)

    @ensure_tensors
    def nan_to_num(self, x, nan=0.0):
        return torch.nan_to_num(x, nan=nan)

    @ensure_tensors
    def min(self, a, axis=None):
        return call_without_none_dim_kwarg(torch.min, a, dim=axis)

    @ensure_tensors
    def minimum(self, a, b):
        return torch.minimum(a, b)

    @ensure_tensors
    def max(self, a, axis=None):
        return call_without_none_dim_kwarg(torch.max, a, dim=axis)

    @ensure_tensors
    def maximum(self, a, b):
        return torch.maximum(a, b)

    @ensure_tensors
    def argmin(self, a, axis=None):
        return call_without_none_dim_kwarg(torch.argmin, a, dim=axis)

    @ensure_tensors
    def argmax(self, a, axis=None):
        return call_without_none_dim_kwarg(torch.argmax, a, dim=axis)

    @ensure_tensors
    def sum(self, a, axis=None):
        return call_without_none_dim_kwarg(torch.sum, a, dim=axis)

    @ensure_tensors
    def prod(self, a, axis=None):
        return call_without_none_dim_kwarg(torch.prod, a, dim=axis)

    @ensure_tensors
    def mean(self, a, axis=None):
        return call_without_none_dim_kwarg(torch.mean, a, dim=axis)

    @ensure_tensors
    def median(self, a, axis=None):
        return call_without_none_dim_kwarg(torch.median, a, dim=axis)

    @ensure_tensors
    def deg2rad(self, a):
        return torch.deg2rad(a)

    @ensure_tensors
    def diff(self, a, axis=-1):
        return call_without_none_dim_kwarg(torch.diff, a, dim=axis)

    @ensure_tensors
    def var(self, a, axis=None):
        return call_without_none_dim_kwarg(torch.var, a, dim=axis)

    @ensure_tensors
    def cumsum(self, a, axis=None):
        return call_without_none_dim_kwarg(torch.cumsum, a, dim=axis)

    @ensure_tensors
    def cross(self, a, b, axis=None):
        # TODO
        return torch.cross(a, b, dim=axis if axis is not None else -1)

    @ensure_tensors
    def nansum(self, a, axis=None):
        return call_without_none_dim_kwarg(torch.nansum, a, dim=axis)

    @ensure_tensors
    def histogram(self, a, bins=10, weights=None):
        return torch.histogram(a, bins=bins, weights=weights)

    def array(self, x, dtype=None):
        if isinstance(dtype, torch.dtype):
            return torch.from_numpy(np.array(x)).to(dtype)
        else:
            return torch.from_numpy(np.array(x, dtype=dtype))

    @ensure_tensors
    def flip(self, a, axis=None):
        dims = list(range(a.ndim)) if axis is None else axis
        return torch.flip(a, dims=dims)

    @ensure_tensors
    def transpose(self, a, axes=None):
        return torch.permute(a, dims=axes)

    @ensure_tensors
    def swapaxes(self, a, axis1, axis2):
        return torch.swapaxes(a, axis1, axis2)

    @ensure_tensors
    def moveaxis(self, a, source, destination):
        return torch.moveaxis(a, source, destination)

    def stack(self, arrays, axis=0):
        return call_without_none_dim_kwarg(torch.stack, arrays, dim=axis)

    @ensure_tensors
    def tile(self, A, reps):
        return torch.tile(A, reps)

    def concatenate(self, arrays, axis=0):
        return call_without_none_dim_kwarg(torch.concatenate, arrays, dim=axis)

    @ensure_tensors
    def meshgrid(self, *xi, indexing="xy"):
        return torch.meshgrid(*xi, indexing=indexing)

    @ensure_tensors
    def linspace(self, start, stop, num=50):
        return torch.linspace(start, stop, steps=num)

    @ensure_tensors
    def arange(self, start, stop=None, step=None):
        return torch.arange(start, stop, step)

    @ensure_tensors
    def expand_dims(self, a, axis):
        return call_without_none_dim_kwarg(torch.unsqueeze, a, dim=axis)

    @ensure_tensors
    def ceil(self, x):
        return torch.ceil(x)

    @ensure_tensors
    def floor(self, x):
        return torch.floor(x)

    @ensure_tensors
    def modf(self, x):
        integer_part = torch.trunc(x)
        fractional_part = x - integer_part
        return fractional_part, integer_part

    @ensure_tensors
    def round(self, x):
        return torch.round(x)

    @ensure_tensors
    def clip(self, a, a_min, a_max):
        return torch.clip(a, a_min, a_max)

    @ensure_tensors
    def where(self, condition, x, y):
        return torch.where(condition, x, y)

    @ensure_tensors
    def select(self, condlist, choicelist, default=0):
        return torch.select(condlist, choicelist, default)

    @ensure_tensors
    def logical_or(self, x1, x2):
        return torch.logical_or(x1, x2)

    @ensure_tensors
    def logical_and(self, x1, x2):
        return torch.logical_and(x1, x2)

    @ensure_tensors
    def squeeze(self, a, axis=None):
        return call_without_none_dim_kwarg(torch.squeeze, a, dim=axis)

    @ensure_tensors
    def ravel(self, a):
        return torch.ravel(a)

    @ensure_tensors
    def take(self, a, indices, axis=None):
        # TODO
        return torch.take(a, indices, axis=axis)

    @ensure_tensors
    def interp(self, x, xp, fp, left=None, right=None, period=None):
        # TODO
        return torch.interpolate(x, xp, fp, left, right, period)

    @ensure_tensors
    def gather(self, a, indices):
        return a[indices]

    @ensure_tensors
    def shape(self, x):
        return torch.Tensor.shape(x)

    class add:
        @staticmethod
        @ensure_tensors
        def at(a, indices, b):
            return a.at[indices].add(b)

    def jit(self, fun, static_argnums=None, static_argnames=None):
        return fun
        return torch.compile(fun)

    def vmap(self, fun, in_axes, out_axes=0):
        return torch.vmap(fun, in_axes, out_axes)

    def scan(self, f, init, xs):
        carry = init
        ys = []
        for x in xs:
            carry, y = f(carry, x)
            ys.append(y)
        return carry, torch.stack(ys)

    def make_traceable(self, cls: Type["Module"]):
        # TODO: Make the class act like a PyTorch Module
        # Noop
        return cls
