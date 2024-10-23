from dataclasses import dataclass
import functools
import numpy as np
import scipy

from vbeam.fastmath.backend import Backend, i_at
# from vbeam.fastmath.traceable import (
#     get_traceable_aux_fields,
#     get_traceable_data_fields,
#     is_traceable_dataclass,
# )

# def ensure_numpy_wrapper(f):
#     @functools.wraps(f)
#     def wrapped(*args, **kwargs):
#         return NumpyWrapper(f(*args, **kwargs))
#     return wrapped

# class NumpyWrapper(np.ndarray):

#     def __new__(cls, a):
#         obj = np.asarray(a).view(cls)
#         return obj

#     def to(self, device):
#         if device!='cpu':
#             raise ValueError("Numpy backend only supports device=CPU")
#         return self

class NumpyBackend(Backend):
    @property
    def ndarray(self):
        return np.ndarray

    def zeros(self, shape, dtype=None, device='cpu'):
        return np.zeros(shape, dtype)

    def ones(self, shape, dtype=None, device='cpu'):
        return np.ones(shape, dtype)

    @property
    def pi(self):
        return np.pi

    @property
    def int8(self):
        return np.int8

    @property
    def int16(self):
        return np.int16

    @property
    def int32(self):
        return np.int32 
    
    @property
    def int64(self):
        return np.int64       
    
    @property
    def uint8(self):
        return np.uint8

    @property
    def uint16(self):
        return np.uint16

    @property
    def uint32(self):
        return np.uint32 
    
    @property
    def uint64(self):
        return np.uint64       

    @property
    def float32(self):
        return np.float32 
    
    @property
    def float64(self):
        return np.float64   

    @property
    def complex64(self):
        return np.complex64   
    
    @property
    def complex128(self):
        return np.complex128         

    def to_dtype(self, x, dtype):
        return x.astype(dtype)
    
    def to_int32(self, x):
        return x.astype(np.int32)
    
    def to_int64(self, x):
        return x.astype(np.int64)
    
    def to_uint8(self, x):
        return x.astype(np.uint8)

    def to_uint16(self, x):
        return x.astype(np.uint16)
    
    def to_uint32(self, x):
        return x.astype(np.uint32) 
    
    def to_uint64(self, x):
        return x.astype(np.uint64)  
          
    def to_float32(self, x):
        return x.astype(np.float32) 
    
    def to_float64(self, x):
        return x.astype(np.float64) 
      
    def to_complex64(self, x):
        return x.astype(np.complex64) 
      
    def to_complex128(self, x):
        return x.astype(np.complex128)  

    def abs(self, x):
        return np.abs(x)

    def exp(self, x):
        return np.exp(x)

    def log(self, x):
        return np.log(x)

    def log10(self, x):
        return np.log10(x)

    def sin(self, x):
        return np.sin(x)

    def cos(self, x):
        return np.cos(x)

    def tan(self, x):
        return np.tan(x)

    def arcsin(self, x):
        return np.arcsin(x)

    def arccos(self, x):
        return np.arccos(x)

    def arctan2(self, y, x):
        return np.arctan2(y, x)

    def sqrt(self, x):
        return np.sqrt(x)

    def sign(self, x):
        return np.sign(x)

    def nan_to_num(self, x, nan=0.0):
        return np.nan_to_num(x, nan=nan)

    def min(self, a, axis=None):
        return np.min(a, axis=axis)
    
    def minimum(self, a, b):
        return np.minimum(a, b)
    
    def max(self, a, axis=None):
        return np.max(a, axis=axis)
    
    def maximum(self, a, b):
        return np.minimum(a, b)

    def argmin(self, a, axis=None):
        return np.argmin(a, axis=axis)

    def argmax(self, a, axis=None):
        return np.argmax(a, axis=axis)

    def minimum(self, a, b):
        return np.minimum(a, b)

    def maximum(self, a, b):
        return np.maximum(a, b)

    def sum(self, a, axis=None):
        return np.sum(a, axis=axis)

    def prod(self, a, axis=None):
        return np.prod(a, axis=axis)

    def mean(self, a, axis=None, keepdims=False):
        return np.mean(a, axis=axis, keepdims=keepdims)
    
    def median(self, a, axis=None):
        return np.median(a, axis=axis)

    def deg2rad(self, a):
        return np.deg2rad(a)

    def diff(self, a, axis=-1):
        return np.diff(a, axis=axis)

    def var(self, a, axis=None):
        return np.var(a, axis=axis)

    def cumsum(self, a, axis=None):
        return np.cumsum(a, axis=axis)

    def cross(self, a, b, axisa=-1, axisb=-1, axisc=-1, axis=None):
        return np.cross(a, b, axisa=axisa, axisb=axisb, axisc=axisc, axis=axis)

    def nansum(self, a, axis=None):
        return np.nansum(a, axis=axis)

    def histogram(self, a, bins=10, weights=None):
        return np.histogram(a, bins=bins, weights=weights)

    def array(self, x, dtype=None):
        return np.array(x, dtype=dtype)
    
    def flip(self, a, axis=None):
        return np.flip(a, axis=axis)

    def transpose(self, a, axes=None):
        return np.transpose(a, axes=axes)

    def swapaxes(self, a, axis1, axis2):
        return np.swapaxes(a, axis1, axis2)

    def moveaxis(self, a, source, destination):
        return np.moveaxis(a, source, destination)

    def stack(self, arrays, axis=0):
        return np.stack(arrays, axis=axis)

    def tile(self, A, reps):
        return np.tile(A, reps)

    def concatenate(self, arrays, axis=0):
        return np.concatenate(arrays, axis=axis)

    def meshgrid(self, *xi, indexing="xy"):
        return np.meshgrid(*xi, indexing=indexing)

    def linspace(self, start, stop, num=50):
        return np.linspace(start, stop, num=num)

    def arange(self, start, stop=None, step=None):
        return np.arange(start, stop, step)

    def expand_dims(self, a, axis):
        return np.expand_dims(a, axis=axis)

    def ceil(self, x):
        return np.ceil(x)

    def floor(self, x):
        return np.floor(x)

    def modf(self, x):
        return np.modf(x)

    def round(self, x):
        return np.round(x)

    def clip(self, a, a_min, a_max):
        return np.clip(a, a_min, a_max)

    def where(self, condition, x, y):
        return np.where(condition, x, y)

    def select(self, condlist, choicelist, default=0):
        return np.select(condlist, choicelist, default)

    def logical_or(self, x1, x2):
        return np.logical_or(x1, x2)

    def logical_and(self, x1, x2):
        return np.logical_and(x1, x2)

    def squeeze(self, a, axis=None):
        return np.squeeze(a, axis=axis)

    def ravel(self, a):
        return np.ravel(a)

    def take(self, a, indices, axis=None):
        return np.take(a, indices, axis=axis)

    def interp(self, x, xp, fp, left=None, right=None, period=None):
        return np.interp(x, xp, fp, left, right, period)

    def gather(self, a, indices):
        return a[indices]

    def shape(self, x):
        return np.shape(x)
    
    def ascontiguousarray(self, x):
        return np.ascontiguousarray(x)
    
    def conj(self, x):
        return np.conj(x)
    
    def real(self, x):
        return np.real(x)    
    
    def imag(self, x):
        return np.imag(x)   
    
    def matmul(self, x1, x2):
        return np.matmul(x1, x2)

    def reshape(self, x, shape):
        return np.reshape(x, shape)

    def bitwise_and(self, x1, x2):
        return np.bitwise_and(x1, x2)   
    
    def bitwise_or(self, x1, x2):
        return np.bitwise_or(x1, x2)    

    def left_shift(self, x1, x2):
        return np.left_shift(x1, x2)  
    
    def right_shift(self, x1, x2):
        return np.right_shift(x1, x2)  
    
    def double(self, x):
        return np.double(x)      

    def dot(self, x1, x2):
        return np.dot(x1, x2) 

    def power(self, x1, x2):
        return np.power(x1, x2)     
    
    def eye(self, N, M=None):
        return np.eye(N, M=M)     
        
    def from_numpy(self, x):
        return x
    
    def to_numpy(self, x):
        return x

    def to_device(self, x, device):
        if device!='cpu':
            raise ValueError("Numpy backend only supports device=CPU")        
        return x

    def get_activate_backend(self) -> str:
        return 'numpy'    

    def correlate2d(self, x1, x2):
        return scipy.signal.correlate2d(x1, x2)
    
    def angle(self, x):
        return np.angle(x)    

    class fft:
        @staticmethod
        def ifftshift(x, axes=None):
            return np.fft.ifftshift(x, axes=axes)

    class random:
        @staticmethod
        def randn(*vars, dtype=np.float64, device='cpu'):
            if device!='cpu':
                raise ValueError(f"Unsupported device {device} on numpy backend")
            return np.random.randn(*vars).astype(dtype)
        
    class linalg:
        @staticmethod
        def qr(x, mode='reduced'):
            return np.linalg.qr(x, mode)      

    class add:
        @staticmethod
        def at(a, indices, b):
            a = a.copy()
            np.add.at(a, indices, b)
            return a

    def jit(self, fun, static_argnums=None, static_argnames=None):
        return fun  # No-op

    def vmap(self, fun, in_axes, out_axes=0):
        v_axes = [(i, ax) for i, ax in enumerate(in_axes) if ax is not None]

        def vectorized_fun(*args, **kwargs):
            v_sizes = [args[i].shape[ax] for i, ax in v_axes]
            v_ax_size = v_sizes[0]
            assert all(
                [v_size == v_ax_size for v_size in v_sizes]
            ), "All vectorized axes must have the same number of elements."

            results = []
            for i in range(v_ax_size):
                new_args = [
                    args[j] if ax is None else i_at(args[j], i, ax)
                    for j, ax in enumerate(in_axes)
                ]
                results.append(fun(*new_args, **kwargs))
            results = _recombine_traceables(results)
            results = _set_out_axes(results, out_axes)
            return results

        return vectorized_fun

    def scan(self, f, init, xs):
        carry = init
        ys = []
        for x in xs:
            carry, y = f(carry, x)
            ys.append(y)
        return carry, np.stack(ys)

    def as_traceable_dataclass_obj(self, obj, data_fields, aux_fields):
        # Numpy is "traceable" by default (because there is no tracing).
        as_dataclass = dataclass(type(obj))  # Make it a dataclass, though.
        obj.__class__ = as_dataclass
        return obj
    



def _set_out_axes(result: np.ndarray, out_axis: int):
    result_type = type(result)
    if is_traceable_dataclass(result_type):
        data_fields = get_traceable_data_fields(result_type)
        aux_fields = get_traceable_aux_fields(result_type)
        kwargs = {field: getattr(result_type, field) for field in aux_fields}
        for field in data_fields:
            kwargs[field] = _set_out_axes(getattr(result, field), out_axis)
        return result_type(**kwargs)
    elif isinstance(result, tuple):
        return tuple([_set_out_axes(r, out_axis) for r in result])
    else:
        return np.moveaxis(result, 0, out_axis)


def _recombine_traceables(results: list):
    """Ensure that the returned value of a vmapped function is consistent with jax.

    If the results is a list of traceable dataclass objects, then they are combined into
    a single instance of that object.
    If it is a tuple, a tuple is returned, processed recursively.
    Otherwise, the results are simply returned as a numpy array."""
    if is_traceable_dataclass(type(results[0])):
        assert all([is_traceable_dataclass(type(result)) for result in results])
        result_type = type(results[0])
        data_fields = get_traceable_data_fields(result_type)
        aux_fields = get_traceable_aux_fields(result_type)
        kwargs = {field: getattr(result_type, field) for field in aux_fields}
        for field in data_fields:
            results_for_field = [getattr(result, field) for result in results]
            kwargs[field] = np.stack(results_for_field, 0)
        return result_type(**kwargs)
    elif isinstance(results[0], tuple):
        assert all([isinstance(result, tuple) for result in results])
        n_items = len(results[0])
        return tuple(
            [_recombine_traceables([r[i] for r in results]) for i in range(n_items)]
        )
    else:
        return np.array(results)


