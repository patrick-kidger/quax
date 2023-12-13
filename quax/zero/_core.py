import functools as ft
from typing import Any

import equinox as eqx
import jax.core
import jax.lax as lax
import jax.numpy as jnp
import numpy as np

from .._core import ArrayValue, DenseArrayValue, quaxify_keepwrap, register


class Zero(ArrayValue):
    """Represents a symbolic zero value. Operations like `array + zero` will be
    reduced down to just `array` at trace time.

    This is essentially a generalised version of the symbolic zeros used by JAX in its
    autodifferentiation rules -- in our case, we can apply them at any time, not just
    during AD.
    """

    _shape: tuple[int, ...] = eqx.field(static=True)
    _dtype: Any = eqx.field(static=True)

    def __init__(self, shape: tuple[int, ...], dtype: Any):
        """**Arguments:**

        - `shape`: the shape of the zero array.
        - `dtype`: the dtype of the zero array.
        """

        self._shape = shape
        self._dtype = dtype

    def aval(self):
        return jax.core.ShapedArray(self._shape, self._dtype)

    def materialise(self):
        return jnp.zeros(self._shape, self._dtype)


@register(lax.broadcast_in_dim_p)
def _(value: DenseArrayValue, *, broadcast_dimensions, shape) -> ArrayValue:
    arraylike = value.array
    aval = jax.core.get_aval(arraylike)
    if isinstance(aval, jax.core.ConcreteArray) and aval.shape == () and aval.val == 0:
        return Zero(shape, np.result_type(arraylike))
    else:
        # Avoid an infinite loop, by pushing a new interpreter to the dynamic
        # interpreter stack.
        with jax.ensure_compile_time_eval():
            out = lax.broadcast_in_dim_p.bind(
                arraylike, broadcast_dimensions=broadcast_dimensions, shape=shape
            )
        return DenseArrayValue(out)  # pyright: ignore


@register(lax.broadcast_in_dim_p)
def _(value: Zero, *, broadcast_dimensions, shape) -> Zero:
    del broadcast_dimensions
    return Zero(shape, value.dtype)


@register(lax.convert_element_type_p)
def _(value: Zero, *, new_dtype, weak_type) -> Zero:
    del weak_type
    return Zero(value.shape, new_dtype)


@quaxify_keepwrap
def _shape_dtype(x, y, value):
    shape = jnp.broadcast_shapes(x.shape, y.shape)
    dtype = jnp.result_type(x.dtype, y.dtype)
    return jnp.broadcast_to(value, shape).astype(dtype)


@register(lax.add_p)
def _(x: ArrayValue, y: Zero) -> ArrayValue:
    return _shape_dtype(x, y, value=x)


@register(lax.add_p)
def _(x: Zero, y: ArrayValue) -> ArrayValue:
    return _shape_dtype(x, y, value=y)


@register(lax.add_p)
def _(x: Zero, y: Zero) -> Zero:
    return _shape_dtype(x, y, value=x)


@register(lax.mul_p)
def _(x: ArrayValue, y: Zero) -> Zero:
    return _shape_dtype(x, y, value=y)


@register(lax.mul_p)
def _(x: Zero, y: ArrayValue) -> Zero:
    return _shape_dtype(x, y, value=x)


@register(lax.mul_p)
def _(x: Zero, y: Zero) -> Zero:
    return _shape_dtype(x, y, value=x)


@register(lax.dynamic_update_slice_p)
def _(operand: Zero, update: Zero, *indices) -> Zero:
    del update, indices
    return operand


@register(lax.dynamic_slice_p)
def _(operand: Zero, *indices, slice_sizes) -> Zero:
    del indices
    return Zero(slice_sizes, operand.dtype)


@register(lax.slice_p)
def _(operand: Zero, *, start_indices, limit_indices, strides) -> Zero:
    if strides is None:
        strides = [1 for _ in start_indices]
    shape = [
        (limit - start) // stride
        for start, limit, stride in zip(start_indices, limit_indices, strides)
    ]
    return Zero(tuple(shape), operand.dtype)


def _zero_matmul(lhs: ArrayValue, rhs: ArrayValue, kwargs) -> Zero:
    out_struct = jax.eval_shape(
        ft.partial(lax.dot_general_p.bind, **kwargs), lhs.aval(), rhs.aval()
    )
    return Zero(out_struct.shape, out_struct.dtype)


@register(lax.dot_general_p)
def _(lhs: Zero, rhs: ArrayValue, **kwargs) -> Zero:
    return _zero_matmul(lhs, rhs, kwargs)


@register(lax.dot_general_p)
def _(lhs: ArrayValue, rhs: Zero, **kwargs) -> Zero:
    return _zero_matmul(lhs, rhs, kwargs)


@register(lax.dot_general_p)
def _(lhs: Zero, rhs: Zero, **kwargs) -> Zero:
    return _zero_matmul(lhs, rhs, kwargs)
