from typing import Any

import equinox as eqx
import jax.core
import jax.lax as lax
import jax.numpy as jnp
import numpy as np

from ..core import ArrayValue, DenseArrayValue, register


class Zero(ArrayValue):
    _shape: tuple[int, ...] = eqx.field(static=True)
    _dtype: Any = eqx.field(static=True)

    def __init__(self, shape: tuple[int, ...], dtype: Any):
        self._shape = shape
        self._dtype = dtype

    def aval(self):
        return jax.core.ShapedArray(self.shape, self.dtype)

    def materialise(self):
        return jnp.zeros(self.shape, self.dtype)


@register(lax.broadcast_in_dim_p)
def _(value: DenseArrayValue, *, broadcast_dimensions, shape) -> ArrayValue:
    arraylike = value.array
    if (
        isinstance(
            arraylike, (bool, int, float, complex, np.bool_, np.integer, np.inexact)
        )
        and value == 0
    ):
        return Zero(shape, np.result_type(arraylike))
    else:
        out = lax.broadcast_in_dim_p.bind(
            arraylike, broadcast_dimensions=broadcast_dimensions, shape=shape
        )
        return DenseArrayValue(out)


@register(lax.add_p)
def _(x: ArrayValue, y: Zero) -> ArrayValue:
    return x


@register(lax.add_p)
def _(x: Zero, y: ArrayValue) -> ArrayValue:
    return y


@register(lax.add_p)
def _(x: Zero, y: Zero) -> Zero:
    return x


@register(lax.mul_p)
def _(x: ArrayValue, y: Zero) -> Zero:
    return y


@register(lax.mul_p)
def _(x: Zero, y: ArrayValue) -> Zero:
    return x


@register(lax.mul_p)
def _(x: Zero, y: Zero) -> Zero:
    return x


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
    return Zero(shape, operand.dtype)


def _zero_matmul(lhs: ArrayValue, rhs: ArrayValue, kwargs) -> Zero:
    out_aval = lax.dot_general_p.abstract_eval(lhs.aval(), rhs.aval(), **kwargs)
    return Zero(out_aval.shape, out_aval.dtype)


@register(lax.dot_general_p)
def _(lhs: Zero, rhs: ArrayValue, **kwargs) -> Zero:
    return _zero_matmul(lhs, rhs, kwargs)


@register(lax.dot_general_p)
def _(lhs: ArrayValue, rhs: Zero, **kwargs) -> Zero:
    return _zero_matmul(lhs, rhs, kwargs)


@register(lax.dot_general_p)
def _(lhs: Zero, rhs: Zero, **kwargs) -> Zero:
    return _zero_matmul(lhs, rhs, kwargs)
