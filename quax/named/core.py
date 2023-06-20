import dataclasses
from collections.abc import Callable
from typing import Any, Union

import equinox as eqx
import jax.core
import jax.lax as lax
import jax.numpy as jnp
import plum
from jaxtyping import ArrayLike

from ..core import ArrayValue, quaxify, register


@dataclasses.dataclass(frozen=True)
class Axis:
    name: str
    size: int


AxisSelector = Union[Axis, str]


class NamedArray(ArrayValue):
    array: ArrayLike
    axes: tuple[Axis, ...] = eqx.field(static=True)
    allow_materialise: bool = eqx.field(default=False, static=True)

    def __post_init__(self):
        if len(set((axis.name for axis in self.axes))) != len(self.axes):
            raise ValueError("Axis names for `NamedArray` must be unique.")
        if jnp.ndim(self.array) != len(self.axes):
            raise ValueError("`NamedArray` must have every axis be named.")
        for size, axis in zip(jnp.shape(self.array), self.axes):
            if size != axis.size:
                raise ValueError(f"Mismatched axis size for axis {axis}.")

    def materialise(self):
        if self.allow_materialise:
            return self.array
        else:
            raise RuntimeError(
                "Refusing to materialise `NamedArray` with `allow_materialise=False`."
            )

    def aval(self) -> jax.core.ShapedArray:
        return jax.core.get_aval(self.array)  # pyright: ignore


def _broadcast_axes(axes1, axes2):
    if len(axes1) == 0:
        return axes2
    elif len(axes2) == 0:
        return axes1
    elif set(axes1).issubset(set(axes2)):
        return axes2
    elif set(axes2).issubset(set(axes1)):
        return axes1
    else:
        raise ValueError(f"Cannot broadcast {axes1} against {axes2}")


def _wrap_elementwise_binop(
    op: Callable[[Any, Any], Any], prim: jax.core.Primitive
) -> Callable[[ArrayValue, ArrayValue], NamedArray]:
    @plum.Dispatcher().abstract
    def _op(x, y) -> NamedArray:
        assert False

    @_op.dispatch
    @register(prim)
    def _(x: NamedArray, y: NamedArray) -> NamedArray:
        axes = _broadcast_axes(x.axes, y.axes)
        return NamedArray(op(x.array, y.array), axes)

    @_op.dispatch
    @register(prim)
    def _(x: ArrayValue, y: NamedArray) -> NamedArray:
        if x.shape == ():
            return NamedArray(op(x, y.array), y.axes)
        else:
            raise ValueError(f"Cannot apply {op} to non-scalar array and named array.")

    @_op.dispatch
    @register(prim)
    def _(x: NamedArray, y: ArrayValue) -> NamedArray:
        if y.shape == ():
            return NamedArray(op(x.array, y), x.axes)
        else:
            raise ValueError(f"Cannot apply {op} to non-scalar array and named array.")

    return _op


add = _wrap_elementwise_binop(lax.add, lax.add_p)
mul = _wrap_elementwise_binop(lax.mul, lax.mul_p)
sub = _wrap_elementwise_binop(lax.sub, lax.sub_p)


def resolve_axis(array: NamedArray, axis: AxisSelector):
    axes = [axis.name for axis in array.axes]
    if isinstance(axis, Axis):
        axis = axis.name
    return axes.index(axis)


def trace(
    array: NamedArray,
    *,
    offset: int = 0,
    axis1: AxisSelector,
    axis2: AxisSelector,
    dtype=None,
) -> NamedArray:
    index1 = resolve_axis(array, axis1)
    index2 = resolve_axis(array, axis2)
    if index1 == index2:
        raise ValueError("Cannot trace along the same named axis.")
    inner = jnp.trace(
        array.array, offset=offset, axis1=index1, axis2=index2, dtype=dtype
    )
    axes = tuple(x for i, x in enumerate(array.axes) if i not in (index1, index2))
    return NamedArray(inner, axes)


@register(lax.dot_general_p)
def _(lhs: NamedArray, rhs: NamedArray, *, dimension_numbers, **kwargs) -> NamedArray:
    ((lhs_contract, rhs_contract), (lhs_batch, rhs_batch)) = dimension_numbers
    if {lhs.axes[i] for i in lhs_contract} != {rhs.axes[i] for i in rhs_contract}:
        raise TypeError("Cannot contract mismatched dimensions.")
    if {lhs.axes[i] for i in lhs_batch} != {rhs.axes[i] for i in rhs_batch}:
        raise TypeError("Cannot batch mismatched dimensions.")
    out = lax.dot_general(lhs.array, rhs.array, dimension_numbers, **kwargs)
    shared = tuple(lhs.axes[i] for i in lhs_batch)
    lhs_used = lhs_contract + lhs_batch
    rhs_used = rhs_contract + rhs_batch
    lhs_unused = tuple(axis for i, axis in enumerate(lhs.axes) if i not in lhs_used)
    rhs_unused = tuple(axis for i, axis in enumerate(rhs.axes) if i not in rhs_used)
    out_axes = shared + lhs_unused + rhs_unused
    return NamedArray(out, out_axes)


matmul = quaxify(jnp.matmul)
tensordot = quaxify(jnp.tensordot)
