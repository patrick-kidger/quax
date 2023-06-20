import abc
import functools as ft
import operator
import typing_extensions
from collections.abc import Callable
from typing import Union

import equinox as eqx
import jax
import jax._src
import jax.core as core
import jax.tree_util as jtu
import plum
from jaxtyping import ArrayLike


#
# Rules
#


_rules: dict[core.Primitive, Callable] = {}


def register(primitive: core.Primitive):
    def _register(rule):
        try:
            existing_rule = _rules[primitive]
        except KeyError:

            @plum.Dispatcher().abstract
            def existing_rule():
                assert False

            _rules[primitive] = existing_rule
        existing_rule.dispatch(rule)
        return rule

    return _register


#
# Interpreter
#


class _QuaxTracer(core.Tracer):
    __slots__ = ("value",)

    def __init__(self, trace: "_QuaxTrace", value: "Value"):
        assert _is_value(value)
        self._trace = trace
        self.value = value

    @property
    def aval(self):
        return self.value.aval()

    def full_lower(self):
        if isinstance(self.value, DenseArrayValue):
            return core.full_lower(self.value.array)
        else:
            return self


def _default_process(primitive, values, params):
    arrays = [x.materialise() for x in values]
    with jax.ensure_compile_time_eval():
        out = primitive.bind(*arrays, **params)
    if primitive.multiple_results:
        return [DenseArrayValue(x) for x in out]
    else:
        return DenseArrayValue(out)


class _QuaxTrace(core.Trace[_QuaxTracer]):
    def pure(self, val: Union[ArrayLike, "Value"]) -> _QuaxTracer:
        if not _is_value(val):
            val = DenseArrayValue(val)
        return _QuaxTracer(self, val)  # pyright: ignore

    def lift(self, tracer: core.Tracer) -> _QuaxTracer:
        return _QuaxTracer(self, DenseArrayValue(tracer))

    def sublift(self, tracer: _QuaxTracer) -> _QuaxTracer:
        return tracer

    def process_primitive(self, primitive, tracers, params):
        values = [t.value for t in tracers]
        try:
            rule = _rules[primitive]
        except KeyError:
            out = _default_process(primitive, values, params)
        else:
            try:
                out = rule(*values, **params)
            except plum.NotFoundLookupError:
                out = _default_process(primitive, values, params)
        if primitive.multiple_results:
            return [_QuaxTracer(self, x) for x in out]
        else:
            return _QuaxTracer(self, out)  # pyright: ignore

    def process_custom_jvp_call(self, primitive, fun, jvp, tracers, *, symbolic_zeros):
        return fun.call_wrapped(*tracers)


#
# API
#


def _wrap_tracer(trace: _QuaxTrace, x):
    if _is_value(x):
        return _QuaxTracer(trace, x)
    else:
        return x


def _unwrap_tracer(trace, unwrap_builtin_value, x):
    if eqx.is_array_like(x):
        x = trace.full_raise(x)
    if isinstance(x, _QuaxTracer):
        if unwrap_builtin_value and isinstance(x.value, DenseArrayValue):
            return x.value.array
        else:
            return x.value
    else:
        return x


class _Quaxify(eqx.Module):
    fn: Callable
    unwrap_builtin_value: bool

    @property
    def __wrapped__(self):
        return self.fn

    def __call__(self, *args, **kwargs):
        with core.new_main(_QuaxTrace, dynamic=True) as main:
            trace = _QuaxTrace(main, core.cur_sublevel())
            # Note that we do *not* wrap arraylikes here. We let that happen in
            # `_QuaxTrace.{pure,lift}` as necessary. This means that we can do e.g.
            # quaxify(jnp.moveaxis)(array, source=0, destination=-1).
            fn, args, kwargs = jtu.tree_map(
                ft.partial(_wrap_tracer, trace),
                (self.fn, args, kwargs),
                is_leaf=_is_value,
            )
            out = fn(*args, **kwargs)
            out = jtu.tree_map(
                ft.partial(_unwrap_tracer, trace, self.unwrap_builtin_value), out
            )
            return out

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return eqx.Partial(self, instance)


def quaxify(fn, unwrap_builtin_value: bool = True):
    return eqx.module_update_wrapper(
        _Quaxify(fn, unwrap_builtin_value=unwrap_builtin_value), fn
    )


quaxify_keepwrap = ft.partial(quaxify, unwrap_builtin_value=False)


#
# Values
#


class Value(eqx.Module):
    @abc.abstractmethod
    def materialise(self) -> ArrayLike:
        pass

    @abc.abstractmethod
    def aval(self) -> core.AbstractValue:
        pass


def _is_value(x) -> "typing_extensions.StrictTypeGuard[Value]":
    return isinstance(x, Value)


def _flip_binop(binop):
    def _binop(x, y):
        return binop(y, x)

    return _binop


class ArrayValue(Value):
    @abc.abstractmethod
    def aval(self) -> core.ShapedArray:
        pass

    @property
    def dtype(self):
        return self.aval().dtype

    @property
    def ndim(self):
        return self.aval().ndim

    @property
    def size(self):
        return self.aval().size

    @property
    def itemsize(self):
        return self.aval().itemsize  # pyright: ignore

    @property
    def shape(self):
        return self.aval().shape

    @property
    def sharding(self):
        raise ValueError("ArrayValues do not have a notion of sharding.")

    @property
    def addressable_shards(self):
        raise ValueError("ArrayValues do not have a notion of sharding.")

    __add__ = quaxify(operator.add)
    __radd__ = quaxify(_flip_binop(operator.add))
    __sub__ = quaxify(operator.sub)
    __rsub__ = quaxify(_flip_binop(operator.sub))
    __mul__ = quaxify(operator.mul)
    __rmul__ = quaxify(_flip_binop(operator.mul))
    __matmul__ = quaxify(operator.matmul)
    __rmatmul__ = quaxify(_flip_binop(operator.matmul))
    # TODO: add all other methods and properties and things


class DenseArrayValue(ArrayValue):
    array: ArrayLike

    def materialise(self) -> ArrayLike:
        return self.array

    def aval(self) -> core.ShapedArray:
        return core.get_aval(self.array)  # pyright: ignore


@register(jax._src.pjit.pjit_p)  # pyright: ignore
def _(*args: ArrayValue, jaxpr, **kwargs):
    return quaxify_keepwrap(core.jaxpr_as_fun(jaxpr))(*args)


#
# Posterity: we use a final-style (on-the-fly) interpreter above, but this is what an
# initial-style (staged) interpreter looks like.
# The final-style is preferred where possible, as it (a) supports Python control flow,
# and (b) I speculate should sometimes be faster. (E.g. when nesting multiple quaxifys,
# and not needing to parse the jaxpr whilst building the jaxpr in an upper level.)
#
#
# def _to_value(x):
#     if eqx.is_array(x):
#         return DenseArrayValue(x)
#     else:
#         return x


# def _to_struct(x):
#     if _is_value(x):
#         if not isinstance(x.aval(), core.ShapedArray):
#             raise NotImplementedError
#         return jax.ShapeDtypeStruct(x.shape, x.dtype)
#     else:
#         return x


# def _is_struct(x):
#     return isinstance(x, jax.ShapeDtypeStruct)


# def _default_process2(primitive, values, params):
#     values = tuple(x.materialise() for x in values)
#     subfuns, bind_params = primitive.get_bind_params(params)
#     ans = primitive.bind(*subfuns, *values, **bind_params)
#     if primitive.multiple_results:
#         return [DenseArrayValue(x) for x in ans]
#     else:
#         return DenseArrayValue(ans)


# def _safe_map(fn, *args):
#     args = [list(args) for args in args]
#     length = len(args[0])
#     assert all(len(arg) == length for arg in args[1:])
#     return list(map(fn, *args))


# class _Quaxify2(eqx.Module):
#     fn: Callable
#     unwrap_builtin_value: bool

#     @property
#     def __wrapped__(self):
#         return self.fn

#     def __call__(self, *args, **kwargs):
#         flat, treedef = jtu.tree_flatten((args, kwargs), is_leaf=_is_value)
#         flat = [_to_value(x) for x in flat]
#         flat_struct = [_to_struct(x) for x in flat]
#         dynamic_flat_struct, static_flat = eqx.partition(flat_struct, _is_struct)

#         def _fn(_dynamic_flat):
#             _flat = eqx.combine(_dynamic_flat, static_flat)
#             _args, _kwargs = jtu.tree_unflatten(treedef, _flat)
#             _out = self.fn(*_args, **_kwargs)
#             _out_flat, _out_treedef = jtu.tree_flatten(_out)
#             _dynamic_out_flat, _static_out_flat = eqx.partition(
#                 _out_flat, eqx.is_array
#             )
#             return _dynamic_out_flat, eqxi.Static((_out_treedef, _static_out_flat))

#         jaxpr, (_, static) = jax.make_jaxpr(_fn, return_shape=True)(
#             dynamic_flat_struct
#         )
#         consts = jaxpr.consts
#         jaxpr = jaxpr.jaxpr
#         out_treedef, static_out_flat = static.value

#         def read(v: core.Atom):
#             return v.val if isinstance(v, core.Literal) else env[v]

#         def write(v: core.Var, val: Value):
#             assert isinstance(val, Value)
#             assert core.raise_to_shaped(v.aval) == core.raise_to_shaped(val.aval())
#             env[v] = val

#         env: dict[core.Var, Value] = {}
#         consts = [DenseArrayValue(x) for x in consts]
#         dynamic_flat = [x for x in flat if _is_value(x)]
#         _safe_map(write, jaxpr.constvars, consts)
#         _safe_map(write, jaxpr.invars, dynamic_flat)
#         for eqn in jaxpr.eqns:
#             values = _safe_map(read, eqn.invars)
#             try:
#                 rule = _rules[eqn.primitive]
#             except KeyError:
#                 ans = _default_process2(eqn.primitive, values, eqn.params)
#             else:
#                 try:
#                     ans = rule(*values, **eqn.params)
#                 except plum.NotFoundLookupError:
#                     ans = _default_process2(eqn.primitive, values, eqn.params)
#             if eqn.primitive.multiple_results:
#                 _safe_map(write, eqn.outvars, ans)
#             else:
#                 [outvar] = eqn.outvars
#                 write(outvar, ans)
#         dynamic_out_flat = _safe_map(read, jaxpr.outvars)
#         if self.unwrap_builtin_value:
#             dynamic_out_flat = [x.array if isinstance(x, DenseArrayValue) else x
#                                 for x in dynamic_out_flat]
#         out_flat = eqx.combine(static_out_flat, dynamic_out_flat)
#         out = jtu.tree_unflatten(out_treedef, out_flat)
#         return out

#     def __get__(self, instance, owner):
#         if instance is None:
#             return self
#         return eqx.Partial(self, instance)
