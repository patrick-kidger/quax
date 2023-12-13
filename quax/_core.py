import abc
import functools as ft
import itertools as it
import operator
from collections.abc import Callable, Sequence
from typing import Any, Union

import equinox as eqx
import jax
import jax._src
import jax.core as core
import jax.extend.linear_util as lu
import jax.numpy as jnp
import jax.tree_util as jtu
import plum
from jax.custom_derivatives import SymbolicZero as SZ
from jaxtyping import ArrayLike


#
# Rules
#


_rules: dict[core.Primitive, Callable] = {}


def register(primitive: core.Primitive):
    """Registers a multiple dispatch implementation for this JAX primitive.

    Used as decorator, and requires type annotations to perform multiple dispatch:
    ```python
    @quax.register(jax.lax.add_p)
    def _(x: SomeValue, y: SomeValue):
        return ...  # some implementation
    ```
    All positional arguments will be (subclasses of) [`quax.Value`][] -- these are the
    set of types that Quax will attempt to perform multiple dispatch with.

    All keyword arguments will be the parameters for this primitive, as passed to
    `prim.bind(... **params)`.

    **Arguments:**

    - `primitive`: The `jax.core.Primitive` to provide a multiple dispatch
        implementation for.

    **Returns:**

    A decorator for registering a multiple dispatch rule with the specified primitive.
    """

    def _register(rule: Callable):
        try:
            existing_rule = _rules[primitive]  # pyright: ignore
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
    defaults = {type(x).default for x in values}
    raises = False
    if len(defaults) == 0:  # e.g. jax.debug.print("")
        default = Value.default
    elif len(defaults) == 1:
        [default] = defaults
    elif len(defaults) == 2:
        default1, default2 = defaults
        if default1 is Value.default:
            default = default2
        elif default2 is Value.default:
            default = default1
        else:
            raises = True
    else:
        raises = True
    if raises:
        types = {type(x) for x in values}
        raise TypeError(
            f"Multiple array-ish types {types} are specifying default process rules."
        )

    # Avoid an infinite loop, by pushing a new interpreter to the dynamic interpreter
    # stack.
    with jax.ensure_compile_time_eval():
        return default(primitive, values, params)  # pyright: ignore


class _QuaxTrace(core.Trace[_QuaxTracer]):
    def pure(self, val: Union[ArrayLike, "Value"]) -> _QuaxTracer:
        if not _is_value(val):
            val = DenseArrayValue(val)  # pyright: ignore
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
            return [_QuaxTracer(self, x) for x in out]  # pyright: ignore
        else:
            return _QuaxTracer(self, out)  # pyright: ignore

    def process_custom_jvp_call(self, primitive, fun, jvp, tracers, *, symbolic_zeros):
        in_values = [t.value for t in tracers]
        in_leaves, in_treedef = jtu.tree_flatten(in_values)
        fun, out_treedef1 = _custom_jvp_fun_wrap(fun, self.main, in_treedef)  # pyright: ignore
        jvp, out_treedef2 = _custom_jvp_jvp_wrap(jvp, self.main, in_treedef)  # pyright: ignore
        with jax.ensure_compile_time_eval():
            out_leaves = primitive.bind(
                fun, jvp, *in_leaves, symbolic_zeros=symbolic_zeros
            )
        _, out_treedef = lu.merge_linear_aux(out_treedef1, out_treedef2)
        out_values = jtu.tree_unflatten(out_treedef, out_leaves)
        return [_QuaxTracer(self, x) for x in out_values]

    # TODO: add other process_* rules


@lu.transformation_with_aux  # pyright: ignore
def _custom_jvp_fun_wrap(main, in_treedef, *in_leaves):
    trace = main.with_cur_sublevel()
    in_values = jtu.tree_unflatten(in_treedef, in_leaves)
    in_tracers = [x if type(x) is SZ else _QuaxTracer(trace, x) for x in in_values]
    out_tracers = yield in_tracers, {}
    # The symbolic zero branch here will actually create a `quax.zero.Zero`!
    out_tracers = [
        jnp.zeros(t.aval.shape, t.aval.dtype) if type(t) is SZ else t  # pyright: ignore
        for t in out_tracers
    ]
    out_values = [trace.full_raise(t).value for t in out_tracers]
    out_leaves, out_treedef = jtu.tree_flatten(out_values)
    yield out_leaves, out_treedef


@lu.transformation_with_aux  # pyright: ignore
def _custom_jvp_jvp_wrap(main, in_treedef, *in_primals_and_tangents):
    trace = main.with_cur_sublevel()
    in_primals = in_primals_and_tangents[: len(in_primals_and_tangents) // 2]
    in_tangents = in_primals_and_tangents[len(in_primals_and_tangents) // 2 :]
    in_primal_values = jtu.tree_unflatten(in_treedef, in_primals)
    in_tangent_values = jtu.tree_unflatten(in_treedef, in_tangents)
    in_tracers = [trace.pure(x) for x in it.chain(in_primal_values, in_tangent_values)]
    out_tracers = yield in_tracers, {}
    # The symbolic zero branch here will actually create a `quax.zero.Zero`!
    out_tracers = [
        jnp.zeros(t.aval.shape, t.aval.dtype) if type(t) is SZ else t  # pyright: ignore
        for t in out_tracers
    ]
    out_values = [trace.full_raise(t).value for t in out_tracers]
    out_primal_values = out_values[: len(out_values) // 2]
    out_tangent_values = out_values[len(out_values) // 2 :]
    out_primal_values2 = []
    out_tangent_values2 = []
    assert len(out_primal_values) == len(out_tangent_values)
    for primal, tangent in zip(out_primal_values, out_tangent_values):
        if primal.__class__ != tangent.__class__:
            primal = primal.materialise()
            tangent = tangent.materialise()
        out_primal_values2.append(primal)
        out_tangent_values2.append(tangent)
    out_primals, out_primal_treedef = jtu.tree_flatten(out_primal_values2)
    out_tangents, out_tangent_treedef = jtu.tree_flatten(out_tangent_values2)
    if out_primal_treedef != out_tangent_treedef:
        raise ValueError(
            "Primals and tangents had the same class, but different flattened results."
        )
    yield out_primals + out_tangents, out_primal_treedef


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
    """Quaxify's a function, so that it understands custom array-ish objects like
    `quax.lora.LoraArray`. When this function is called, multiple dispatch will be
    performed against these types.

    **Arguments:**

    - `fn`: the function to wrap.

    **Returns:**

    A copy of `fn`, that understands all Quax types.
    """
    return eqx.module_update_wrapper(
        _Quaxify(fn, unwrap_builtin_value=unwrap_builtin_value)
    )


quaxify_keepwrap = ft.partial(quaxify, unwrap_builtin_value=False)


#
# Values
#


class Value(eqx.Module):
    """Represents an object which Quax can perform multiple dispatch with.

    In practice you will probably want to inherit from [`quax.ArrayValue`][] instead,
    which represents specifically an array-like object that can be used for multiple
    dispatch. (It adds a number of methods like `__add__`, `.shape`, etc.)
    """

    @abc.abstractmethod
    def materialise(self) -> Any:
        """All concrete subclasses must implement this method, specifying how to
        materialise this object into any type that is understood by JAX. This is so that
        the usual JAX primitive implementations can be applied as a fallback: all
        objects are materialised, and then the usual implementation called on them.

        It is acceptable for this function to just raise an error -- in this case
        the error will be surfaced to the end user, indicating that an operation is
        not supported for this array-ish object.
        """

    @abc.abstractmethod
    def aval(self) -> core.AbstractValue:
        """All concrete subclasses must implement this method, specifying the abstract
        value seen by JAX.
        """

    @staticmethod
    def default(primitive, values, params) -> Union["Value", Sequence["Value"]]:
        """This is the default rule for when no rule has been `quax.register`'d for the
        primitive.

        This base implementation of `default` will be used if no subclass overrides this
        method.

        If there is precisely one override of this method (amongst all arguments to the
        primitive bind), then that implementation of `default` will be used.

        If there are multiple overrides of this method (due to multiple subclasses of
        Value appearing amongst the arguments of the primitive bind), then tracing will
        error. In this case a rule must be explicitly specified.
        """
        arrays = [x.materialise() for x in values]
        out = primitive.bind(*arrays, **params)
        if primitive.multiple_results:
            return [DenseArrayValue(x) for x in out]
        else:
            return DenseArrayValue(out)


def _is_value(x):
    return isinstance(x, Value)


def _flip_binop(binop):
    def _binop(x, y):
        return binop(y, x)

    return _binop


class ArrayValue(Value):
    """A [`quax.Value`][] for specifically array-like types. If you are creating a
    custom array-ish object then you should typically inherit from this.
    """

    @abc.abstractmethod
    def materialise(self) -> ArrayLike:
        """All concrete subclasses must implement this method, specifying how to
        materialise this object into a standard JAX array. This is so that the usual
        JAX primitive implementations can be applied as a fallback: all array-ish
        objects are materialised, and then the usual implementation called on them.

        It is acceptable for this function to just raise an error -- in this case
        the error will be surfaced to the end user, indicating that an operation is
        not supported for this array-ish object.
        """

    @abc.abstractmethod
    def aval(self) -> core.ShapedArray:
        """All concrete subclasses must implement this method, specifying the abstract
        value seen by JAX. The return must be a `jax.core.ShapedArray`.
        """

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
    """Internal type used to wrap up a JAX arraylike into Quax's `Value` system."""

    array: ArrayLike

    def materialise(self) -> ArrayLike:
        return self.array

    def aval(self) -> core.ShapedArray:
        return core.get_aval(self.array)  # pyright: ignore


@register(jax._src.pjit.pjit_p)  # pyright: ignore
def _(*args: ArrayValue, jaxpr, inline, **kwargs):
    del kwargs
    fun = quaxify_keepwrap(core.jaxpr_as_fun(jaxpr))
    if inline:
        return fun(*args)
    else:
        leaves, treedef = jtu.tree_flatten(args)  # remove all Values
        flat_fun = lambda x: fun(*jtu.tree_unflatten(treedef, x))
        with jax.ensure_compile_time_eval():  # replace the dynamic QuaxTrace
            return jax.jit(flat_fun)(leaves)  # now we can call without Quax.


# TODO: also register higher-order primitives like `lax.cond_p` etc.


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
