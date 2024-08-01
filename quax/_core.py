import abc
import functools as ft
import itertools as it
from collections.abc import Callable, Sequence
from typing import Any, cast, Generic, TypeVar, Union
from typing_extensions import TypeGuard

import equinox as eqx
import jax
import jax._src
import jax.core as core
import jax.extend.linear_util as lu
import jax.numpy as jnp
import jax.tree_util as jtu
import plum
from jax.custom_derivatives import SymbolicZero as SZ
from jaxtyping import ArrayLike, PyTree


CT = TypeVar("CT", bound=Callable)

#
# Rules
#


_rules: dict[core.Primitive, plum.Function] = {}


def register(primitive: core.Primitive, *, precedence: int = 0) -> Callable[[CT], CT]:
    """Registers a multiple dispatch implementation for this JAX primitive.

    !!! Example

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

    - `precedence`: The precedence of this rule.
        See `plum.Dispatcher.dispatch` for details.

    **Returns:**

    A decorator for registering a multiple dispatch rule with the specified primitive.
    """

    def _register(rule: CT) -> CT:
        try:
            existing_rule = _rules[primitive]  # pyright: ignore
        except KeyError:

            def existing_rule():
                assert False

            existing_rule.__name__ = f"{primitive}_dispatcher"
            existing_rule.__qualname__ = f"{primitive}_dispatcher"
            existing_rule = plum.Dispatcher().abstract(existing_rule)

            _rules[primitive] = existing_rule
        existing_rule.dispatch(rule, precedence=precedence)
        return rule

    return _register


#
# Interpreter
#


class _QuaxTracer(core.Tracer):
    __slots__ = ("value",)

    def __init__(self, trace: "_QuaxTrace", value: "Value") -> None:
        assert _is_value(value)
        self._trace = trace
        self.value = value

    @property
    def aval(self):
        return self.value.aval()

    def full_lower(self):
        if isinstance(self.value, _DenseArrayValue):
            return core.full_lower(self.value.array)
        else:
            return self


def _default_process(
    primitive: core.Primitive, values: Sequence[Union[ArrayLike, "Value"]], params
):
    defaults = set()
    for x in values:
        if isinstance(x, Value):
            x_default = type(x).default
            if x_default is Value.default:
                pass
            else:
                defaults.add(x_default)
        elif eqx.is_array_like(x):
            # Ignore any unwrapped _DenseArrayValues
            pass
        else:
            assert False
    if len(defaults) == 0:
        default = Value.default
    elif len(defaults) == 1:
        [default] = defaults
    else:
        types = {type(x) for x in values}
        raise TypeError(
            f"Multiple array-ish types {types} are specifying default process rules."
        )

    # Avoid an infinite loop, by pushing a new interpreter to the dynamic interpreter
    # stack.
    with jax.ensure_compile_time_eval():
        return default(primitive, values, params)  # pyright: ignore


def _wrap_if_array(x: Union[ArrayLike, "Value"]) -> "Value":
    if eqx.is_array_like(x):
        return _DenseArrayValue(cast(ArrayLike, x))
    else:
        return cast(Value, x)


class _QuaxTrace(core.Trace[_QuaxTracer]):
    def pure(self, val: ArrayLike) -> _QuaxTracer:
        if _is_value(val):
            raise TypeError(
                f"Encountered Quax value of type {type(val)}. These must be "
                "transformed by passing them across a `quax.quaxify` boundary before "
                "being used.\n"
                "For example, the following is incorrect, as `SomeValue()` is not "
                "explicitly passed across the API boundary:\n"
                "```\n"
                "def f(x):\n"
                "    return x + SomeValue()\n"
                "\n"
                "quax.quaxify(f)(AnotherValue())\n"
                "```\n"
                "This should instead be written as the following:\n"
                "explicitly passed across the API boundary:\n"
                "```\n"
                "def f(x, y):\n"
                "    return x + y\n"
                "\n"
                "quax.quaxify(f)(AnotherValue(), SomeValue())\n"
                "```\n"
                "To better understand this, remember that the purpose of Quax is "
                "take a JAX program (given as a function) that acts on arrays, and to "
                "instead run it with array-ish types. But in the first example above, "
                "the original program already has an array-ish type, even before the "
                "`quaxify` is introduced."
            )
        if not eqx.is_array_like(val):
            raise TypeError(f"{type(val)} is not a JAX type.")
        return _QuaxTracer(self, _DenseArrayValue(val))  # pyright: ignore

    def lift(self, tracer: core.Tracer) -> _QuaxTracer:
        return _QuaxTracer(self, _DenseArrayValue(tracer))

    def sublift(self, tracer: _QuaxTracer) -> _QuaxTracer:
        return tracer

    def process_primitive(self, primitive, tracers, params):
        values = [t.value for t in tracers]
        values = tuple(
            x.array if isinstance(x, _DenseArrayValue) else x for x in values
        )
        try:
            rule = _rules[primitive]
        except KeyError:
            out = _default_process(primitive, values, params)
        else:
            try:
                method, _ = rule.resolve_method(values)
            except plum.NotFoundLookupError:
                out = _default_process(primitive, values, params)
            else:
                out = method(*values, **params)
        if primitive.multiple_results:
            return [_QuaxTracer(self, _wrap_if_array(x)) for x in out]  # pyright: ignore
        else:
            return _QuaxTracer(self, _wrap_if_array(out))  # pyright: ignore

    def process_custom_jvp_call(self, primitive, fun, jvp, tracers, *, symbolic_zeros):
        in_values = [t.value for t in tracers]
        # Each `t.value` will be some `Value`, and thus a PyTree. Here we flatten the
        # `Value`-ness away.
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
    # Calling `_QuaxTracer` directly here, not using `trace.{pure,lift}` as each `x` is
    # a `Value`, not an array (=> pure) or tracer (=> lift).
    in_tracers = [
        _QuaxTracer(trace, x) for x in it.chain(in_primal_values, in_tangent_values)
    ]
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


def _unwrap_tracer(trace, x):
    if eqx.is_array_like(x):
        x = trace.full_raise(x)
    if isinstance(x, _QuaxTracer):
        if isinstance(x.value, _DenseArrayValue):
            return x.value.array
        else:
            return x.value
    else:
        return x


class _Quaxify(eqx.Module, Generic[CT]):
    fn: CT
    filter_spec: PyTree[Union[bool, Callable[[Any], bool]]]
    dynamic: bool = eqx.field(static=True)

    @property
    def __wrapped__(self) -> CT:
        return self.fn

    def __call__(self, *args, **kwargs):
        with core.new_main(_QuaxTrace, dynamic=self.dynamic) as main:
            trace = _QuaxTrace(main, core.cur_sublevel())
            # Note that we do *not* wrap arraylikes here. We let that happen in
            # `_QuaxTrace.{pure,lift}` as necessary. This means that we can do e.g.
            # quaxify(jnp.moveaxis)(array, source=0, destination=-1).
            dynamic, static = eqx.partition(
                (self.fn, args, kwargs), self.filter_spec, is_leaf=_is_value
            )
            dynamic = jtu.tree_map(
                ft.partial(_wrap_tracer, trace),
                dynamic,
                is_leaf=_is_value,
            )
            fn, args, kwargs = eqx.combine(dynamic, static)
            out = fn(*args, **kwargs)
            out = jtu.tree_map(ft.partial(_unwrap_tracer, trace), out)
            return out

    def __get__(self, instance: Union[object, None], owner: Any):
        if instance is None:
            return self
        return eqx.Partial(self, instance)


def quaxify(
    fn: CT,
    filter_spec: PyTree[Union[bool, Callable[[Any], bool]]] = True,
) -> _Quaxify[CT]:
    """'Quaxifies' a function, so that it understands custom array-ish objects like
    [`quax.examples.lora.LoraArray`][]. When this function is called, multiple dispatch
    will be performed against the types it is called with.

    **Arguments:**

    - `fn`: the function to wrap.
    - `filter_spec`: which arguments to quaxify. Advanced usage, see tip below.

    **Returns:**

    A copy of `fn`, that understands all Quax types.

    !!! Tip "Only quaxifying some argments"

        Calling `quax.quaxify(fn, filter_spec)(*args, **kwargs)` will under-the-hood run
        `dynamic, static = eqx.partition((fn, args, kwargs), filter_spec)`, and then
        only quaxify those arguments in `dynamic`. This allows for passing through some
        [`quax.Value`][]s into the function unchanged, typically so that they can hit a
        nested `quax.quaxify`. See the
        [advanced tutorial](../examples/redispatch.ipynb).
    """
    return cast(
        _Quaxify[CT],
        eqx.module_update_wrapper(_Quaxify(fn, filter_spec, dynamic=False)),
    )


#
# Values
#


class Value(eqx.Module):
    """Represents an object which Quax can perform multiple dispatch with.

    In practice you will almost always want to inherit from [`quax.ArrayValue`][]
    instead, which represents specifically an array-ish object that can be used for
    multiple dispatch.
    """

    @abc.abstractmethod
    def aval(self) -> core.AbstractValue:
        """All concrete subclasses must implement this method, specifying the abstract
        value seen by JAX.

        **Arguments:**

        Nothing.

        **Returns:**

        Any subclass of `jax.core.AbstractValue`. Typically a `jax.core.ShapedArray`.
        """

    @staticmethod
    def default(
        primitive: core.Primitive, values: Sequence[Union[ArrayLike, "Value"]], params
    ) -> Union[ArrayLike, "Value", Sequence[Union[ArrayLike, "Value"]]]:
        """This is the default rule for when no rule has been [`quax.register`][]'d for
        a primitive.

        When performing multiple dispatch `primitive.bind(value1, value2, value3)`,
        then:

        1. If there is a dispatch rule matching the types of `value1`, `value2`, and
            `value3`, then that will be used.
        2. If precisely one of the types of `value{1,2,3}` overloads this method, then
            that default rule will be used.
        3. If precisely zero of the types of `value{1,2,3}` overloads this method, then
            all values are [`quax.Value.materialise`][]d, and the usual JAX
            implementation is called.
        4. If multiple of the types of `value{1,2,3}` overload this method, then a
            trace-time error will be raised.

        **Arguments:**

        - `primitive`: the `jax.core.Primitive` being considered.
        - `values`: a sequence of what values this primitive is being called with. Each
            value can either be [`quax.Value`][]s, or a normal JAX arraylike (i.e.
            `bool`/`int`/`float`/`complex`/NumPy scalar/NumPy array/JAX array).
        - `params`: the keyword parameters to the primitive.

        **Returns:**

        The result of binding this primitive against these types. If
        `primitive.multiple_results is False` then this should be a single `quax.Value`
        or JAX arraylike. If `primitive.multiple_results is True`, then this should be
        a tuple/list of such values.

        !!! Example

            The default implementation discussed above performs the following:
            ```python
            @staticmethod
            def default(primitive, values, params):
                arrays = [x if equinox.is_array_like(x) else x.materialise()
                          for x in values]
                return primitive.bind(*arrays, **params)
            ```
            (Using the [Equinox](https://github.com/patrick-kidger/equinox) library that
            underlies much of the JAX ecosystem.)
        """

        arrays: list[ArrayLike] = []
        for x in values:
            if _is_value(x):
                arrays.append(x.materialise())
            elif eqx.is_array_like(x):
                arrays.append(cast(ArrayLike, x))
            else:
                assert False
        return primitive.bind(*arrays, **params)

    @abc.abstractmethod
    def materialise(self) -> Any:
        """All concrete subclasses must implement this method, specifying how to
        materialise this object into a JAX type (i.e. almost always a JAX array, unless
        you're doing something obscure using tokens or refs).

        !!! Example

            For example, a LoRA array consists of three arrays `(W, A, B)`, combined as
            `W + AB`. [`quax.examples.lora.LoraArray`] leaves these as three separate
            arrays for efficiency, but calling `lora_array.materialise()` will evaluate
            `W + AB` and return a normal JAX array.

        This is so that the usual JAX primitive implementations can be applied as a
        fallback: the array-ish object is materialised, and then the usual JAX
        implementation called on it. (See [`quax.Value.default`][].)

        !!! Info

            It is acceptable for this function to just raise an error -- in this case
            the error will be surfaced to the end user, indicating that an operation is
            not supported for this array-ish object.

        **Arguments:**

        Nothing.

        **Returns:**

        A JAX type; typically a JAX array.
        """


def _is_value(x) -> TypeGuard[Value]:
    return isinstance(x, Value)


class ArrayValue(Value):
    """A subclass [`quax.Value`][] for specifically array-like types. If you are
    creating a custom array-ish object then you should typically inherit from this.

    Provides the properties `.shape`, `.dtype`, `.ndim`, `.size`, each as a shortcut for
    `self.aval().shape` etc.
    """

    @abc.abstractmethod
    def materialise(self) -> ArrayLike:
        pass

    @abc.abstractmethod
    def aval(self) -> core.ShapedArray:
        pass

    @property
    def shape(self):
        return self.aval().shape

    @property
    def dtype(self):
        return self.aval().dtype

    @property
    def ndim(self):
        return self.aval().ndim

    @property
    def size(self):
        return self.aval().size


class _DenseArrayValue(ArrayValue):
    """Internal type used to wrap up a JAX arraylike into Quax's `Value` system.

    This is an implementation detail hidded from the user! It is unwrapped straight
    before calling a dispatch rule, and re-wrapped immediately afterwards.
    """

    array: ArrayLike

    def materialise(self) -> ArrayLike:
        return self.array

    def aval(self) -> core.ShapedArray:
        return core.get_aval(self.array)  # pyright: ignore


@register(jax._src.pjit.pjit_p)  # pyright: ignore
def _(*args: Union[ArrayLike, ArrayValue], jaxpr, inline, **kwargs):
    del kwargs
    fun = quaxify(core.jaxpr_as_fun(jaxpr))
    if inline:
        return fun(*args)
    else:
        leaves, treedef = jtu.tree_flatten(args)  # remove all Values
        flat_fun = lambda x: fun(*jtu.tree_unflatten(treedef, x))
        with jax.ensure_compile_time_eval():  # replace the dynamic QuaxTrace
            return jax.jit(flat_fun)(leaves)  # now we can call without Quax.


@register(jax.lax.while_p)
def _(
    *args: Union[ArrayValue, ArrayLike],
    cond_nconsts: int,
    cond_jaxpr,
    body_nconsts: int,
    body_jaxpr,
):
    cond_consts = args[:cond_nconsts]
    body_consts = args[cond_nconsts : cond_nconsts + body_nconsts]
    init_vals = args[cond_nconsts + body_nconsts :]

    # compute jaxpr of quaxified body and condition function
    quax_cond_fn = quaxify(core.jaxpr_as_fun(cond_jaxpr))
    quax_cond_jaxpr = jax.make_jaxpr(quax_cond_fn)(*cond_consts, *init_vals)
    quax_body_fn = quaxify(core.jaxpr_as_fun(body_jaxpr))
    quax_body_jaxpr = jax.make_jaxpr(quax_body_fn)(*body_consts, *init_vals)

    cond_leaves, _ = jtu.tree_flatten(cond_consts)
    body_leaves, _ = jtu.tree_flatten(body_consts)
    init_val_leaves, val_treedef = jtu.tree_flatten(init_vals)

    out_val = jax.lax.while_p.bind(
        *cond_leaves,
        *body_leaves,
        *init_val_leaves,
        cond_nconsts=cond_nconsts,
        cond_jaxpr=quax_cond_jaxpr,
        body_nconsts=body_nconsts,
        body_jaxpr=quax_body_jaxpr,
    )
    result = jtu.tree_unflatten(val_treedef, out_val)
    return result


# TODO: also register higher-order primitives like `lax.cond_p` etc.
