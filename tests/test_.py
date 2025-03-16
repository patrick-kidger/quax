from typing import cast

import equinox as eqx
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jr
import pytest
from jaxtyping import Array, TypeCheckError
from plum import NotFoundLookupError

import quax
import quax.examples.lora as lora
import quax.examples.named as named


def test_trigger():
    """Test triggering jax.errors.UnexpectedTracerError."""

    # mlp = eqx.nn.MLP(2, 2, 64, 3, activation=jax.nn.softplus, key=jr.key(0))
    # mlp = lora.loraify(mlp, rank=3, key=jr.key(0))
    # vector = jr.normal(jr.key(0), (2,))

    # @eqx.filter_jit
    # @quax.quaxify
    # @eqx.filter_grad
    # def run2(mlp, vector):
    #     return jnp.sum(mlp(vector))

    # run2(mlp, vector)

    # -------------------------------------------

    with pytest.raises(RuntimeError, match="Refusing to materialise"):
        x_false = lora.LoraArray(
            jr.normal(jr.key(0), (3, 3)), rank=2, allow_materialise=False, key=jr.key(1)
        )

        _ = quax.quaxify(jax.nn.relu)(x_false)

    # -------------------------------------------

    x = jnp.arange(4.0).reshape(2, 2)
    y = lora.LoraArray(x, rank=1, key=jr.key(0))

    def f(x):
        return jax.lax.add_p.bind(x, y)

    # Error type depends on whether jaxtyping is on
    with pytest.raises((TypeCheckError, NotFoundLookupError)):
        _ = quax.quaxify(f)(y)

    # -------------------------------------------

    b = jr.normal(jr.key(0), (3, 4))  # noqa: F841

    # -------------------------------------------

    Foo = named.Axis(3)
    Bar = named.Axis(3)
    a = named.NamedArray(jr.normal(jr.key(0), (3, 3)), (Foo, Bar))
    b = named.NamedArray(jr.normal(jr.key(0), (3, 3)), (Foo, Bar))
    out1 = quax.quaxify(lambda x, y: x + y)(a, b)
    out2 = quax.quaxify(lax.add)(a, b)
    true_out = named.NamedArray(a.array + b.array, (Foo, Bar))
    assert out1 == true_out
    assert out2 == true_out

    # -------------------------------------------

    Foo = named.Axis(3)
    Bar = named.Axis(3)
    Qux = named.Axis(None)
    a = named.NamedArray(jr.normal(jr.key(0), (3, 3)), (Foo, Bar))
    b = named.NamedArray(jr.normal(jr.key(0), (3, 3)), (Bar, Qux))

    quax.quaxify(lambda x, y: x @ y)(a, b)
    quax.quaxify(jnp.matmul)(a, b)

    match = "Cannot contract mismatched dimensions"
    with pytest.raises(TypeError, match=match):
        quax.quaxify(lambda x, y: x @ y)(b, a)
    match = "Cannot contract mismatched dimensions"
    with pytest.raises(TypeError, match=match):
        quax.quaxify(jnp.matmul)(b, a)

    # Existing program
    linear = eqx.nn.Linear(3, 4, key=jr.key(0))

    # Wrap our desired inputs into NamedArrays
    In = named.Axis(3)
    Out = named.Axis(4)
    named_bias = named.NamedArray(cast(Array, linear.bias), (Out,))
    named_weight = named.NamedArray(linear.weight, (Out, In))
    named_linear = eqx.tree_at(
        lambda l: (l.bias, l.weight), linear, (named_bias, named_weight)
    )
    vector = named.NamedArray(jr.normal(jr.key(0), (3,)), (In,))

    # Wrap function with quaxify.
    out = quax.quaxify(named_linear)(vector)
    # Output is a NamedArray!
    true_out = named.NamedArray(linear(vector.array), (Out,))
    assert out == true_out

    # -------------------------------------------

    A = named.Axis(None)
    B = named.Axis(None)
    C = named.Axis(None)

    with jax.checking_leaks():
        x = jr.normal(jr.key(0), (2, 3, 4))

    named_x = named.NamedArray(x, (A, B, C))
    out = named.trace(named_x, axis1=A, axis2=C)
    true_out = jnp.trace(x, axis1=0, axis2=2)
    assert out.axes == (B,)
    assert jnp.array_equal(out.enable_materialise().materialise(), true_out)
