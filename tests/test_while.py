import jax
import jax.numpy as jnp

import quax
from quax.examples.unitful import Unitful, meters, kilograms


def outer_fn(a: jax.Array, b: jax.Array, c: jax.Array):
    # body has b as static argument
    def body_fn(a: jax.Array):
        return a + b

    # cond has c as static argument
    def cond_fn(a: jax.Array):
        return (a < c).squeeze()
    
    res = jax.lax.while_loop(
        body_fun=body_fn,
        cond_fun=cond_fn,
        init_val=a,
    )
    return res

def test_while_basic():
    a = Unitful(jnp.asarray(1.), meters)
    b = Unitful(jnp.asarray(2.), meters)
    c = Unitful(jnp.asarray(10.), meters)
    res = quax.quaxify(outer_fn)(a, b, c)
    assert res.array == 11
    assert res.units == {meters: 1}
    
def test_while_different_units():
    a = Unitful(jnp.asarray([1.]), meters)
    b = Unitful(jnp.asarray([2.]), meters)
    c = Unitful(jnp.asarray([10.]), kilograms)
    try:
        quax.quaxify(outer_fn)(a, b, c)
    except:
        assert True
    else:
        assert False
        
def test_while_jit():
    a = Unitful(jnp.asarray(1.), meters)
    b = Unitful(jnp.asarray(2.), meters)
    c = Unitful(jnp.asarray(10.), meters)
    res = quax.quaxify(jax.jit(outer_fn))(a, b, c)
    assert res.array == 11
    assert res.units == {meters: 1}

def test_while_vmap():
    a = Unitful(jnp.arange(1), meters)
    b = Unitful(jnp.asarray(2), meters)
    c = Unitful(jnp.arange(2, 13, 2), meters)
    vmap_fn = jax.vmap(outer_fn, in_axes=(None, None, 0))
    res = quax.quaxify(vmap_fn)(a, b, c)
    for i in range(len(c.array)):  # type: ignore
        assert res.array[i] == c.array[i]  # type: ignore
    assert res.units == {meters: 1}
