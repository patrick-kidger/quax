from typing import cast

import jax
import jax.core
import jax.lax as lax
import jax.numpy as jnp
from jaxtyping import Array

import quax


def test_jit_inline():
    @quax.quaxify
    def f(x):
        return 2 * x + 1

    jaxpr = jax.make_jaxpr(f)(1.0)
    assert str(jaxpr).count("pjit") == 0


def test_default_override():
    records = []

    class Record(quax.ArrayValue):
        array: Array

        def materialise(self):
            assert False

        def aval(self):
            return cast(jax.core.ShapedArray, jax.core.get_aval(self.array))

        @staticmethod
        def default(primitive, values, params):
            arrays = [x.array if isinstance(x, Record) else x for x in values]
            records.append(primitive)
            out = quax.quaxify(primitive.bind)(*arrays, **params)
            if primitive.multiple_results:
                return [Record(x) for x in out]
            else:
                return Record(out)

    @quax.register(lax.mul_p)
    def _(a: Record, b: Record):
        return Record(a.array * b.array)

    x = Record(jnp.array(1.0))
    y = Record(jnp.array(2.0))
    z = Record(jnp.array(2.0))

    @quax.quaxify
    def f(a, b, c):
        return a + b * c

    f(x, y, z)
    assert records == [lax.add_p]
    records.clear()

    f(jnp.array(1.0), y, jnp.array(2.0))
    assert records == [lax.mul_p, lax.add_p]
