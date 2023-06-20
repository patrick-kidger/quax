import equinox as eqx
import jax.random as jr
import pytest

import quax


def test_init(getkey):
    Foo = quax.named.Axis("Foo", 3)
    Bar = quax.named.Axis("Bar", 3)
    a = jr.normal(getkey(), (3, 3))
    a = quax.named.NamedArray(a, (Foo, Bar))

    with pytest.raises(ValueError):
        a = jr.normal(getkey(), (3, 3))
        a = quax.named.NamedArray(a, (Foo, Foo))

    b = jr.normal(getkey(), (3, 4))
    with pytest.raises(ValueError):
        b = quax.named.NamedArray(b, (Foo, Bar))


def test_add(getkey):
    Foo = quax.named.Axis("Foo", 3)
    Bar = quax.named.Axis("Bar", 3)
    a = jr.normal(getkey(), (3, 3))
    a = quax.named.NamedArray(a, (Foo, Bar))
    b = jr.normal(getkey(), (3, 3))
    b = quax.named.NamedArray(b, (Foo, Bar))
    out1 = a + b
    out2 = quax.named.add(a, b)
    true_out = quax.named.NamedArray(a.array + b.array, (Foo, Bar))
    assert out1 == true_out
    assert out2 == true_out


def test_matmul(getkey):
    Foo = quax.named.Axis("Foo", 3)
    Bar = quax.named.Axis("Bar", 3)
    Qux = quax.named.Axis("Qux", 4)
    a = jr.normal(getkey(), (3, 3))
    a = quax.named.NamedArray(a, (Foo, Bar))
    b = jr.normal(getkey(), (3, 4))
    b = quax.named.NamedArray(b, (Bar, Qux))

    a @ b  # pyright: ignore
    quax.named.matmul(a, b)

    with pytest.raises(TypeError):
        b @ a  # pyright: ignore
    with pytest.raises(TypeError):
        quax.named.matmul(b, a)


def test_existing_function(getkey):
    # We can use NamedArrays even in functions that weren't designed for it! The output
    # will be a NamedArray as well!
    # In this case, eqx.nn.Linear.__call__ was not written expecting a NamedArray. It
    # works anyway :)

    # Existing program
    linear = eqx.nn.Linear(3, 4, key=getkey())

    # Wrap our desired inputs into NamedArrays
    In = quax.named.Axis("In", 3)
    Out = quax.named.Axis("Out", 4)
    named_bias = quax.named.NamedArray(linear.bias, (Out,))
    named_weight = quax.named.NamedArray(linear.weight, (Out, In))
    named_linear = eqx.tree_at(
        lambda l: (l.bias, l.weight), linear, (named_bias, named_weight)
    )
    vector = quax.named.NamedArray(jr.normal(getkey(), (3,)), (In,))

    # Wrap function with quaxify.
    out = quax.quaxify(named_linear)(vector)
    print(out)

    # Output is a NamedArray!
    true_out = quax.named.NamedArray(linear(vector.array), (Out,))
    assert out == true_out


# TODO: test the rest of the API!
