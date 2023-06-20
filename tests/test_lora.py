import equinox as eqx
import jax.lax as lax
import jax.random as jr

import quax


def test_linear(getkey):
    linear = eqx.nn.Linear(10, 12, key=getkey())
    lora_weight = quax.lora.LoraArray(linear.weight, rank=2, key=getkey())
    linear = eqx.tree_at(lambda l: l.weight, linear, lora_weight)
    vector = jr.normal(getkey(), (10,))
    quax.quaxify(linear)(vector)
    eqx.filter_jit(quax.quaxify(linear))(vector)


def test_loraify(getkey):
    mlp = eqx.nn.MLP(2, 2, 64, 3, key=getkey())
    mlp = quax.lora.loraify(mlp, rank=3, key=getkey())
    vector = jr.normal(getkey(), (2,))
    quax.quaxify(mlp)(vector)
    eqx.filter_jit(quax.quaxify(mlp))(vector)


def test_complicated_dot(getkey):
    lhs = jr.normal(getkey(), (3, 2, 1, 4, 5))
    rhs = jr.normal(getkey(), (8, 1, 5, 7, 3, 9, 10))
    lhs_contract = (4,)
    rhs_contract = (2,)
    lhs_batch = (2, 0)
    rhs_batch = (1, 4)
    dimension_numbers = ((lhs_contract, rhs_contract), (lhs_batch, rhs_batch))
    out = lax.dot_general(lhs, rhs, dimension_numbers)
    lhs_lora = quax.lora.LoraArray(lhs, rank=11, key=getkey())
    out_lora = quax.quaxify(lax.dot_general)(lhs_lora, rhs, dimension_numbers)
    assert out.shape == out_lora.shape

    dimension_numbers2 = ((rhs_contract, lhs_contract), (rhs_batch, lhs_batch))
    out2 = lax.dot_general(rhs, lhs, dimension_numbers2)
    out_lora2 = quax.quaxify(lax.dot_general)(rhs, lhs_lora, dimension_numbers2)
    assert out2.shape == out_lora2.shape
