import equinox as eqx
import jax.numpy as jnp
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


def test_stop_gradient(getkey):
    mlp = eqx.nn.MLP(2, 2, 64, 3, key=getkey())
    mlp_true = quax.lora.loraify(mlp, rank=3, stop_gradient=True, key=getkey())
    mlp_false = quax.lora.loraify(mlp, rank=3, stop_gradient=False, key=getkey())
    vector = jr.normal(getkey(), (2,))

    @eqx.filter_jit
    @eqx.filter_grad
    @quax.quaxify
    def run1(mlp, vector):
        return jnp.sum(mlp(vector))

    @eqx.filter_jit
    @quax.quaxify
    @eqx.filter_grad
    def run2(mlp, vector):
        return jnp.sum(mlp(vector))

    # Not efficient!
    @quax.quaxify
    @eqx.filter_jit
    @eqx.filter_grad
    def run3(mlp, vector):
        return jnp.sum(mlp(vector))

    grad1true = run1(mlp_true, vector)
    grad2true = run2(mlp_true, vector)
    grad3true = run3(mlp_true, vector)

    grad1false = run1(mlp_false, vector)
    grad2false = run2(mlp_false, vector)
    grad3false = run3(mlp_false, vector)

    for grad in (grad1true, grad2true, grad3true):
        assert (grad.layers[1].weight.w  == 0).all()
        assert not (grad.layers[1].weight.a  == 0).all()
        assert not (grad.layers[1].weight.b  == 0).all()
        assert not (grad.layers[1].bias  == 0).all()

    for grad in (grad1false, grad2false, grad3false):
        assert not (grad.layers[1].weight.w  == 0).all()
        assert not (grad.layers[1].weight.a  == 0).all()
        assert not (grad.layers[1].weight.b  == 0).all()
        assert not (grad.layers[1].bias  == 0).all()
