import equinox as eqx
import jax.core
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from jaxtyping import Array, PRNGKeyArray, PyTree, Shaped

from ..core import ArrayValue, quaxify_keepwrap, register


class LoraArray(ArrayValue):
    w: Shaped[Array, "*batch x y"]
    a: Shaped[Array, "*batch x z"]
    b: Shaped[Array, "*batch z y"]
    allow_materialise: bool = eqx.field(static=True)

    def __init__(
        self,
        weight: Shaped[Array, "*batch x y"],
        rank: int,
        scale: float = 0.01,
        allow_materialise: bool = False,
        *,
        key: PRNGKeyArray
    ):
        *batch, x, y = weight.shape
        self.w = weight
        self.a = jr.normal(key, (*batch, x, rank), dtype=weight.dtype) * scale
        self.b = jnp.zeros((*batch, rank, y), dtype=weight.dtype)
        self.allow_materialise = allow_materialise

    def materialise(self):
        if self.allow_materialise:
            batch = tuple(range(self.w.ndim - 2))
            lhs_contract = (self.a.ndim - 1,)
            rhs_contract = (self.b.ndim - 2,)
            dimension_numbers = ((lhs_contract, rhs_contract), (batch, batch))
            return self.w + lax.dot_general(self.a, self.b, dimension_numbers)
        else:
            raise RuntimeError(
                "Refusing to materialise `LoraArray` with `allow_materialise=False`."
            )

    def aval(self):
        return jax.core.ShapedArray(self.w.shape, self.w.dtype)


def _is_linear(x):
    return isinstance(x, eqx.nn.Linear)


def loraify(
    model: PyTree,
    rank: int,
    scale: float = 0.01,
    allow_materialise: bool = False,
    *,
    key: PRNGKeyArray
) -> PyTree:
    def _loraify(x):
        nonlocal key
        if _is_linear(x):
            key, subkey = jr.split(key)
            lora_weight = LoraArray(
                x.weight, rank, scale, allow_materialise, key=subkey
            )
            return eqx.tree_at(lambda l: l.weight, x, lora_weight)
        else:
            return x

    return jtu.tree_map(_loraify, model, is_leaf=_is_linear)


@quaxify_keepwrap
def _lora_arrayr_matmul_impl(w, a, b, rhs, lhs_batch, ndim, dimension_numbers, kwargs):
    n_sharedbatch = len(lhs_batch)  # = len(rhs_batch)
    # All of the lora batch dimensions that aren't a dot_general batch dimension.
    n_lorabatch = ndim - n_sharedbatch - 2
    assert n_lorabatch >= 0
    out1 = lax.dot_general(w, rhs, dimension_numbers, **kwargs)
    # out1 has shape (*sharedbatch, *lorabatch, x, *otherbatch)
    out2 = lax.dot_general(b, rhs, dimension_numbers, **kwargs)
    # out2 has shape(*sharedbatch, *lorabatch, z, *otherbatch)
    lhs_contract2 = (w.ndim - 1,)
    rhs_contract2 = (n_sharedbatch + n_lorabatch,)
    rhs_batch2 = tuple(range(n_sharedbatch + n_lorabatch))
    lhs_batch2 = lhs_batch + tuple(i for i in rhs_batch2 if i not in lhs_batch)
    dimension_numbers2 = ((lhs_contract2, rhs_contract2), (lhs_batch2, rhs_batch2))
    out3 = lax.dot_general(a, out2, dimension_numbers2, **kwargs)
    # out3 has shape (*sharedbatch, *lorabatch, x, *otherbatch)
    return out1 + out3


@register(lax.dot_general_p)
def _lora_array_matmul(
    lhs: LoraArray, rhs: ArrayValue, *, dimension_numbers, **kwargs
) -> ArrayValue:
    ((lhs_contract, rhs_contract), (lhs_batch, rhs_batch)) = dimension_numbers
    [ndim] = {lhs.a.ndim, lhs.b.ndim, lhs.w.ndim}
    if lhs_contract == (ndim - 1,) and (ndim - 2 not in lhs_batch):
        return _lora_arrayr_matmul_impl(
            lhs.w, lhs.a, lhs.b, rhs, lhs_batch, ndim, dimension_numbers, kwargs
        )
    else:
        return quaxify_keepwrap(lax.dot_general)(
            lhs.materialise(), rhs, dimension_numbers, **kwargs
        )


@register(lax.dot_general_p)
def _(lhs: ArrayValue, rhs: LoraArray, *, dimension_numbers, **kwargs) -> ArrayValue:
    ((lhs_contract, rhs_contract), (lhs_batch, rhs_batch)) = dimension_numbers
    dimension_numbers_flipped = ((rhs_contract, lhs_contract), (rhs_batch, lhs_batch))
    out = _lora_array_matmul(
        rhs, lhs, dimension_numbers=dimension_numbers_flipped, **kwargs
    )
    # out has shape (*sharedbatch, *rhs_uncontracted, *lhs_uncontracted)
    n_sharedbatch = len(lhs_batch)
    n_rhs_uncontracted = rhs.aval().ndim - len(rhs_contract) - len(rhs_batch)
    src = tuple(range(n_sharedbatch, n_sharedbatch + n_rhs_uncontracted))
    dest = tuple(range(-n_rhs_uncontracted, 0))
    return quaxify_keepwrap(jnp.moveaxis)(out, src, dest)
