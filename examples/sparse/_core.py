import equinox as eqx
import jax.core
import jax.lax as lax
import jax.numpy as jnp
import quax
from jaxtyping import Array, Shaped


class BCOO(quax.ArrayValue):
    """Represents a batch-coordinate format array."""

    data: Shaped[Array, "*batch nse"]
    indices: Shaped[Array, "*batch nse n_sparse"]
    shape: tuple[int, ...] = eqx.field(static=True)  # pyright: ignore
    allow_materialise: bool = eqx.field(default=False, static=True)

    def __post_init__(self):
        *batch1, nse1 = self.data.shape
        *batch2, nse2, n_sparse = self.indices.shape
        batch3 = self.shape[: len(batch1)]
        batches = {tuple(batch1), tuple(batch2), tuple(batch3)}
        if len(batches) != 1:
            raise ValueError(f"Inconsistent batch sizes {batches} in `BCOO`.")
        if nse1 != nse2:
            raise ValueError("Inconsistent number of sparse entries in `BCOO`.")
        if len(self.shape[len(batch1) :]) != n_sparse:
            raise ValueError(
                "Must specify a sparse index for all non-batch dimensions in `BCOO`."
            )

    def aval(self):
        return jax.core.ShapedArray(self.shape, self.data.dtype)

    def materialise(self):
        if self.allow_materialise:
            zeros = jnp.zeros(self.shape, self.data.dtype)
            add = lambda z, i, d: z.at[i].add(d)
            return _op_sparse_to_dense(self, zeros, add)
        else:
            raise RuntimeError(
                "Refusing to materialise sparse matrix with `allow_materialise=False`."
            )


def _op_sparse_to_dense(x, y, op):
    indices = tuple(jnp.moveaxis(x.indices, -1, 0))
    for _ in range(x.data.ndim - 1):
        op = jax.vmap(op)
    return op(y, indices, x.data)


@quax.register(lax.add_p)
def _(x: BCOO, y: BCOO):
    assert x.shape == y.shape
    data = jnp.concatenate([x.data, y.data], axis=-1)
    indices = jnp.concatenate([x.indices, y.indices], axis=-2)
    allow_materialise = x.allow_materialise and y.allow_materialise
    return BCOO(data, indices, x.shape, allow_materialise)


@quax.register(lax.add_p)
def _add_bcoo_dense(x: BCOO, y: quax.DenseArrayValue) -> quax.DenseArrayValue:
    y_array = jnp.asarray(y.array)
    add = lambda z, i, d: z.at[i].add(d)
    return quax.DenseArrayValue(_op_sparse_to_dense(x, y_array, add))


@quax.register(lax.add_p)
def _(x: quax.DenseArrayValue, y: BCOO) -> quax.DenseArrayValue:
    return _add_bcoo_dense(y, x)


@quax.register(lax.mul_p)
def _(x: BCOO, y: BCOO):
    # This is actually surprisingly hard.
    raise NotImplementedError(
        "elementwise multiplication between two sparse matrices is not implemented"
    )


@quax.register(lax.mul_p)
def _mul_bcoo_dense(x: BCOO, y: quax.DenseArrayValue) -> BCOO:
    indices = tuple(jnp.moveaxis(x.indices, -1, 0))
    data = x.data * jnp.asarray(y.array)[indices]
    return BCOO(data, x.indices, x.shape, x.allow_materialise)


@quax.register(lax.mul_p)
def _(x: quax.DenseArrayValue, y: BCOO) -> BCOO:
    return _mul_bcoo_dense(y, x)
