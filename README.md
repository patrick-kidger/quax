# Quax

*Significantly inspired by https://github.com/davisyoshida/qax, https://github.com/stanford-crfm/levanter, and `jax.experimental.sparse`.*  
*Right now this is wildly experimental -- this is me sharing an in-progress research project, not releasing one of my more-polished libraries.*  
*The LoRA, named arrays, and PRNGs are all lightly tested. The other use-cases are untested.*

Uses JAX's nonstandard interpretation to handle array-ish objects, like:

- LoRA weight matrices
- arrays with named dimensions
- symbolic zeros
- structured (e.g. tridiagonal) matrices
- sparse arrays
- PRNG keys
- quantisation?
- bounded dynamism?

...and if desired perform multiple dispatch on them!

Moreover, this works even with existing programs, that were not written to accept such array-ish objects! :D

Furthermore both sparse arrays and PRNG keys are use-cases already present in core JAX, so it looks like we might have a solution that is of interest to core JAX as well.

## Examples

### LoRA

```python
import equinox as eqx
import jax.random as jr
import quax

# Start off with any JAX program
key1, key2, key3 = jr.split(jr.PRNGKey(0), 3)
linear = eqx.nn.Linear(10, 12, key=key1)
vector = jr.normal(key2, (10,))

# Make one of the inputs be an array-ish object
# (Incidentally, there's a quax.lora.loraify function to do this automatically.)
lora_weight = quax.lora.LoraArray(linear.weight, rank=2, key=key3)
lora_linear = eqx.tree_at(lambda l: l.weight, linear, lora_weight)

# Wrap your function call in quaxify
out = quax.quaxify(lora_linear)(vector)
```

### Named arrays

```python
import equinox as eqx
import jax.random as jr
import quax

# Existing program
linear = eqx.nn.Linear(3, 4, key=jr.PRNGKey(0))

# Wrap our desired inputs into NamedArrays
In = quax.named.Axis("In", 3)
Out = quax.named.Axis("Out", 4)
named_bias = quax.named.NamedArray(linear.bias, (Out,))
named_weight = quax.named.NamedArray(linear.weight, (Out, In))
named_linear = eqx.tree_at(lambda l: (l.bias, l.weight), linear, (named_bias, named_weight))
vector = quax.named.NamedArray(jr.normal(jr.PRNGKey(1), (3,)), (In,))

# Wrap function with quaxify. Output will be a NamedArray!
out = quax.quaxify(named_linear)(vector)
print(out)  # NamedArray(array=f32[4], axes=(Axis(name='Out', size=4),))
```

### PRNGs

```python
import jax
import jax.lax as lax
import jax.numpy as jnp
import quax

# `key` is a PyTree wrapping a u32[2] array.
key = quax.prng.ThreeFry(0)
quax.prng.normal(key)

# Some primitives (lax.add_p) are disallowed.
key + 1  # TypeError!
quax.quaxify(lax.add)(key, 1)  # TypeError!

# Some primitives (lax.select_n) are allowed.
# We're calling `jnp.where(..., pytree1, pytree2)` -- on pytrees, not arrays!
pred = jnp.array(True)
key2 = quax.prng.ThreeFry(1)

@jax.jit
@quax.quaxify
def run(pred, key1, key2):
    return jnp.where(pred, key1, key2)

run(pred, key, key2)
```

## Speculation

I think we could use this to do all kinds of crazy things.

For example, write a nondeterministic Turing machine -- overload `lax.cond_p` to evaluate both branches, then store both of their outputs?
