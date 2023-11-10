# Quax

Uses JAX's nonstandard interpretation to perform multiple dispatch on custom array-ish objects, like:

- LoRA weight matrices
- symbolic zeros
- arrays with named dimensions
- structured (e.g. tridiagonal) matrices
- sparse arrays
- etc!

This works via a custom JAX transform. This means that it works even with existing programs, that were not written to accept such array-ish objects: just wrap the program in the `quaxify` transform. A typical use-case is applying LoRA to fine-tune arbitrary models.

Implementations for **LoRA** and **symbolic zeros** are both already built-in to Quax. You can also create your own types and rules very easily; see the [examples library](https://github.com/patrick-kidger/quax/tree/main/examples) for demo implementations for **named arrays**, **sparse arrays**, **tridiagonal matrices**, and **PRNG keys**.

## Installation

```
pip install quax
```

## Example: LoRA

```python
import equinox as eqx
import jax.random as jr
import quax

# Start off with any JAX program: here, the forward pass through a linear layer.
key1, key2, key3 = jr.split(jr.PRNGKey(0), 3)
linear = eqx.nn.Linear(10, 12, key=key1)
vector = jr.normal(key2, (10,))

# Make some of the inputs be an array-ish object. This function finds all
# `eqx.nn.Linear` layers, and wraps their weights in `LoraArray`s.
lora_linear = quax.lora.loraify(linear, rank=2, key=key3)
# For this simple model, we could also do it manually.
lora_weight = quax.lora.LoraArray(linear.weight, rank=2, key=key3)
lora_linear = eqx.tree_at(lambda l: l.weight, linear, lora_weight)

# Wrap your function call in quaxify. This transform calls your original function,
# whilst looking up any multiple dispatch rules registered for any custom array-ish
# objects.
out = quax.quaxify(lora_linear)(vector)
```

## Work in progress!

This library is a work in progress! Right now it should support enough to run LoRA on common models. However, some operations (e.g. `jax.lax.cond_p`) are not yet supported. If you attempt to use these then an error will be thrown whilst tracing your program.

If you find yourself hitting any of these, then go ahead and open an issue, and/or a pull request!

## See also: other libraries in the JAX ecosystem

[Equinox](https://github.com/patrick-kidger/equinox): neural networks.

[jaxtyping](https://github.com/google/jaxtyping): type annotations for shape/dtype of arrays.

[Optax](https://github.com/deepmind/optax): first-order gradient (SGD, Adam, ...) optimisers.

[Diffrax](https://github.com/patrick-kidger/diffrax): numerical differential equation solvers.

[Optimistix](https://github.com/patrick-kidger/optimistix): root finding, minimisation, fixed points, and least squares.

[Lineax](https://github.com/google/lineax): linear solvers.

[BlackJAX](https://github.com/blackjax-devs/blackjax): probabilistic+Bayesian sampling.

[Orbax](https://github.com/google/orbax): checkpointing (async/multi-host/multi-device).

[sympy2jax](https://github.com/google/sympy2jax): SymPy<->JAX conversion; train symbolic expressions via gradient descent.

[Eqxvision](https://github.com/paganpasta/eqxvision): computer vision models.

[Levanter](https://github.com/stanford-crfm/levanter): scalable+reliable training of foundation models (e.g. LLMs).

[PySR](https://github.com/milesCranmer/PySR): symbolic regression. (Non-JAX honourable mention!)

## Acknowledgements

Significantly inspired by https://github.com/davisyoshida/qax, https://github.com/stanford-crfm/levanter, and `jax.experimental.sparse`.
