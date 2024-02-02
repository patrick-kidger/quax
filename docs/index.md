# Quax

JAX + multiple dispatch + custom array-ish objects.

!!! Example

    For example, this can be mean overloading matrix multiplication to exploit sparsity or structure, or automatically rewriting a LoRA's matmul `(W + AB)v` into the more-efficient `Wv + ABv`.

Applications include:

- LoRA weight matrices
- symbolic zeros
- arrays with named dimensions
- structured (e.g. tridiagonal) matrices
- sparse arrays
- quantised arrays
- arrays with physical units attached
- etc! (See the built-in `quax.examples` library for most of the above!)

This works via a custom JAX transform. Take an existing JAX program, wrap it in a `quax.quaxify`, and then pass in the custom array-ish objects. This means it will work even with existing programs, that were not written to accept such array-ish objects!

_(Just like how `jax.vmap` takes a program, but reinterprets each operation as its batched version, so to will `quax.quaxify` take a program and reinterpret each operation according to what array-ish types are passed.)_

## Installation

```
pip install quax
```

## Getting started

To use the built-in LoRA library, check out the [`quax`](./api/quax.md) and [`quax.examples.lora`](./api/lora.md) pages in the left bar.

To start writing your own library (with your own array-ish type) using Quax, then check out the [custom type tutorial](./examples/custom_rules.ipynb).

## Example: LoRA

This example demonstrates everything you need to use the built-in `quax.examples.lora` library.

--8<-- ".lora-example.md"

## See also: other libraries in the JAX ecosystem

[Equinox](https://github.com/patrick-kidger/equinox): neural networks.

[jaxtyping](https://github.com/patrick-kidger/jaxtyping): type annotations for shape/dtype of arrays.

[Optax](https://github.com/deepmind/optax): first-order gradient (SGD, Adam, ...) optimisers.

[Diffrax](https://github.com/patrick-kidger/diffrax): numerical differential equation solvers.

[Optimistix](https://github.com/patrick-kidger/optimistix): root finding, minimisation, fixed points, and least squares.

[Lineax](https://github.com/patrick-kidger/lineax): linear solvers.

[BlackJAX](https://github.com/blackjax-devs/blackjax): probabilistic+Bayesian sampling.

[Orbax](https://github.com/google/orbax): checkpointing (async/multi-host/multi-device).

[sympy2jax](https://github.com/patrick-kidger/sympy2jax): SymPy<->JAX conversion; train symbolic expressions via gradient descent.

[Eqxvision](https://github.com/paganpasta/eqxvision): computer vision models.

[Levanter](https://github.com/stanford-crfm/levanter): scalable+reliable training of foundation models (e.g. LLMs).

[PySR](https://github.com/milesCranmer/PySR): symbolic regression. (Non-JAX honourable mention!)
