# Symbolic zeros, structured matrices, ...

Quax also includes several other example libraries.

These are deliberately not documented further here, as we have no intention of turning these into officially-supported fully-fledged Quax libraries.

However if you want to write your own Quax library then they exist so that you can take a look at their source code -- as a useful demonstration, or as a starting point.

- `quax.examples.named`: arrays with named axes.
- `quax.examples.prng`: PRNGs as array-ish values. (Rather than the special-cased `jax.random.key` you normally use.)
- `quax.examples.sparse`: sparse arrays as array-ish values. (Rather than the `jax.experimental.sparse` implementation.)
- `quax.examples.structured_matrices`: a tridiagonal matrix with an efficient matmul implementation.
- `quax.examples.zero`: symbolic zeros, so that e.g. `a + zero` immediately returns `a` during tracing, or so that `zero[:5]` returns a zero of a different shape.
