import importlib.metadata

from . import (
    lora as lora,
    named as named,
    prng as prng,
    sparse as sparse,
    structured_matrices as structured_matrices,
    zero as zero,
)
from .core import (
    ArrayValue as ArrayValue,
    DenseArrayValue as DenseArrayValue,
    quaxify as quaxify,
    register as register,
    Value as Value,
)


__version__ = importlib.metadata.version("quax")
