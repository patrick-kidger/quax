import importlib.metadata

from . import examples as examples
from ._core import (
    ArrayValue as ArrayValue,
    quaxify as quaxify,
    register as register,
    Value as Value,
)

lora = examples.lora  # backward compatibility
zero = examples.zero  # backward compatibility


__version__ = importlib.metadata.version("quax")
