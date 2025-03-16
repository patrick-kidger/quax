import jax
import pytest


class GetKey:
    def __call__(self):
        return jax.random.key(0)


@pytest.fixture()
def getkey():
    return GetKey()
