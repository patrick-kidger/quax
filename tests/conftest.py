import jax.random as jr
import pytest


@pytest.fixture()
def getkey():
    return lambda: jr.PRNGKey(0)
