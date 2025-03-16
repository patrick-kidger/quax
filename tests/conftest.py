import equinox.internal as eqxi
import pytest


@pytest.fixture()
def getkey():
    return eqxi.GetKey()
