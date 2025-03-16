import pytest


@pytest.fixture()
def getkey():
    import equinox.internal as eqxi

    return eqxi.GetKey()
