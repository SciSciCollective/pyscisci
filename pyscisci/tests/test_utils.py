from pyscisci.utils import load_int


def test_load_int():
    assert load_int(1) == 1
    assert load_int("") is None
    assert load_int("x") is None
