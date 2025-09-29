"""Test for fanpy.tools.performance."""

from fanpy.tools.performance import current_memory

def test_current_memory():
    mem = current_memory()
    # current memory should be a positive float
    assert isinstance(mem, float)
    assert mem > 0