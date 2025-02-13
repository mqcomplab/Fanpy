"""Test fanpy.sd_list."""

from fanpy.tools import sd_list

import pytest


def test_satisfies_conditions():
    """Test sd_list.satisfies_conditions."""
    assert sd_list.satisfies_conditions(0b1111, 2, None, None, None)

    assert not sd_list.satisfies_conditions(0b0001, 2, -1, None, None)
    assert not sd_list.satisfies_conditions(0b0001, 2, -0.5, None, None)
    assert not sd_list.satisfies_conditions(0b0001, 2, 0, None, None)
    assert sd_list.satisfies_conditions(0b0001, 2, 0.5, None, None)
    assert not sd_list.satisfies_conditions(0b0001, 2, 1, None, None)

    assert sd_list.satisfies_conditions(0b1100, 2, -1, None, None)
    assert not sd_list.satisfies_conditions(0b1100, 2, -0.5, None, None)
    assert not sd_list.satisfies_conditions(0b1100, 2, 0, None, None)
    assert not sd_list.satisfies_conditions(0b1100, 2, 0.5, None, None)
    assert not sd_list.satisfies_conditions(0b1100, 2, 1, None, None)

    assert sd_list.satisfies_conditions(0b1110, 2, None, 2, None)
    assert sd_list.satisfies_conditions(0b1110, 2, None, 1, None)
    assert not sd_list.satisfies_conditions(0b1110, 2, None, 0, None)

    assert sd_list.satisfies_conditions(0b1110, 2, -0.5, 1, None)
    assert not sd_list.satisfies_conditions(0b1110, 2, 0.5, 1, None)
    assert not sd_list.satisfies_conditions(0b1110, 2, -0.5, 0, None)


def test_ci_sd_list():
    """Test sd_list.sd_list."""
    with pytest.raises(TypeError):
        sd_list.sd_list(4.0, 6)
    with pytest.raises(TypeError):
        sd_list.sd_list(None, 6)
    with pytest.raises(TypeError):
        sd_list.sd_list("4", 6)
    with pytest.raises(TypeError):
        sd_list.sd_list(3, 4.0)
    with pytest.raises(TypeError):
        sd_list.sd_list(3, None)
    with pytest.raises(TypeError):
        sd_list.sd_list(3, "4")
    with pytest.raises(TypeError):
        sd_list.sd_list(4, 6, num_limit=3.0)
    with pytest.raises(TypeError):
        sd_list.sd_list(4, 6, num_limit="3")
    with pytest.raises(ValueError):
        sd_list.sd_list(4, 6, num_limit=0)
    with pytest.raises(TypeError):
        sd_list.sd_list(4, 6, exc_orders=3)
    with pytest.raises(TypeError):
        sd_list.sd_list(4, 6, exc_orders=[3.0])
    with pytest.raises(TypeError):
        sd_list.sd_list(4, 6, spin="3")
    with pytest.raises(TypeError):
        sd_list.sd_list(4, 6, seniority="3")
    with pytest.raises(ValueError):
        sd_list.sd_list(4, 6, spin=0.5, seniority=0)
    with pytest.raises(ValueError):
        sd_list.sd_list(4, 6, spin=-1, seniority=1)

    assert sd_list.sd_list(4, 6, num_limit=1) == [0b011011]
    assert sd_list.sd_list(4, 6, num_limit=2) == [0b011011, 0b011101]
    assert sd_list.sd_list(4, 6, num_limit=10, exc_orders=[1]) == [
        0b011011,
        0b011101,
        0b111001,
        0b001111,
        0b101011,
        0b011110,
        0b111010,
        0b010111,
        0b110011,
    ]
    assert sd_list.sd_list(4, 6, num_limit=10, exc_orders=[2]) == [
        0b011011,
        0b101101,
        0b111100,
        0b110101,
        0b101110,
        0b100111,
        0b110110,
    ]
    assert sd_list.sd_list(4, 6, exc_orders=[1, 2]) == [
        0b011011,
        0b011101,
        0b111001,
        0b001111,
        0b101011,
        0b011110,
        0b111010,
        0b010111,
        0b110011,
        0b101101,
        0b111100,
        0b110101,
        0b101110,
        0b100111,
        0b110110,
    ]
    assert sd_list.sd_list(4, 6, exc_orders=[2, 1]) == [
        0b011011,
        0b101101,
        0b111100,
        0b110101,
        0b101110,
        0b100111,
        0b110110,
        0b011101,
        0b111001,
        0b001111,
        0b101011,
        0b011110,
        0b111010,
        0b010111,
        0b110011,
    ]
    assert sd_list.sd_list(4, 6, exc_orders=[1, 2], spin=0) == [
        0b011011,
        0b011101,
        0b101011,
        0b011110,
        0b110011,
        0b101101,
        0b110101,
        0b101110,
        0b110110,
    ]
    assert sd_list.sd_list(4, 6, exc_orders=[1, 2], spin=0.5) == []
    assert sd_list.sd_list(4, 6, exc_orders=[1, 2], spin=-0.5) == []
    assert sd_list.sd_list(4, 6, exc_orders=[1, 2], spin=1) == [0b001111, 0b010111, 0b100111]
    assert sd_list.sd_list(4, 6, exc_orders=[1, 2], spin=-1) == [0b111001, 0b111010, 0b111100]


def test_doci_sd_list():
    """Test sd_list.doci_sd_list."""
    assert sd_list.sd_list(4, 8, num_limit=1, seniority=0) == [0b00110011]
    assert sd_list.sd_list(4, 8, num_limit=2, seniority=0) == [0b00110011, 0b01010101]
    assert sd_list.sd_list(4, 8, exc_orders=[2], seniority=0) == [
        0b00110011,
        0b01010101,
        0b10011001,
        0b01100110,
        0b10101010,
    ]
    assert sd_list.sd_list(4, 8, exc_orders=[4], seniority=0) == [0b00110011, 0b11001100]
    assert sd_list.sd_list(4, 8, exc_orders=[2, 4], seniority=0) == [
        0b00110011,
        0b01010101,
        0b10011001,
        0b01100110,
        0b10101010,
        0b11001100,
    ]
    assert sd_list.sd_list(4, 8, exc_orders=[4, 2], seniority=0) == [
        0b00110011,
        0b11001100,
        0b01010101,
        0b10011001,
        0b01100110,
        0b10101010,
    ]
