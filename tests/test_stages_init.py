"""Tests for quenchpath.stages' shared assert_mole_fraction()/
assert_number_density() guards - in particular assert_number_density()'s
optional `species=` check (added for GitHub issue #15: a genuine number
density cube for the *wrong* gas has correct units and still contains
"number density" in its name, so the bare check alone can't catch a
CH4/CO mixup - see quenchpath.stages.stage_07_taus's own
_load_chem_rate_cubes()).
"""

import pytest
from iris.cube import Cube

from quenchpath.stages import assert_mole_fraction, assert_number_density


def test_assert_number_density_passes_for_a_real_number_density():
    cube = Cube(1e10, long_name="CH4 number density", units="cm-3")
    assert_number_density(cube)  # must not raise


def test_assert_number_density_rejects_wrong_units():
    cube = Cube(0.5, long_name="CH4 number density", units="1")
    with pytest.raises(AssertionError):
        assert_number_density(cube)


def test_assert_number_density_rejects_wrong_name():
    cube = Cube(1e10, long_name="CH4 mole fraction", units="cm-3")
    with pytest.raises(AssertionError):
        assert_number_density(cube)


def test_assert_number_density_species_check_passes_for_the_right_gas():
    cube = Cube(1e10, long_name="CO number density", units="cm-3")
    assert_number_density(cube, species="CO")  # must not raise


def test_assert_number_density_species_check_catches_the_wrong_gas():
    # The actual bug this guards against: a genuine, correctly-unitsed
    # number density cube for the wrong species - passes the bare
    # units/name check, must still be caught here.
    cube = Cube(1e10, long_name="CO number density", units="cm-3")
    with pytest.raises(AssertionError, match="CH4 number density"):
        assert_number_density(cube, species="CH4")


def test_assert_number_density_species_none_is_the_default_and_skips_the_check():
    cube = Cube(1e10, long_name="CO number density", units="cm-3")
    assert_number_density(cube, species=None)  # must not raise


def test_assert_mole_fraction_passes_for_a_real_mole_fraction():
    cube = Cube(0.1, long_name="CH4 mole fraction", units="1")
    assert_mole_fraction(cube)  # must not raise


def test_assert_mole_fraction_rejects_a_number_density():
    cube = Cube(1e10, long_name="CH4 number density", units="cm-3")
    with pytest.raises(AssertionError):
        assert_mole_fraction(cube)
