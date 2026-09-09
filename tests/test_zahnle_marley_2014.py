"""Unit tests for quenchpath.analysis.zahnle_marley_2014 - Zahnle &
Marley (2014) Eqs 12-14's t_CO fit, and its own calibration-envelope
mask. Expected values computed independently via the `math` module (see
tests/test_chemistry_formulas.py's own module docstring for why this
matters), not by calling back into
quenchpath.analysis.zahnle_marley_2014 itself.
"""

import math

import numpy as np
import pytest
from iris.cube import Cube

from quenchpath.analysis.zahnle_marley_2014 import (
    zahnle_marley_2014_invalid_mask,
    zahnle_marley_2014_tco,
)


@pytest.mark.parametrize(("p_bar", "T", "m"), [(0.5, 900.0, 1.0), (2.0, 1500.0, 3.0)])
def test_zahnle_marley_2014_tco_matches_eqs_12_13_14(make_pres_cube, make_temp_cube, p_bar, T, m):
    tco = zahnle_marley_2014_tco(make_pres_cube(p_bar), make_temp_cube(T), m=m)

    t_q1 = 1.5e-6 * p_bar**-1 * m**-0.7 * math.exp(42000.0 / T)  # Eq 12
    t_q2 = 40.0 * p_bar**-2 * math.exp(25000.0 / T)  # Eq 13
    expected = 1.0 / (1.0 / t_q1 + 1.0 / t_q2)  # Eq 14
    assert tco.data == pytest.approx(expected, rel=1e-12)
    assert tco.units == "s"
    assert tco.name() == "zahnle_marley_2014_tco"


def test_zahnle_marley_2014_tco_converts_pressure_units(make_temp_cube):
    # 5e4 Pa == 0.5 bar - must match a 0.5 bar cube exactly, not come out
    # 1e5x off from silently treating Pa data as if it were already bar.
    pres_pa = Cube(5.0e4, long_name="air_pressure", units="Pa")
    pres_bar = Cube(0.5, long_name="air_pressure", units="bar")
    temp = make_temp_cube(900.0)

    tco_from_pa = zahnle_marley_2014_tco(pres_pa, temp)
    tco_from_bar = zahnle_marley_2014_tco(pres_bar, temp)

    assert tco_from_pa.data == pytest.approx(tco_from_bar.data, rel=1e-12)


def test_zahnle_marley_2014_tco_leaves_pres_cube_unmodified(make_temp_cube):
    # p_bar = pres.copy() inside zahnle_marley_2014_tco() - the caller's
    # own pres cube (e.g. a notebook's shared pres_cube, reused for other
    # things) must not end up silently rescaled to bar as a side effect.
    pres_pa = Cube(5.0e4, long_name="air_pressure", units="Pa")

    zahnle_marley_2014_tco(pres_pa, make_temp_cube(900.0))

    assert pres_pa.units == "Pa"
    assert pres_pa.data == pytest.approx(5.0e4)


@pytest.mark.parametrize("m", [1.0, 3.0])
def test_zahnle_marley_2014_tco_does_not_warn_at_calibrated_metallicities(
    recwarn, make_pres_cube, make_temp_cube, m
):
    zahnle_marley_2014_tco(make_pres_cube(0.5), make_temp_cube(900.0), m=m)

    assert len(recwarn) == 0


def test_zahnle_marley_2014_tco_warns_off_calibrated_metallicities(make_pres_cube, make_temp_cube):
    # Zahnle & Marley only fit t_q1/t_q2 at m=1 and m=3 - m=10 has no
    # calibration data behind it, whatever m**-0.7 happens to evaluate to.
    with pytest.warns(UserWarning, match="m=10"):
        zahnle_marley_2014_tco(make_pres_cube(0.5), make_temp_cube(900.0), m=10.0)


def test_zahnle_marley_2014_invalid_mask_boundary():
    # Zahnle & Marley's own stated bound: quenching in their ensemble
    # always occurred below 2200 K - >= flags the boundary itself as
    # already unvalidated, not just strictly-above it.
    temp = Cube(np.array([1000.0, 2199.9, 2200.0, 2500.0]), long_name="air_temperature", units="K")

    mask = zahnle_marley_2014_invalid_mask(temp)

    np.testing.assert_array_equal(mask.data, [False, False, True, True])
    assert mask.units == "1"
