"""Unit tests for quenchpath.chemistry.formulas' rate-coefficient/
thermodynamics math. Pure functions (Cubes/floats in, Cubes out, no file
I/O) pinning down real historical bugs (K_eq double-RT-division, Troe
broadening-factor bracketing, reverse-rate-coefficient relabel-vs-rescale)
that a future refactor could otherwise silently reintroduce.

Expected values are computed independently via the `math` module in each
test body, not by calling back into quenchpath.chemistry.formulas - so
these are genuine regression checks on the formula translation (Cube/dask
arithmetic vs plain Python), not tautologies.
"""

import math

import pytest
from cf_units import Unit

from quenchpath.chemistry.formulas import (
    chemical_potential,
    equilibrium_constant,
    k_kooij,
    k_troe,
    reverse_rate_coeff,
)


def test_k_kooij(make_temp_cube):
    alpha, beta, gamma, T = 2e-11, 1.5, 100.0, 600.0
    k = k_kooij(make_temp_cube(T), alpha, beta, gamma)
    expected = alpha * (T / 300) ** beta * math.exp(-gamma / T)
    assert k.data == pytest.approx(expected, rel=1e-12)
    assert k.name() == "k_kooij"


def test_k_troe_lindemann_branch(make_temp_cube, bath_gas_mixture, unit_effcs):
    # a_troe == b_troe == c_troe == d_troe == 0 selects the (UNTESTED per
    # its own docstring, but still exercised here) Lindemann-Hinshelwood
    # branch: F == 1 always, no broadening factor.
    T, k_0, k_inf = 800.0, 1e-30, 1e-10
    M = 9 * 1e10  # 9 species in bath_gas_mixture, all effc=1.0, all 1e10 cm-3
    k = k_troe(
        make_temp_cube(T),
        make_temp_cube(k_0),
        make_temp_cube(k_inf),
        0.0,
        0.0,
        0.0,
        0.0,
        unit_effcs,
        bath_gas_mixture,
    )
    p_r = k_0 * M / k_inf
    expected = k_inf * (p_r / (1 + p_r)) * 1.0
    assert k.data == pytest.approx(expected, rel=1e-9)


def test_k_troe_broadening_branch(make_temp_cube, bath_gas_mixture, unit_effcs):
    # x = log10(P_r) + c; frac = 1/(1 + (x/(N - d*x))**2) - d multiplies
    # the whole (log10_P_r + c) bracket, not just log10_P_r on its own.
    T = 800.0
    a_troe, b_troe, c_troe, d_troe = 0.6, 200.0, 1000.0, 1500.0
    k_0, k_inf = 1e-30, 1e-10
    M = 9 * 1e10

    k = k_troe(
        make_temp_cube(T),
        make_temp_cube(k_0),
        make_temp_cube(k_inf),
        a_troe,
        b_troe,
        c_troe,
        d_troe,
        unit_effcs,
        bath_gas_mixture,
    )

    p_r = k_0 * M / k_inf
    f_c = (
        (1 - a_troe) * math.exp(-T / b_troe)
        + a_troe * math.exp(-T / c_troe)
        + math.exp(-d_troe / T)
    )
    c = -0.4 - 0.67 * math.log10(f_c)
    N = 0.75 - 1.27 * math.log10(f_c)
    d = 0.14
    x = math.log10(p_r) + c
    frac = 1 / (1 + (x / (N - d * x)) ** 2)
    f = f_c**frac
    expected = k_inf * (p_r / (1 + p_r)) * f

    assert k.data == pytest.approx(expected, rel=1e-9)


def test_chemical_potential_low_temp_branch(make_temp_cube):
    # Zero out every coefficient but a1/a7 to keep the expected value a
    # one-line hand computation: mu = a1*(1 - ln(T)) - a7.
    T = 500.0
    pln = {
        "species": "test_species",
        "temp_low": 200.0,
        "temp_mid": 1000.0,
        "temp_high": 2000.0,
        "a1_low": 2.5,
        "a2_low": 0.0,
        "a3_low": 0.0,
        "a4_low": 0.0,
        "a5_low": 0.0,
        "a6_low": 0.0,
        "a7_low": 0.7,
        "a1_high": 0.0,
        "a2_high": 0.0,
        "a3_high": 0.0,
        "a4_high": 0.0,
        "a5_high": 0.0,
        "a6_high": 0.0,
        "a7_high": 0.0,
    }
    mu = chemical_potential(make_temp_cube(T), pln)
    expected = 2.5 * (1 - math.log(T)) - 0.7
    assert mu.data == pytest.approx(expected, rel=1e-12)
    # Must stay dimensionless (mu^0/(RT)), not J/mol.
    assert mu.units == Unit("1")


def test_chemical_potential_raises_outside_polynomial_range(make_temp_cube):
    pln = {
        "species": "test_species",
        "temp_low": 200.0,
        "temp_mid": 1000.0,
        "temp_high": 2000.0,
        **{f"a{n}_low": 0.0 for n in range(1, 8)},
        **{f"a{n}_high": 0.0 for n in range(1, 8)},
    }
    with pytest.raises(ValueError):
        chemical_potential(make_temp_cube(5000.0), pln)  # above temp_high


def test_equilibrium_constant(make_temp_cube):
    gibbs = 2.0
    k_eq = equilibrium_constant(make_temp_cube(gibbs))
    assert k_eq.data == pytest.approx(math.exp(-gibbs), rel=1e-12)
    assert k_eq.name() == "equilibrium_constant"


def test_reverse_rate_coeff_rescales_not_just_relabels(make_temp_cube):
    # `.units = "..."` relabels without rescaling the data;
    # `.convert_units(...)` actually rescales. nprods == nreacs here so
    # (kB*T/P_0)**(nprods-nreacs) == 1 and drops out, isolating the
    # relabel-vs-rescale question from the separate kB/P_0 volume-factor
    # one covered by test_reverse_rate_coeff_applies_volume_factor.
    import iris.cube

    k_f = iris.cube.Cube(1.0, units="m3 s-1")
    k_eq = make_temp_cube(2.0)
    k_eq.units = "1"
    k_r = reverse_rate_coeff(make_temp_cube(300.0), k_f, k_eq, nprods=2, nreacs=2)
    # 1.0 m3 s-1 / 2.0 == 0.5 m3 s-1 == 5e5 cm3 s-1 (1 m3 = 1e6 cm3)
    assert k_r.data == pytest.approx(5e5, rel=1e-9)
    assert str(k_r.units) == "cm3 s-1"


def test_reverse_rate_coeff_applies_volume_factor(make_temp_cube):
    import iris.cube

    from quenchpath.chemistry.formulas import P_0, kB

    # nreacs=1 (unimolecular forward, e.g. a decomposition) -> k_f is
    # physically in s-1, not cm3 s-1; nprods=3 -> the reverse direction is
    # three-body/termolecular, k_r in cm6 s-1. Getting this pairing
    # physically consistent matters - an inconsistent choice here produces
    # a genuinely non-convertible unit (an extra length^3 dimension), which
    # is exactly what an earlier draft of this test got wrong (caught by
    # cf_units itself refusing the conversion, not silently).
    T, k_f_val, k_eq_val = 300.0, 1e-9, 4.0
    k_f = iris.cube.Cube(k_f_val, units="s-1")
    k_eq = make_temp_cube(k_eq_val)
    k_eq.units = "1"
    k_r = reverse_rate_coeff(make_temp_cube(T), k_f, k_eq, nprods=3, nreacs=1)

    volume_factor_si = (kB.data * T / P_0.data) ** (3 - 1)  # m^6 (SI)
    expected_m6 = k_f_val / k_eq_val * volume_factor_si  # s-1 * m6 == m6 s-1
    expected_cm6 = expected_m6 * (100**6)  # m6 -> cm6 (1 m = 100 cm)
    assert k_r.data == pytest.approx(expected_cm6, rel=1e-6)
    assert str(k_r.units) == "cm6 s-1"
