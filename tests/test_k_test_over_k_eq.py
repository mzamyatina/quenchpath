"""Unit test for quenchpath.analysis.k_test_over_k_eq.ch4_co_k_test_over_k_eq()'s
formula/stoichiometry wiring (K_test/K_eq for the net CH4 + H2O <->
CO + 3H2 reaction) - same "expected value computed independently, not
by calling back into the module under test" convention as
tests/test_chemistry_formulas.py, extended here to also pin down the
eqn dict's reacs/prods (CO + 3*H2 -> CH4 + H2O, H2 listed 3 times) -
using 4 distinct per-species coefficients so a stoichiometry mistake
(e.g. H2 counted once instead of 3 times) would change the expected
value, not cancel out.
"""

import json
import math

import iris.cube
import pytest

from quenchpath import io
from quenchpath.analysis import k_test_over_k_eq
from quenchpath.chemistry.formulas import P_0, kB
from quenchpath.config import RunConfig


@pytest.fixture
def cfg():
    return RunConfig(planet="tinyplanet", exp="kin", mdh="0", tK="1")


def _pln(a1):
    """A 7-term NASA polynomial dict with only a1 nonzero (a7=0), so
    mu^0/(RT) = a1*(1 - ln(T)) - same minimal-coefficient convention as
    tests/test_chemistry_formulas.py's own chemical_potential() tests."""
    return {
        "temp_low": 200.0,
        "temp_mid": 1000.0,
        "temp_high": 2000.0,
        **{f"a{n}_low": (a1 if n == 1 else 0.0) for n in range(1, 8)},
        **{f"a{n}_high": (a1 if n == 1 else 0.0) for n in range(1, 8)},
    }


def test_ch4_co_k_test_over_k_eq_formula(tmp_path, monkeypatch, cfg):
    monkeypatch.setattr(io, "_CONVERTED_DIR", tmp_path / "converted")
    monkeypatch.setattr(k_test_over_k_eq, "path_to_networks_processed", tmp_path)

    # Distinct a1 per species (a7=0) - a wrong stoichiometry (e.g. H2
    # counted once, not 3 times) would change gibbs_energy_f, not
    # cancel out, unlike if every species shared one mu value.
    plns = {
        "CH4": _pln(2.0),
        "H2O": _pln(3.0),
        "CO": _pln(5.0),
        "H2": _pln(7.0),
    }
    with open(tmp_path / "nasa_polynomials_7_term.json", "w") as f:
        json.dump(plns, f)

    T = 500.0
    number_densities = {"CH4": 2e10, "H2O": 3e10, "CO": 5e10, "H2": 7e10}  # cm-3
    cubes = iris.cube.CubeList(
        [iris.cube.Cube(T, long_name="air_temperature", units="K")]
        + [
            iris.cube.Cube(nd, long_name=f"{gas} number density", units="cm-3")
            for gas, nd in number_densities.items()
        ]
    )
    io.save_converted(cubes, cfg)

    ratio = k_test_over_k_eq.ch4_co_k_test_over_k_eq(cfg)

    # p_i [Pa] = n_i [cm-3 -> m-3] * kB * T; K_test/K_eq below both use
    # p_i/P_0 (dimensionless), matching the K_p convention
    # calc_gibbs_and_equilibrium_constant() itself uses.
    p = {gas: (nd * 1e6 * kB.data * T) / P_0.data for gas, nd in number_densities.items()}
    k_test = (p["CH4"] * p["H2O"]) / (p["CO"] * p["H2"] ** 3)

    # eqn = CO + 3*H2 -> CH4 + H2O (reacs=[CO, H2, H2, H2], prods=[CH4, H2O]) -
    # gibbs_energy_f = sum(mu_prods) - sum(mu_reacs), mu_i = a1_i*(1-ln(T)).
    mu = {"CH4": 2.0, "H2O": 3.0, "CO": 5.0, "H2": 7.0}
    gibbs_energy_f = (mu["CH4"] + mu["H2O"] - (mu["CO"] + 3 * mu["H2"])) * (1 - math.log(T))
    k_eq = math.exp(-gibbs_energy_f)

    assert ratio.data == pytest.approx(k_test / k_eq, rel=1e-9)
    assert ratio.name() == "ch4_co_k_test_over_k_eq"
