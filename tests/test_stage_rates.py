"""Regression tests for quenchpath.stages.stage_04_rates.

tests/data/tiny_processed_run.nc is a real (not synthetic) 2x2x2 slice of
hd189733b's actual stage-03 output (all 30 gas number density cubes +
air_temperature). tests/data/tiny_network.json/tiny_polynomials.json are a
small, hand-picked subset of the real VENOT2019 network/polynomials
(regenerated via quenchpath.chemistry.* from the real raw .dat files, not
hand-typed) covering exactly 4 reactions: one "kooij" (CH3+H<->1CH2+H2),
one "kooij low pressure limit" (H+H+M<->H2+M), and the
H+CH3(+M)<->CH4(+M)/CH4(+M)<->CH3+H(+M) "troe" pair that exercises the
merge_h_ch3_ch4_special_case() special case.

tests/data/tiny_rates_golden.nc is these functions' own output on that
tiny data, captured once - a fast, CI-friendly regression check, not
the primary correctness evidence (that was a full-scale, bit-for-bit
comparison against real hd189733b data, 180 reactions/1080 cubes).
"""

import json
from pathlib import Path

import iris
import numpy as np
import pytest

from quenchpath.chemistry.network import VENOT2019
from quenchpath.config import RunConfig
from quenchpath.stages.stage_04_rates import (
    build_reaction_metadata,
    calc_forward_rate,
    calc_gibbs_and_equilibrium_constant,
    calc_reverse_rate,
    merge_h_ch3_ch4_special_case,
    split_gases_and_bath_gases,
)

DATA_DIR = Path(__file__).parent / "data"


def _attrs_equivalent(fresh, golden):
    """Compare a freshly-computed metadata dict against one that's been
    through a NetCDF save/load round-trip. NetCDF attribute storage
    quirks mean the round-tripped side isn't type-identical even when the
    content is: a single-element list becomes a bare scalar (e.g.
    prods=["H2"] -> "H2"), and a list of floats becomes a numpy array -
    normalize both before comparing instead of relying on dict ==.
    """
    if set(fresh) != set(golden):
        return False
    for key, fresh_val in fresh.items():
        golden_val = golden[key]
        is_numeric_list = (
            isinstance(fresh_val, list) and fresh_val and isinstance(fresh_val[0], (int, float))
        )
        if is_numeric_list:
            if not np.array_equal(fresh_val, golden_val):
                return False
        else:
            if isinstance(fresh_val, list) and not isinstance(golden_val, (list, np.ndarray)):
                golden_val = [golden_val]
            if fresh_val != golden_val:
                return False
    return True


@pytest.fixture
def tiny_setup():
    processed = iris.load(DATA_DIR / "tiny_processed_run.nc")
    all_gases_nd, bath_gases_nd = split_gases_and_bath_gases(processed, VENOT2019["bath_gases"])
    temp = processed.extract_cube("air_temperature")
    temp.attributes = {}

    with open(DATA_DIR / "tiny_network.json") as f:
        eqns = json.load(f)
    with open(DATA_DIR / "tiny_polynomials.json") as f:
        plns = json.load(f)

    return {
        "all_gases_nd": all_gases_nd,
        "bath_gases_nd": bath_gases_nd,
        "temp": temp,
        "eqns": eqns,
        "plns": plns,
    }


def _run_full_pipeline(setup):
    """Same 3-pass + special-case sequence calc_rates() runs, against the
    tiny fixture's in-memory data (no I/O)."""
    eqns = setup["eqns"]
    temp, plns = setup["temp"], setup["plns"]
    all_gases_nd, bath_gases_nd = setup["all_gases_nd"], setup["bath_gases_nd"]

    for eqn in eqns:
        g, k_eq = calc_gibbs_and_equilibrium_constant(eqn, temp, plns)
        eqn.update({"gibbs_energy_f": g, "k_eq": k_eq})
    for eqn in eqns:
        k_f, rate_f = calc_forward_rate(eqn, temp, all_gases_nd, bath_gases_nd)
        eqn.update({"k_f": k_f, "rate_f": rate_f})
    for eqn in eqns:
        k_r, rate_r = calc_reverse_rate(eqn, temp, all_gases_nd)
        eqn.update({"k_r": k_r, "rate_r": rate_r})

    return merge_h_ch3_ch4_special_case(eqns, temp, all_gases_nd)


def test_full_pipeline_matches_golden_output(tiny_setup):
    # NOTE: golden fixture was captured on macOS/its BLAS build; CI runs on
    # Linux. rtol=1e-10 (not exact equality) accommodates tiny last-ULP
    # differences across platforms/BLAS builds - see
    # test_stage_convert.py's identical note for the precedent (a real CI
    # failure from exact equality, ~2.75e-16 relative difference, right at
    # float64 epsilon - not a bug).
    eqns = _run_full_pipeline(tiny_setup)

    golden = iris.load(DATA_DIR / "tiny_rates_golden.nc")
    golden_eqn_strings = {c.attributes["eqn"] for c in golden}
    assert {e["eqn"] for e in eqns} == golden_eqn_strings

    cfg = RunConfig(planet="tinyplanet", exp="kin")
    for eqn in eqns:
        metadata = build_reaction_metadata(eqn, cfg)
        for var in ["gibbs_energy_f", "k_eq", "k_f", "k_r", "rate_f", "rate_r"]:
            cube = eqn[var]
            cube.attributes = metadata
            [golden_cube] = [
                c for c in golden if c.attributes["eqn"] == eqn["eqn"] and c.name() == cube.name()
            ]
            np.testing.assert_allclose(cube.data, golden_cube.data, rtol=1e-10)
            # "Conventions" is a NetCDF/CF marker iris adds on save, not
            # part of the reaction metadata itself - the golden cube went
            # through a save/load round-trip, the freshly-computed one
            # didn't, so exclude it from the comparison (see
            # _attrs_equivalent()'s docstring for the other round-trip
            # quirks it normalizes: single-element lists become scalars,
            # float lists become numpy arrays). "planet"/"mdh"/"tK" are
            # excluded too - build_reaction_metadata() gained them after
            # this golden fixture was captured, so they're genuinely
            # absent from it, not a round-trip artifact.
            fresh_attrs = dict(cube.attributes)
            for key in ("planet", "mdh", "tK"):
                fresh_attrs.pop(key, None)
            golden_attrs = dict(golden_cube.attributes)
            golden_attrs.pop("Conventions", None)
            assert _attrs_equivalent(fresh_attrs, golden_attrs)


def test_merge_h_ch3_ch4_special_case_drops_the_reverse_entry(tiny_setup):
    eqns = _run_full_pipeline(tiny_setup)
    eqn_strings = [e["eqn"] for e in eqns]

    assert "H+CH3(+M)↔CH4(+M)" in eqn_strings
    # The separate CH4(+M)<->CH3+H(+M) entry must be dropped once its
    # rates are folded into H+CH3(+M)<->CH4(+M).
    assert "CH4(+M)↔CH3+H(+M)" not in eqn_strings
    assert len(eqn_strings) == len(set(eqn_strings)) == 3


def test_merged_reaction_carries_full_metadata_not_a_placeholder(tiny_setup):
    # dict(h_ch3_m) must carry reacs/prods/rtype/ktype/etc. over from the
    # original entry, not just a near-empty placeholder.
    eqns = _run_full_pipeline(tiny_setup)
    merged = next(e for e in eqns if e["eqn"] == "H+CH3(+M)↔CH4(+M)")

    assert merged["reacs"] == ["H", "CH3"]
    assert merged["prods"] == ["CH4"]
    assert merged["rtype"] == "combination"
    assert merged["ktype"] == "troe"

    metadata = build_reaction_metadata(merged, RunConfig(planet="tinyplanet", exp="kin"))
    assert metadata["kpars_g_inf"] == merged["g_inf"]


def test_build_reaction_metadata_does_not_confuse_g_0_and_g_inf():
    # g_0 and g_inf happen to coincide for every Troe reaction in the real
    # network (both always 100.0), so a real-data check can't catch these
    # being swapped - use deliberately different synthetic values instead.
    eqn = {
        "eqn": "A+B(+M)↔C(+M)",
        "reacs": ["A", "B"],
        "prods": ["C"],
        "rtype": "combination",
        "ktype": "troe",
        "alpha_0": 1.0,
        "beta_0": 1.0,
        "gamma_0": 1.0,
        "F_0": 1.0,
        "g_0": 111.0,
        "alpha_inf": 1.0,
        "beta_inf": 1.0,
        "gamma_inf": 1.0,
        "F_inf": 1.0,
        "g_inf": 222.0,
        "a_troe": 1.0,
        "b_troe": 1.0,
        "c_troe": 1.0,
        "d_troe": 1.0,
        "effcs": [1.0] * 22,
    }
    metadata = build_reaction_metadata(eqn, RunConfig(planet="tinyplanet", exp="kin"))
    assert metadata["kpars_g_0"] == 111.0
    assert metadata["kpars_g_inf"] == 222.0


def test_merged_reaction_k_eq_is_dimensionless_k_p_not_k_c(tiny_setup):
    # k_eq must be the (kB*T/P_0)**(nprods-nreacs) dimensionless K_p
    # convention every other reaction's k_eq uses, not the raw k_f/k_r
    # ratio (K_c, units cm3).
    eqns = _run_full_pipeline(tiny_setup)
    merged = next(e for e in eqns if e["eqn"] == "H+CH3(+M)↔CH4(+M)")

    assert str(merged["k_eq"].units) == "1"
    k_c = merged["k_f"] / merged["k_r"]
    assert not np.allclose(merged["k_eq"].data, k_c.data)


def test_calc_forward_rate_dispatches_by_ktype(tiny_setup):
    kooij_eqn = next(e for e in tiny_setup["eqns"] if e["ktype"] == "kooij")
    low_p_eqn = next(e for e in tiny_setup["eqns"] if e["ktype"] == "kooij low pressure limit")

    k_f_kooij, _ = calc_forward_rate(
        kooij_eqn, tiny_setup["temp"], tiny_setup["all_gases_nd"], tiny_setup["bath_gases_nd"]
    )
    k_f_low_p, _ = calc_forward_rate(
        low_p_eqn, tiny_setup["temp"], tiny_setup["all_gases_nd"], tiny_setup["bath_gases_nd"]
    )
    assert k_f_kooij.name() == "forward_reaction_rate_coefficient"
    assert k_f_low_p.name() == "forward_reaction_rate_coefficient"


def test_calc_forward_rate_rejects_unknown_ktype(tiny_setup):
    bad_eqn = dict(tiny_setup["eqns"][0])
    bad_eqn["ktype"] = "not_a_real_ktype"
    with pytest.raises(ValueError, match="No such ktype"):
        calc_forward_rate(
            bad_eqn, tiny_setup["temp"], tiny_setup["all_gases_nd"], tiny_setup["bath_gases_nd"]
        )
