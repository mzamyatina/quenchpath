"""Regression tests for quenchpath.chemistry.polynomials.
tests/data/tiny_coeff_NASA_sc.dat is the real coeff_NASA_sc.dat's first 6
lines (H2, O-3P) verbatim - a real fixed-width fragment, not hand-typed,
so column alignment can't drift from the actual format.
"""

import json
from pathlib import Path

import pytest

from quenchpath.chemistry.polynomials import (
    find_duplicate_species,
    load_and_save_nasa_polynomials,
    parse_nasa_polynomials,
    validate_network_coverage,
)

DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture
def tiny_raw_lines():
    return (DATA_DIR / "tiny_coeff_NASA_sc.dat").read_text().splitlines()


def test_parse_nasa_polynomials(tiny_raw_lines):
    plns = parse_nasa_polynomials(tiny_raw_lines)

    assert set(plns) == {"H2", "O-3P"}

    h2 = plns["H2"]
    assert h2["species"] == "H2"
    assert h2["temp_low"] == 200.0
    assert h2["temp_mid"] == 1000.0
    assert h2["temp_high"] == 6000.0
    assert h2["a1_high"] == pytest.approx(2.93286579e00)
    assert h2["a7_high"] == pytest.approx(-1.02432887e00)
    assert h2["a1_low"] == pytest.approx(2.34433112e00)
    assert h2["a7_low"] == pytest.approx(6.83010238e-01)

    o3p = plns["O-3P"]
    assert o3p["temp_low"] == 300.0
    assert o3p["temp_mid"] == 1000.0
    assert o3p["temp_high"] == 5000.0


def test_parse_nasa_polynomials_rejects_truncated_input(tiny_raw_lines):
    # A trailing partial record (not a whole number of 3-line species
    # blocks) must raise, not silently reuse the previous species'
    # coefficients for the truncated one.
    with pytest.raises(AssertionError):
        parse_nasa_polynomials(tiny_raw_lines + ["CH4                  200  6000  1000"])


def test_find_duplicate_species_none_in_tiny_fixture(tiny_raw_lines):
    assert find_duplicate_species(tiny_raw_lines) == []


def test_parse_nasa_polynomials_keeps_first_entry_on_duplicate(tiny_raw_lines):
    # When a species has more than one set of 7-term NASA polynomials, keep
    # the FIRST one - matching the reference reader's own convention (see
    # the module docstring/parse_nasa_polynomials' docstring), which makes
    # no difference to any species actually used by VENOT2019 as it
    # stands. This test pins the choice down with a synthetic case where the two
    # entries have genuinely different coefficients, so a future change
    # back to last-wins would be caught here even though it wouldn't show
    # up in the real-data dict-for-dict check.
    o3p_block = tiny_raw_lines[3:6]  # O-3P's 3 lines, reused as a "second"
    # entry for H2 with clearly different (fabricated) high-temp coefficients
    # so first-vs-last is unambiguous from the parsed result alone.
    second_h2_entry = [
        "H2                   999  9999  9999",
        " 9.99999999E+00  9.99999999E-04 -9.99999999E-07  9.99999999E-11 -9.99999999E-16 -9.99999999E+02 -9.99999999E+00",  # noqa: E501
        " 9.99999999E+00  9.99999999E-03 -9.99999999E-05  9.99999999E-08 -9.99999999E-12 -9.99999999E+02  9.99999999E-01",  # noqa: E501
    ]
    raw_lines = tiny_raw_lines[:3] + o3p_block + second_h2_entry

    assert find_duplicate_species(raw_lines) == ["H2"]

    plns = parse_nasa_polynomials(raw_lines)
    assert plns["H2"]["temp_high"] == 6000.0  # first entry's value, not 9999
    assert plns["H2"]["a1_high"] == pytest.approx(2.93286579e00)  # not 9.99...


def test_find_duplicate_species_detects_repeats(tiny_raw_lines):
    # Duplicate the H2 entry (first 3 lines) to synthesize a repeat.
    duplicated = tiny_raw_lines[:3] + tiny_raw_lines
    assert find_duplicate_species(duplicated) == ["H2"]


def test_validate_network_coverage_passes_for_covered_species(tiny_raw_lines):
    plns = parse_nasa_polynomials(tiny_raw_lines)
    dups = find_duplicate_species(tiny_raw_lines)
    validate_network_coverage(plns, dups, network_gases={"H2", "O-3P"})  # no raise


def test_validate_network_coverage_raises_on_missing_species(tiny_raw_lines):
    plns = parse_nasa_polynomials(tiny_raw_lines)
    dups = find_duplicate_species(tiny_raw_lines)
    with pytest.raises(ValueError, match="No polynomials for"):
        validate_network_coverage(plns, dups, network_gases={"H2", "CH4"})


def test_validate_network_coverage_raises_on_duplicate_species(tiny_raw_lines):
    duplicated = tiny_raw_lines[:3] + tiny_raw_lines
    plns = parse_nasa_polynomials(duplicated)
    dups = find_duplicate_species(duplicated)
    with pytest.raises(ValueError, match="more than one set"):
        validate_network_coverage(plns, dups, network_gases={"H2"})


def test_load_and_save_nasa_polynomials_writes_json(tmp_path):
    # Stage 01 driver: load+parse+validate+save in one call. Uses a
    # tmp_path raw/processed layout rather than the real networks/venot2019
    # dirs to stay fast/isolated.
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    raw_dir.mkdir()
    processed_dir.mkdir()
    (raw_dir / "coeff_NASA_sc.dat").write_text((DATA_DIR / "tiny_coeff_NASA_sc.dat").read_text())

    plns = load_and_save_nasa_polynomials(raw_dir, processed_dir, network_gases={"H2", "O-3P"})

    assert set(plns) == {"H2", "O-3P"}
    saved = json.loads((processed_dir / "nasa_polynomials_7_term.json").read_text())
    assert saved == plns


def test_load_and_save_nasa_polynomials_raises_and_does_not_write_on_missing_coverage(tmp_path):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    raw_dir.mkdir()
    processed_dir.mkdir()
    (raw_dir / "coeff_NASA_sc.dat").write_text((DATA_DIR / "tiny_coeff_NASA_sc.dat").read_text())

    with pytest.raises(ValueError, match="No polynomials for"):
        load_and_save_nasa_polynomials(raw_dir, processed_dir, network_gases={"H2", "CH4"})

    assert not (processed_dir / "nasa_polynomials_7_term.json").exists()
