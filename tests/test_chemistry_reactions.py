"""Regression tests for quenchpath.chemistry.reactions.
tests/data/tiny_reactions_*.dat are each the real reactions_N.dat file's
first 2 lines verbatim - real fixed-width fragments, not hand-typed.
"""

import json
from pathlib import Path

import pytest

from quenchpath.chemistry.reactions import (
    load_and_save_venot2019_network,
    load_venot2019_network,
    parse_kooij_low_pressure_limit_reactions,
    parse_kooij_reactions,
    parse_troe_reactions,
)

DATA_DIR = Path(__file__).parent / "data"


def _lines(name):
    return (DATA_DIR / name).read_text().splitlines()


def test_parse_kooij_reactions():
    eqns = parse_kooij_reactions(_lines("tiny_reactions_1.dat"), rtype="bimolecular")
    assert len(eqns) == 2

    first = eqns[0]
    assert first["reacs"] == ["CH3", "H"]
    assert first["prods"] == ["1CH2", "H2"]
    assert first["eqn"] == "CH3+H↔1CH2+H2"
    assert first["alpha"] == pytest.approx(0.997e-10)
    assert first["beta"] == pytest.approx(0.0)
    assert first["gamma"] == pytest.approx(0.754e04)
    assert first["F"] == pytest.approx(2.0)
    assert first["g"] == pytest.approx(100.0)
    assert first["rtype"] == "bimolecular"
    assert first["ktype"] == "kooij"

    second = eqns[1]
    assert second["reacs"] == ["CH4", "H"]
    assert second["prods"] == ["CH3", "H2"]
    assert second["eqn"] == "CH4+H↔CH3+H2"
    assert second["alpha"] == pytest.approx(0.583e-12)
    assert second["beta"] == pytest.approx(3.0)


def test_parse_kooij_low_pressure_limit_reactions():
    eqns = parse_kooij_low_pressure_limit_reactions(
        _lines("tiny_reactions_2.dat"), rtype="combination"
    )
    assert len(eqns) == 2

    first = eqns[0]
    assert first["reacs"] == ["H", "H"]
    assert first["prods"] == ["H2"]
    assert first["eqn"] == "H+H+M↔H2+M"
    assert first["alpha_0"] == pytest.approx(0.172e-31)
    assert first["beta_0"] == pytest.approx(-1.0)
    assert first["gamma_0"] == pytest.approx(0.0)
    assert first["F_0"] == pytest.approx(2.0)
    assert first["g_0"] == pytest.approx(100.0)
    assert len(first["effcs"]) == 22
    assert first["effcs"][:5] == [
        pytest.approx(v) for v in [4.000e-01, 7.500e-01, 1.500e00, 6.500e00, 3.000e00]
    ]
    assert first["rtype"] == "combination"
    assert first["ktype"] == "kooij low pressure limit"


def test_parse_troe_reactions():
    eqns = parse_troe_reactions(_lines("tiny_reactions_3.dat"), rtype="combination")
    assert len(eqns) == 2

    first = eqns[0]
    assert first["reacs"] == ["H", "CH3"]
    assert first["prods"] == ["CH4"]
    assert first["eqn"] == "H+CH3(+M)↔CH4(+M)"
    assert first["alpha_0"] == pytest.approx(0.135e-27)
    assert first["alpha_inf"] == pytest.approx(0.277e-09)
    assert first["a_troe"] == pytest.approx(0.37)
    assert first["b_troe"] == pytest.approx(0.332e04)
    assert first["c_troe"] == pytest.approx(0.61e02)
    assert first["d_troe"] == pytest.approx(0.1e19)
    assert len(first["effcs"]) == 22
    assert first["ktype"] == "troe"


def test_load_venot2019_network_uses_tiny_directory(tmp_path):
    # Build a minimal 7-file directory (one line each) mirroring
    # load_venot2019_network()'s expected layout, to check file-group
    # ordering/dispatch end-to-end without touching the real (large) data.
    single_kooij_line = _lines("tiny_reactions_1.dat")[:1]
    single_low_p_line = _lines("tiny_reactions_2.dat")[:1]
    single_troe_line = _lines("tiny_reactions_3.dat")[:1]

    for n in (1, 7):
        (tmp_path / f"reactions_{n}.dat").write_text("\n".join(single_kooij_line))
    for n in (2, 8, 13):
        (tmp_path / f"reactions_{n}.dat").write_text("\n".join(single_low_p_line))
    for n in (3, 10):
        (tmp_path / f"reactions_{n}.dat").write_text("\n".join(single_troe_line))

    eqns = load_venot2019_network(tmp_path)
    assert len(eqns) == 7
    # Order must follow the original script's `for n in [1, 2, 3, 7, 8, 10,
    # 13]` loop exactly, not e.g. grouped by parser type.
    assert [e["ktype"] for e in eqns] == [
        "kooij",
        "kooij low pressure limit",
        "troe",
        "kooij",
        "kooij low pressure limit",
        "troe",
        "kooij low pressure limit",
    ]
    assert [e["rtype"] for e in eqns] == [
        "bimolecular",
        "combination",
        "combination",
        "decomposition",
        "decomposition",
        "decomposition",
        "desexcitation",
    ]


def test_load_and_save_venot2019_network_writes_json(tmp_path):
    # Stage 02 driver: load+parse+save in one call, reusing the same tiny
    # 7-file directory layout as test_load_venot2019_network_uses_tiny_
    # directory above.
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    raw_dir.mkdir()
    processed_dir.mkdir()
    single_kooij_line = _lines("tiny_reactions_1.dat")[:1]
    single_low_p_line = _lines("tiny_reactions_2.dat")[:1]
    single_troe_line = _lines("tiny_reactions_3.dat")[:1]
    for n in (1, 7):
        (raw_dir / f"reactions_{n}.dat").write_text("\n".join(single_kooij_line))
    for n in (2, 8, 13):
        (raw_dir / f"reactions_{n}.dat").write_text("\n".join(single_low_p_line))
    for n in (3, 10):
        (raw_dir / f"reactions_{n}.dat").write_text("\n".join(single_troe_line))

    eqns = load_and_save_venot2019_network(raw_dir, processed_dir)

    assert len(eqns) == 7
    saved = json.loads((processed_dir / "venot2019_network.json").read_text())
    assert saved == eqns
