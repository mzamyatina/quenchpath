"""Tests for quenchpath.io's chunked/batched save-load round-trips
(stage 04's rate-cube chunking, stage 05's per-pressure-level JSON
Lines, ch4_co_paths' resolution/mdh/tK nesting) - chunk-boundary math
and glob-pattern scoping are risky enough to be worth a fast regression
test rather than relying on hand-verification against real data alone.

All tests monkeypatch quenchpath.io's module-level directory constants
to a pytest tmp_path, so nothing here touches the real outputs/ tree.
"""

import json

import iris.coords
import iris.cube
import numpy as np
import pytest

from quenchpath import io
from quenchpath.config import GridResolution, RunConfig, SpeciesConversion


@pytest.fixture
def cfg():
    return RunConfig(planet="tinyplanet", exp="kin", mdh="0", tK="1")


def _rate_cube(eqn, var_name, value=1.0):
    cube = iris.cube.Cube(value, var_name=var_name)
    cube.attributes = {"eqn": eqn}
    return cube


# ---------------------------------------------------------------------
# stage 04: save_rate_cubes/load_rate_cubes chunking
# ---------------------------------------------------------------------


def test_save_rate_cubes_chunks_into_expected_file_count(tmp_path, monkeypatch, cfg):
    monkeypatch.setattr(io, "_RATES_DIR", tmp_path)
    monkeypatch.setattr(io, "_RATE_CUBES_PER_FILE", 3)

    cubes = iris.cube.CubeList(_rate_cube(f"eqn{i}", "k_eq") for i in range(7))
    io.save_rate_cubes(cubes, cfg)

    # 7 cubes at 3/file -> 3 files (3+3+1), not 1 file or 7 files.
    saved = sorted(tmp_path.glob("*.nc"))
    assert len(saved) == 3
    assert [p.name for p in saved] == [io.rate_cubes_filename(cfg, i) for i in range(3)]


def test_save_and_load_rate_cubes_roundtrip(tmp_path, monkeypatch, cfg):
    monkeypatch.setattr(io, "_RATES_DIR", tmp_path)
    monkeypatch.setattr(io, "_RATE_CUBES_PER_FILE", 2)

    cubes = iris.cube.CubeList(
        [
            _rate_cube("A+B<->C", "k_eq", value=1.5),
            _rate_cube("A+B<->C", "rate_f", value=2.5),
            _rate_cube("D+E<->F", "k_eq", value=3.5),
        ]
    )
    io.save_rate_cubes(cubes, cfg)

    loaded = io.load_rate_cubes(cfg)
    assert len(loaded) == 3
    by_eqn_and_name = {(c.attributes["eqn"], c.var_name): float(c.data) for c in loaded}
    assert by_eqn_and_name == {
        ("A+B<->C", "k_eq"): 1.5,
        ("A+B<->C", "rate_f"): 2.5,
        ("D+E<->F", "k_eq"): 3.5,
    }


def test_load_rate_cubes_does_not_cross_contaminate_different_configs(tmp_path, monkeypatch):
    # Two different RunConfigs' rate-cube files must never mix.
    monkeypatch.setattr(io, "_RATES_DIR", tmp_path)
    monkeypatch.setattr(io, "_RATE_CUBES_PER_FILE", 10)

    cfg_a = RunConfig(planet="planeta", exp="kin")
    cfg_b = RunConfig(planet="planetb", exp="kin")
    io.save_rate_cubes(iris.cube.CubeList([_rate_cube("X<->Y", "k_eq", value=1.0)]), cfg_a)
    io.save_rate_cubes(iris.cube.CubeList([_rate_cube("X<->Y", "k_eq", value=2.0)]), cfg_b)

    [cube_a] = io.load_rate_cubes(cfg_a)
    [cube_b] = io.load_rate_cubes(cfg_b)
    assert float(cube_a.data) == 1.0
    assert float(cube_b.data) == 2.0


# ---------------------------------------------------------------------
# stage 05: save_paths_level/load_paths_from_disk
# ---------------------------------------------------------------------


def test_save_paths_level_writes_one_file_per_level(tmp_path, monkeypatch, cfg):
    monkeypatch.setattr(io, "path_to_intermediates", tmp_path)

    level_1 = [{"pres": 1.0, "lat": 0.0, "lon": 0.0, "segments": []}]
    level_2 = [
        {"pres": 2.0, "lat": 0.0, "lon": 0.0, "segments": []},
        {"pres": 2.0, "lat": 1.0, "lon": 0.0, "segments": []},
    ]
    io.save_paths_level(level_1, cfg, GridResolution.SAMPLE, level_index=0, level_height=100.0)
    io.save_paths_level(level_2, cfg, GridResolution.SAMPLE, level_index=1, level_height=200.0)

    saved = sorted(io.paths_dir(cfg, GridResolution.SAMPLE).glob("*.jsonl"))
    assert len(saved) == 2  # one file per level, not one per location


def test_paths_level_filename_distinguishes_adjacent_levels():
    # Two levels close enough in coordinate *value* to round to the same
    # formatted string (e.g. adjacent native level_height levels near the
    # top of the atmosphere) must still get distinct filenames - keyed by
    # integer level index, not the level's own value.
    cfg = RunConfig(planet="tinyplanet", exp="kin")
    assert io.paths_level_filename(cfg, GridResolution.SAMPLE, 64) != io.paths_level_filename(
        cfg, GridResolution.SAMPLE, 65
    )


def test_ch4_co_paths_does_not_cross_contaminate_different_mdh_tk(tmp_path, monkeypatch):
    # Two calc-paths runs for the same planet/exp but different mdh/tK
    # must never mix per-level files.
    monkeypatch.setattr(io, "path_to_intermediates", tmp_path)

    cfg_mdh0 = RunConfig(planet="tinyplanet", exp="kin", mdh="0", tK="1")
    cfg_mdh1 = RunConfig(planet="tinyplanet", exp="kin", mdh="1", tK="1")
    io.save_paths_level(
        [{"pres": 1.0, "lat": 0.0, "lon": 0.0, "segments": ["mdh0"]}],
        cfg_mdh0,
        GridResolution.SAMPLE,
        level_index=0,
        level_height=100.0,
    )
    io.save_paths_level(
        [{"pres": 1.0, "lat": 0.0, "lon": 0.0, "segments": ["mdh1"]}],
        cfg_mdh1,
        GridResolution.SAMPLE,
        level_index=0,
        level_height=100.0,
    )

    locations_mdh0 = io.load_paths_from_disk(cfg_mdh0, GridResolution.SAMPLE)
    locations_mdh1 = io.load_paths_from_disk(cfg_mdh1, GridResolution.SAMPLE)
    assert [loc["segments"] for loc in locations_mdh0] == [["mdh0"]]
    assert [loc["segments"] for loc in locations_mdh1] == [["mdh1"]]


def test_save_and_load_paths_from_disk_roundtrip(tmp_path, monkeypatch, cfg):
    monkeypatch.setattr(io, "path_to_intermediates", tmp_path)

    level_1 = [{"pres": 1.0, "lat": 0.0, "lon": 0.0, "segments": ["A->B"]}]
    level_2 = [
        {"pres": 2.0, "lat": 0.0, "lon": 0.0, "segments": ["C->D"]},
        {"pres": 2.0, "lat": 1.0, "lon": 0.0, "segments": ["E->F"]},
    ]
    io.save_paths_level(level_1, cfg, GridResolution.SAMPLE, level_index=0, level_height=100.0)
    io.save_paths_level(level_2, cfg, GridResolution.SAMPLE, level_index=1, level_height=200.0)

    locations = io.load_paths_from_disk(cfg, GridResolution.SAMPLE)
    assert len(locations) == 3
    by_lat_pres = {(d["pres"], d["lat"]): d["segments"] for d in locations}
    assert by_lat_pres == {
        (1.0, 0.0): ["A->B"],
        (2.0, 0.0): ["C->D"],
        (2.0, 1.0): ["E->F"],
    }
    # hgt_idx/hgt aren't in the saved per-location dicts (see the header
    # line test below) - load_paths_from_disk() folds them back in
    # from each file's header.
    by_lat_hgt = {(d["pres"], d["lat"]): (d["hgt_idx"], d["hgt"]) for d in locations}
    assert by_lat_hgt == {
        (1.0, 0.0): (0, 100.0),
        (2.0, 0.0): (1, 200.0),
        (2.0, 1.0): (1, 200.0),
    }


def test_paths_dir_default_matches_the_legacy_layout(cfg):
    # The single most important regression guard for this axis: omitting
    # conversion= (or passing SpeciesConversion.CH4_CO explicitly) must
    # reproduce the exact pre-SpeciesConversion path - every already-saved
    # CH4-to-CO run on disk depends on this staying true.
    path = io.paths_dir(cfg, GridResolution.SAMPLE)
    assert path.name == "kin_mdh0_tK1"
    assert path.parent.name == "tinyplanet"
    assert path.parent.parent.name == "sample"
    assert path.parent.parent.parent.name == "ch4_co_paths"
    assert path == io.paths_dir(cfg, GridResolution.SAMPLE, SpeciesConversion.CH4_CO)


def test_ch4_co_paths_does_not_cross_contaminate_different_directions(tmp_path, monkeypatch, cfg):
    # A CH4-to-CO and a CO-to-CH4 run for the exact same cfg/grid_resolution
    # must never mix per-level files.
    monkeypatch.setattr(io, "path_to_intermediates", tmp_path)

    io.save_paths_level(
        [{"pres": 1.0, "lat": 0.0, "lon": 0.0, "segments": ["ch4_co"]}],
        cfg,
        GridResolution.SAMPLE,
        level_index=0,
        level_height=100.0,
        conversion=SpeciesConversion.CH4_CO,
    )
    io.save_paths_level(
        [{"pres": 1.0, "lat": 0.0, "lon": 0.0, "segments": ["co_ch4"]}],
        cfg,
        GridResolution.SAMPLE,
        level_index=0,
        level_height=100.0,
        conversion=SpeciesConversion.CO_CH4,
    )

    ch4_co_locations = io.load_paths_from_disk(
        cfg, GridResolution.SAMPLE, conversion=SpeciesConversion.CH4_CO
    )
    co_ch4_locations = io.load_paths_from_disk(
        cfg, GridResolution.SAMPLE, conversion=SpeciesConversion.CO_CH4
    )
    assert [loc["segments"] for loc in ch4_co_locations] == [["ch4_co"]]
    assert [loc["segments"] for loc in co_ch4_locations] == [["co_ch4"]]
    assert io.paths_dir(cfg, GridResolution.SAMPLE, SpeciesConversion.CH4_CO) != (
        io.paths_dir(cfg, GridResolution.SAMPLE, SpeciesConversion.CO_CH4)
    )


def test_load_paths_from_disk_raises_when_nothing_saved(tmp_path, monkeypatch, cfg):
    monkeypatch.setattr(io, "path_to_intermediates", tmp_path)
    with pytest.raises(FileNotFoundError):
        io.load_paths_from_disk(cfg, GridResolution.SAMPLE)


def test_save_paths_level_writes_a_header_line(tmp_path, monkeypatch, cfg):
    # First line is a header record (cfg identity + this level's
    # hgt_idx/hgt) - not a location dict - so a per-line read of the rest
    # of the file never has to repeat them.
    monkeypatch.setattr(io, "path_to_intermediates", tmp_path)

    io.save_paths_level(
        [{"pres": 1.0, "lat": 0.0, "lon": 0.0, "segments": []}],
        cfg,
        GridResolution.SAMPLE,
        level_index=3,
        level_height=1234.5,
    )

    [saved] = io.paths_dir(cfg, GridResolution.SAMPLE).glob("*.jsonl")
    with open(saved) as f:
        header = json.loads(f.readline())
        location = json.loads(f.readline())
        assert f.readline() == ""  # exactly one location line after the header

    assert header == {
        "planet": cfg.planet,
        "exp": cfg.exp,
        "mdh": cfg.mdh,
        "tK": cfg.tK,
        "grid_resolution": "sample",
        "conversion": "ch4_co",
        "hgt_idx": 3,
        "hgt": 1234.5,
    }
    assert "hgt_idx" not in location
    assert "hgt" not in location
    assert "planet" not in location
    assert "exp" not in location


# ---------------------------------------------------------------------
# stage 03: save_converted/load_converted
# ---------------------------------------------------------------------


def test_save_and_load_converted_roundtrip(tmp_path, monkeypatch, cfg):
    monkeypatch.setattr(io, "_CONVERTED_DIR", tmp_path)

    cubes = iris.cube.CubeList(
        [
            iris.cube.Cube(250.0, long_name="air_temperature", units="K"),
            iris.cube.Cube(1e10, long_name="CH4 number density", units="cm-3"),
        ]
    )
    io.save_converted(cubes, cfg)

    saved = list(tmp_path.glob("*.nc"))
    assert [p.name for p in saved] == [io.converted_filename(cfg)]

    loaded = io.load_converted(cfg)
    assert float(loaded.extract_cube("air_temperature").data) == 250.0
    assert float(loaded.extract_cube("CH4 number density").data) == 1e10


# ---------------------------------------------------------------------
# merge_paths: save_merged_paths/load_merged_paths
# ---------------------------------------------------------------------


def test_merged_ch4_co_paths_nests_by_resolution_and_roundtrips(tmp_path, monkeypatch, cfg):
    # sample/full must write to separate subdirectories.
    monkeypatch.setattr(io, "path_to_results", tmp_path)

    def _cubes():
        temp_cube = iris.cube.Cube([1000.0], long_name="temp", var_name="temp", units="K")
        pres_cube = iris.cube.Cube([1e5], var_name="pres", units="Pa")
        path_index_cube = iris.cube.Cube([1], var_name="path_index", units="1")
        net_rate_inv_cube = iris.cube.Cube([7.0], var_name="net_rate_inv", units="s")
        max_rate_inv_cube = iris.cube.Cube([5.0], var_name="max_rate_inv", units="s")
        return temp_cube, pres_cube, path_index_cube, net_rate_inv_cube, max_rate_inv_cube

    for gr in (GridResolution.SAMPLE, GridResolution.FULL):
        temp_cube, pres_cube, path_index_cube, net_rate_inv_cube, max_rate_inv_cube = _cubes()
        io.save_merged_paths(
            temp_cube,
            pres_cube,
            path_index_cube,
            net_rate_inv_cube,
            max_rate_inv_cube,
            {1: {"eqns": ["X"], "directions": ["forward"], "nodes": ["CH4", "CO"]}},
            cfg,
            gr,
        )

    sample_dir = io.merged_paths_dir(GridResolution.SAMPLE)
    full_dir = io.merged_paths_dir(GridResolution.FULL)
    assert sample_dir != full_dir
    assert [p.name for p in sample_dir.glob("*.nc")] == [
        io.merged_paths_filename(cfg, GridResolution.SAMPLE)
    ]
    assert [p.name for p in full_dir.glob("*.nc")] == [
        io.merged_paths_filename(cfg, GridResolution.FULL)
    ]

    loaded_temp, _, _, _, _, path_index_map = io.load_merged_paths(cfg, GridResolution.SAMPLE)
    assert float(loaded_temp.data[0]) == 1000.0
    assert path_index_map == {1: {"eqns": ["X"], "directions": ["forward"], "nodes": ["CH4", "CO"]}}


def test_merged_ch4_co_paths_separates_directions_and_roundtrips(tmp_path, monkeypatch, cfg):
    monkeypatch.setattr(io, "path_to_results", tmp_path)

    def _cubes(value):
        temp_cube = iris.cube.Cube([value], long_name="temp", var_name="temp", units="K")
        pres_cube = iris.cube.Cube([1e5], var_name="pres", units="Pa")
        path_index_cube = iris.cube.Cube([1], var_name="path_index", units="1")
        net_rate_inv_cube = iris.cube.Cube([7.0], var_name="net_rate_inv", units="s")
        max_rate_inv_cube = iris.cube.Cube([5.0], var_name="max_rate_inv", units="s")
        return temp_cube, pres_cube, path_index_cube, net_rate_inv_cube, max_rate_inv_cube

    for conversion, value in [
        (SpeciesConversion.CH4_CO, 1000.0),
        (SpeciesConversion.CO_CH4, 2000.0),
    ]:
        io.save_merged_paths(
            *_cubes(value),
            {1: {"eqns": ["X"], "directions": ["forward"], "nodes": ["CH4", "CO"]}},
            cfg,
            GridResolution.SAMPLE,
            conversion=conversion,
        )

    ch4_co_dir = io.merged_paths_dir(GridResolution.SAMPLE, SpeciesConversion.CH4_CO)
    co_ch4_dir = io.merged_paths_dir(GridResolution.SAMPLE, SpeciesConversion.CO_CH4)
    assert ch4_co_dir != co_ch4_dir

    loaded_ch4_co, *_rest = io.load_merged_paths(
        cfg, GridResolution.SAMPLE, conversion=SpeciesConversion.CH4_CO
    )
    loaded_co_ch4, *_rest = io.load_merged_paths(
        cfg, GridResolution.SAMPLE, conversion=SpeciesConversion.CO_CH4
    )
    assert float(loaded_ch4_co.data[0]) == 1000.0
    assert float(loaded_co_ch4.data[0]) == 2000.0
    # self-describing on this axis too (never read back by
    # load_merged_paths() itself, see its own docstring).
    assert loaded_ch4_co.attributes["conversion"] == "ch4_co"
    assert loaded_co_ch4.attributes["conversion"] == "co_ch4"


# ---------------------------------------------------------------------
# stage 07: save_tau_adv/load_tau_adv (adv_kind="ls"|"qn"),
# save_tau_chem/load_tau_chem - saved to separate files/directories
# (not one combined tau.nc) since tau_adv_ls/tau_adv_qn/tau_ch4_co_* aren't
# always computable together (see quenchpath.stages.stage_07_taus's
# calc_and_save_adv_qn()/
# calc_and_save_chem_timescale() docstrings).
# ---------------------------------------------------------------------


def _tau_family_result(keys, pres_name=None, pres_values=None):
    # level_height + (optionally) a pres_mean/pres aux coord.
    # calc_and_save_adv_ls() attaches
    # pres_mean; calc_and_save_adv_qn() attaches
    # no aux coordinate at all any more (see its own docstring - pres
    # would just be an unmodified raw-run passthrough, reloaded directly
    # at plotting time instead) - pres_name=None (the default) for that
    # family's test.
    level_coord = iris.coords.DimCoord([1000.0, 2000.0], long_name="level_height", units="m")
    pres_coord = None
    if pres_name is not None:
        pres_coord = iris.coords.AuxCoord(pres_values, long_name=pres_name, units="Pa")
    result = {}
    for i, key in enumerate(keys):
        cube = iris.cube.Cube(
            [float(i), float(i) * 10], dim_coords_and_dims=[(level_coord.copy(), 0)], units="s"
        )
        if pres_coord is not None:
            cube.add_aux_coord(pres_coord.copy(), 0)
        result[key] = cube
    return result


def test_tau_adv_filename_uses_species_suffix(cfg):
    # tau_adv_qn is genuinely computed per-species (see
    # quenchpath.stages.stage_07_taus.calc_and_save_adv_qn()'s own
    # docstring) - a plain species string, not a
    # quenchpath.config.SpeciesConversion (tau_adv_qn doesn't care which
    # pathway-search direction a caller might compare it against, only
    # which single species it's about). "CH4" is the default.
    # Lowercased in the filename (matching every other on-disk name's own
    # lowercase convention) even though `species` itself is uppercase.
    assert io.tau_adv_filename(cfg, "ls") == "tinyplanet_kin_mdh0_tK1_full_tau_adv_ls.nc"
    assert io.tau_adv_filename(cfg, "qn") == "tinyplanet_kin_mdh0_tK1_full_tau_adv_qn_ch4.nc"
    assert io.tau_adv_filename(cfg, "qn", "CH4") == "tinyplanet_kin_mdh0_tK1_full_tau_adv_qn_ch4.nc"
    assert io.tau_adv_filename(cfg, "qn", "CO") == "tinyplanet_kin_mdh0_tK1_full_tau_adv_qn_co.nc"


def test_tau_adv_filename_ls_ignores_species(cfg):
    # tau_adv_ls has no species dependence of its own (see
    # calc_adv_ls()'s own docstring) - unlike
    # qn, every species must give the exact same filename.
    assert io.tau_adv_filename(cfg, "ls", "CH4") == io.tau_adv_filename(cfg, "ls", "CO")


def test_tau_adv_filename_rejects_unknown_adv_kind(cfg):
    with pytest.raises(ValueError, match="adv_kind"):
        io.tau_adv_filename(cfg, "bogus")


def test_save_and_load_tau_adv_ls_roundtrip(tmp_path, monkeypatch, cfg):
    monkeypatch.setattr(io, "_TAU_ADV_LS_DIR", tmp_path)

    result = _tau_family_result(io._TAU_ADV_LS_KEYS, "pres_mean", [1e5, 1e4])
    io.save_tau_adv(result, cfg, "ls")

    loaded = io.load_tau_adv(cfg, "ls")
    assert set(loaded) == set(io._TAU_ADV_LS_KEYS)
    for key in loaded:
        np.testing.assert_array_equal(loaded[key].data, result[key].data)
        np.testing.assert_array_equal(
            loaded[key].coord("pres_mean").points, result[key].coord("pres_mean").points
        )


def test_save_and_load_tau_adv_qn_roundtrip(tmp_path, monkeypatch, cfg):
    monkeypatch.setattr(io, "_TAU_ADV_QN_DIR", tmp_path)

    result = _tau_family_result(io._TAU_ADV_QN_KEYS)
    io.save_tau_adv(result, cfg, "qn")

    loaded = io.load_tau_adv(cfg, "qn")
    assert set(loaded) == set(io._TAU_ADV_QN_KEYS)
    for key in loaded:
        np.testing.assert_array_equal(loaded[key].data, result[key].data)
        # No pres aux coordinate - see calc_and_save_adv_qn()'s own
        # docstring for why.
        assert loaded[key].coords("pres") == []


def test_save_and_load_tau_adv_qn_roundtrip_separates_by_species(tmp_path, monkeypatch, cfg):
    monkeypatch.setattr(io, "_TAU_ADV_QN_DIR", tmp_path)

    ch4_result = _tau_family_result(io._TAU_ADV_QN_KEYS)
    co_result = _tau_family_result(io._TAU_ADV_QN_KEYS)
    for key in co_result:
        # +100, not *10 - _tau_family_result()'s first key starts at
        # [0.0, 0.0], which *10 wouldn't distinguish from ch4_result's
        # own zero.
        co_result[key] = co_result[key] + 100.0

    io.save_tau_adv(ch4_result, cfg, "qn", species="CH4")
    io.save_tau_adv(co_result, cfg, "qn", species="CO")

    loaded_ch4 = io.load_tau_adv(cfg, "qn", species="CH4")
    loaded_co = io.load_tau_adv(cfg, "qn", species="CO")

    for key in io._TAU_ADV_QN_KEYS:
        np.testing.assert_array_equal(loaded_ch4[key].data, ch4_result[key].data)
        np.testing.assert_array_equal(loaded_co[key].data, co_result[key].data)
        assert not np.array_equal(loaded_ch4[key].data, loaded_co[key].data)


def test_load_tau_adv_rejects_unknown_adv_kind(cfg):
    with pytest.raises(ValueError, match="adv_kind"):
        io.load_tau_adv(cfg, "bogus")


def test_save_tau_adv_rejects_unknown_adv_kind(cfg):
    with pytest.raises(ValueError, match="adv_kind"):
        io.save_tau_adv({}, cfg, "bogus")


def test_save_and_load_tau_chem_roundtrip_separates_rls_and_net(tmp_path, monkeypatch, cfg):
    monkeypatch.setattr(io, "path_to_results", tmp_path)

    rls_cube = iris.cube.Cube([1.0, 2.0], var_name="tau_ch4_co_rls", units="s")
    net_cube = iris.cube.Cube([3.0, 4.0], var_name="tau_ch4_co_net", units="s")
    io.save_tau_chem(rls_cube, cfg, GridResolution.FULL, "rls")
    io.save_tau_chem(net_cube, cfg, GridResolution.FULL, "net")

    rls_dir = io.tau_chem_dir("rls", GridResolution.FULL)
    net_dir = io.tau_chem_dir("net", GridResolution.FULL)
    assert rls_dir != net_dir

    loaded_rls = io.load_tau_chem(cfg, GridResolution.FULL, "rls")
    loaded_net = io.load_tau_chem(cfg, GridResolution.FULL, "net")
    np.testing.assert_array_equal(loaded_rls.data, [1.0, 2.0])
    np.testing.assert_array_equal(loaded_net.data, [3.0, 4.0])


def test_tau_chem_dir_rejects_unknown_chem_kind():
    with pytest.raises(AssertionError):
        io.tau_chem_dir("bogus", GridResolution.FULL)


def test_tau_chem_dir_default_matches_the_legacy_layout(cfg):
    # Same regression guard as test_paths_dir_default_matches_the_
    # legacy_layout() above, for the tau_ch4_co_<chem_kind> tree - every
    # already-saved CH4-to-CO tau file on disk depends on this.
    path = io.tau_chem_dir("rls", GridResolution.FULL)
    assert path.name == "tau_ch4_co_rls"
    assert path == io.tau_chem_dir("rls", GridResolution.FULL, SpeciesConversion.CH4_CO)
    assert io.tau_chem_filename(cfg, GridResolution.FULL, "rls") == (
        "tinyplanet_kin_mdh0_tK1_full_tau_ch4_co_rls.nc"
    )


def test_save_and_load_tau_chem_roundtrip_separates_directions(tmp_path, monkeypatch, cfg):
    monkeypatch.setattr(io, "path_to_results", tmp_path)

    ch4_co_cube = iris.cube.Cube([1.0, 2.0], var_name="tau_ch4_co_rls", units="s")
    co_ch4_cube = iris.cube.Cube([3.0, 4.0], var_name="tau_co_ch4_rls", units="s")
    io.save_tau_chem(
        ch4_co_cube, cfg, GridResolution.FULL, "rls", conversion=SpeciesConversion.CH4_CO
    )
    io.save_tau_chem(
        co_ch4_cube, cfg, GridResolution.FULL, "rls", conversion=SpeciesConversion.CO_CH4
    )

    ch4_co_dir = io.tau_chem_dir("rls", GridResolution.FULL, SpeciesConversion.CH4_CO)
    co_ch4_dir = io.tau_chem_dir("rls", GridResolution.FULL, SpeciesConversion.CO_CH4)
    assert ch4_co_dir != co_ch4_dir
    assert io.tau_chem_filename(cfg, GridResolution.FULL, "rls", SpeciesConversion.CO_CH4) == (
        "tinyplanet_kin_mdh0_tK1_full_tau_co_ch4_rls.nc"
    )

    loaded_ch4_co = io.load_tau_chem(cfg, GridResolution.FULL, "rls", SpeciesConversion.CH4_CO)
    loaded_co_ch4 = io.load_tau_chem(cfg, GridResolution.FULL, "rls", SpeciesConversion.CO_CH4)
    np.testing.assert_array_equal(loaded_ch4_co.data, [1.0, 2.0])
    np.testing.assert_array_equal(loaded_co_ch4.data, [3.0, 4.0])
