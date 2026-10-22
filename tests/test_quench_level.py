"""Tests for quenchpath.analysis.quench_level - subset_to_grid_of(), the
merged quench_level() ratio-Cube function (covering both the ls/qn
"adv_kind" axis and the rls/net "chem_kind" axis), and
quench_level_zahnle_ls/qn(). See tests/test_quench_region.py for
quench_region() itself (its own module, downstream of this one).

Real save/load I/O is exercised via quenchpath.io's own directory
constants monkeypatched to a pytest tmp_path (same convention as
tests/test_io.py).
"""

import iris
import numpy as np
import pytest
from iris.coords import AuxCoord, DimCoord
from iris.cube import Cube

from quenchpath import io
from quenchpath.analysis.quench_level import (
    quench_level,
    quench_level_zahnle_ls,
    quench_level_zahnle_qn,
    subset_to_grid_of,
)
from quenchpath.analysis.zahnle_marley_2014 import zahnle_marley_2014_tco
from quenchpath.config import GridResolution, RunConfig, SpeciesConversion, sample_grid_points


@pytest.fixture
def cfg():
    return RunConfig(planet="tinyplanet", exp="kin", mdh="0", tK="1")


def _lat_lon_coords(n_lats, n_lons):
    lats = DimCoord(np.linspace(-90.0, 90.0, n_lats), standard_name="latitude", units="degrees")
    lon_pnts = np.linspace(0.0, 360.0, n_lons, endpoint=False)
    lons = DimCoord(lon_pnts, standard_name="longitude", units="degrees")
    return lats, lons


def _tau_ch4_cube(chem_kind, level_coord, n_lats=2, n_lons=2):
    """A tau_ch4_co_<chem_kind>-shaped Cube (level_height, latitude,
    longitude), sharing `level_coord` with whatever tau_adv cube it's
    meant to divide against - iris only broadcasts a 1D tau_adv_ls_dom
    against a 3D tau_ch4 via a matching level_height coordinate, not
    just a matching shape."""
    n_levels = len(level_coord.points)
    lats, lons = _lat_lon_coords(n_lats, n_lons)
    data = np.arange(n_levels * n_lats * n_lons, dtype=float).reshape(n_levels, n_lats, n_lons) + 1
    return Cube(
        data,
        var_name=f"tau_ch4_co_{chem_kind}",
        units="s",
        dim_coords_and_dims=[(level_coord.copy(), 0), (lats, 1), (lons, 2)],
    )


def _ls_family_result(level_coord):
    """A tau_adv_ls-shaped result dict (io._TAU_ADV_LS_KEYS), each with
    the pres_mean aux coord save_tau_adv(adv_kind="ls") requires - same convention
    as tests/test_io.py's _tau_family_result()."""
    n_levels = len(level_coord.points)
    pres_coord = AuxCoord(np.linspace(1.0e5, 1.0e3, n_levels), long_name="pres_mean", units="Pa")
    result = {}
    for i, key in enumerate(io._TAU_ADV_LS_KEYS):
        data = np.arange(n_levels, dtype=float) + i + 1
        cube = Cube(data, units="s", dim_coords_and_dims=[(level_coord.copy(), 0)])
        cube.add_aux_coord(pres_coord.copy(), 0)
        cube.rename(key)
        result[key] = cube
    return result


def _qn_family_result(level_coord, n_lats=2, n_lons=2):
    """A tau_adv_qn-shaped result dict (io._TAU_ADV_QN_KEYS) - full 3D
    field, no pres aux coord (see calc_and_save_adv_qn()'s own
    docstring)."""
    n_levels = len(level_coord.points)
    lats, lons = _lat_lon_coords(n_lats, n_lons)
    result = {}
    for i, key in enumerate(io._TAU_ADV_QN_KEYS):
        shape = (n_levels, n_lats, n_lons)
        data = np.arange(n_levels * n_lats * n_lons, dtype=float).reshape(shape) + i + 1
        dims = [(level_coord.copy(), 0), (lats.copy(), 1), (lons.copy(), 2)]
        cube = Cube(data, units="s", dim_coords_and_dims=dims)
        cube.rename(key)
        result[key] = cube
    return result


def _pres_temp_cubes(level_coord, n_lats=2, n_lons=2):
    """air_pressure [bar] / air_temperature [K] Cubes on the same
    (level_height, latitude, longitude) grid as _tau_ch4_cube() above -
    what quench_level_zahnle_ls/qn() take in place of a loaded
    tau_ch4_co cube (see their own docstrings for why there's no
    saved-to-disk quantity to load these from instead)."""
    n_levels = len(level_coord.points)
    lats, lons = _lat_lon_coords(n_lats, n_lons)
    shape = (n_levels, n_lats, n_lons)

    def _dims():
        return [(level_coord.copy(), 0), (lats.copy(), 1), (lons.copy(), 2)]

    pres = Cube(
        np.linspace(1.0, 0.1, n_levels * n_lats * n_lons).reshape(shape),
        long_name="air_pressure",
        units="bar",
        dim_coords_and_dims=_dims(),
    )
    temp = Cube(
        np.linspace(500.0, 2000.0, n_levels * n_lats * n_lons).reshape(shape),
        long_name="air_temperature",
        units="K",
        dim_coords_and_dims=_dims(),
    )
    return pres, temp


def _full_grid_cube(n_levels=12, n_lats=90, n_lons=144):
    """A tau_adv_qn-shaped cube (level_height, latitude, longitude) -
    n_lats=90/n_lons=144 matching sample_grid_points()'s own hardcoded
    lat-slicing assumptions (see tests/test_stage_paths_serial.py's
    _temp_cube() for the same convention)."""
    hgts = np.linspace(1.0e4, 1.0e6, n_levels)
    lats = np.linspace(-90.0, 90.0, n_lats)
    lons = np.linspace(0.0, 360.0, n_lons, endpoint=False)
    return Cube(
        np.arange(n_levels * n_lats * n_lons, dtype=float).reshape(n_levels, n_lats, n_lons),
        dim_coords_and_dims=[
            (DimCoord(hgts, long_name="level_height", units="m"), 0),
            (DimCoord(lats, standard_name="latitude", units="degrees"), 1),
            (DimCoord(lons, standard_name="longitude", units="degrees"), 2),
        ],
    )


def _profile_cube(n_levels=12):
    """A bare level_height-only profile cube - for subset_to_grid_of()
    itself, which doesn't care about pres_mean."""
    hgts = np.linspace(1.0e4, 1.0e6, n_levels)
    return Cube(
        np.arange(n_levels, dtype=float),
        dim_coords_and_dims=[(DimCoord(hgts, long_name="level_height", units="m"), 0)],
    )


def _sample_reference_cube(full_hgts, full_lats=None, full_lons=None):
    """A cube built directly at GridResolution.SAMPLE's own points
    (mirrors a merged-pathway tau_ch4 cube, computed from scratch at
    SAMPLE resolution - not derived by subsetting a FULL cube, so
    subset_to_grid_of() tests below aren't circular) - the independent
    reference grid subset_to_grid_of() aligns onto."""
    sample = sample_grid_points(full_hgts=full_hgts, full_lats=full_lats, full_lons=full_lons)
    dim_coords_and_dims = [(DimCoord(sample["hgts"], long_name="level_height", units="m"), 0)]
    shape = [len(sample["hgts"])]
    if full_lats is not None:
        dim_coords_and_dims.append(
            (DimCoord(sample["lats"], standard_name="latitude", units="degrees"), len(shape))
        )
        shape.append(len(sample["lats"]))
    if full_lons is not None:
        dim_coords_and_dims.append(
            (DimCoord(sample["lons"], standard_name="longitude", units="degrees"), len(shape))
        )
        shape.append(len(sample["lons"]))
    return Cube(np.zeros(shape), dim_coords_and_dims=dim_coords_and_dims)


def test_subset_to_grid_of_on_a_1d_profile():
    cube = _profile_cube(n_levels=12)
    full_hgts = cube.coord("level_height").points
    reference = _sample_reference_cube(full_hgts)

    subset = subset_to_grid_of(cube, reference)

    expected_hgts = sample_grid_points(full_hgts=full_hgts)["hgts"]
    np.testing.assert_array_equal(subset.coord("level_height").points, expected_hgts)
    np.testing.assert_array_equal(subset.data, cube.data[::4])


def test_subset_to_grid_of_on_a_full_3d_field():
    cube = _full_grid_cube(n_levels=12, n_lats=90, n_lons=144)
    full_hgts = cube.coord("level_height").points
    full_lats = cube.coord("latitude").points
    full_lons = cube.coord("longitude").points
    reference = _sample_reference_cube(full_hgts, full_lats, full_lons)

    subset = subset_to_grid_of(cube, reference)

    expected = sample_grid_points(full_hgts=full_hgts, full_lats=full_lats, full_lons=full_lons)
    np.testing.assert_array_equal(subset.coord("level_height").points, expected["hgts"])
    np.testing.assert_array_equal(subset.coord("latitude").points, expected["lats"])
    np.testing.assert_array_equal(subset.coord("longitude").points, expected["lons"])
    assert subset.shape == (len(expected["hgts"]), len(expected["lats"]), len(expected["lons"]))


def test_subset_to_grid_of_is_a_noop_when_already_aligned():
    # A FULL-resolution reference sharing cube's own points: nothing to
    # subset, so the exact same object comes back, not just equal data.
    cube = _profile_cube(n_levels=12)
    reference = _profile_cube(n_levels=12)

    assert subset_to_grid_of(cube, reference) is cube


def test_subset_to_grid_of_is_idempotent():
    # sample_grid_points()'s [::4] isn't idempotent on its own (a
    # second [::4] would over-slice) - subset_to_grid_of() must still
    # be, by comparing against the reference's actual points rather
    # than reapplying the formula blindly. This is the property that
    # replaced the old subset_to_sample_grid()'s base_dir-is-None guard.
    cube = _profile_cube(n_levels=12)
    full_hgts = cube.coord("level_height").points
    reference = _sample_reference_cube(full_hgts)

    once = subset_to_grid_of(cube, reference)
    twice = subset_to_grid_of(once, reference)

    assert twice is once


def test_subset_to_grid_of_ignores_collapsed_scalar_lat_lon():
    # A tau_adv_ls_dom-shaped cube: level_height is the only real
    # dimension; latitude/longitude are scalar AuxCoords left over from
    # spatial_mean()'s own .collapsed() call, not real axes to subset -
    # must be skipped even though the reference here has genuine lat/lon
    # dimensions of its own.
    cube = _profile_cube(n_levels=12)
    cube.add_aux_coord(AuxCoord([0.0], standard_name="latitude", units="degrees"))
    cube.add_aux_coord(AuxCoord([0.0], standard_name="longitude", units="degrees"))
    full_hgts = cube.coord("level_height").points
    full_lats = np.linspace(-90.0, 90.0, 90)
    full_lons = np.linspace(0.0, 360.0, 144, endpoint=False)
    reference = _sample_reference_cube(full_hgts, full_lats, full_lons)

    subset = subset_to_grid_of(cube, reference)

    np.testing.assert_array_equal(subset.coord("latitude").points, [0.0])
    np.testing.assert_array_equal(subset.coord("longitude").points, [0.0])


def test_subset_to_grid_of_raises_on_no_overlap():
    cube = _profile_cube(n_levels=12)
    unrelated_reference = Cube(
        np.zeros(3),
        dim_coords_and_dims=[(DimCoord([1.0, 2.0, 3.0], long_name="level_height", units="m"), 0)],
    )

    with pytest.raises(AssertionError):
        subset_to_grid_of(cube, unrelated_reference)


@pytest.mark.parametrize("chem_kind", ["rls", "net"])
def test_quench_level_ls_divides_its_own_tau_ch4_by_tau_adv_ls(
    tmp_path, monkeypatch, cfg, chem_kind
):
    monkeypatch.setattr(io, "path_to_results", tmp_path)
    monkeypatch.setattr(io, "_TAU_ADV_LS_DIR", tmp_path / "tau_adv_ls")

    level_coord = DimCoord([1000.0, 2000.0], long_name="level_height", units="m")
    tau_ch4_cube = _tau_ch4_cube(chem_kind, level_coord)
    io.save_tau_chem(tau_ch4_cube, cfg, GridResolution.FULL, chem_kind)
    io.save_tau_adv(_ls_family_result(level_coord), cfg, "ls")

    ratio = quench_level(cfg, GridResolution.FULL, chem_kind, "ls")

    tau_adv_data = io.load_tau_adv(cfg, "ls")["tau_adv_ls_dom"].data
    expected = tau_ch4_cube.data / tau_adv_data[:, None, None]
    np.testing.assert_allclose(ratio.data, expected)


def test_quench_level_ls_aligns_tau_adv_onto_tau_ch4s_own_grid(tmp_path, monkeypatch, cfg):
    monkeypatch.setattr(io, "path_to_results", tmp_path)
    monkeypatch.setattr(io, "_TAU_ADV_LS_DIR", tmp_path / "tau_adv_ls")

    # tau_adv_ls_dom always saved at its full native resolution (12
    # levels) - see _aligned_tau_adv()'s own docstring for why.
    # sample_grid_points()'s hardcoded [::4] means a SAMPLE-resolution
    # tau_ch4 needs 3 levels (12 // 4) to line up with it.
    full_level_coord = DimCoord(np.linspace(1.0e4, 1.0e6, 12), long_name="level_height", units="m")
    io.save_tau_adv(_ls_family_result(full_level_coord), cfg, "ls")
    io.save_tau_chem(_tau_ch4_cube("net", full_level_coord), cfg, GridResolution.FULL, "net")

    sample_hgts = sample_grid_points(full_hgts=full_level_coord.points)["hgts"]
    sample_level_coord = DimCoord(sample_hgts, long_name="level_height", units="m")
    sample_tau_ch4 = _tau_ch4_cube("net", sample_level_coord)
    io.save_tau_chem(sample_tau_ch4, cfg, GridResolution.SAMPLE, "net")

    full_ratio = quench_level(cfg, GridResolution.FULL, "net", "ls")
    sample_ratio = quench_level(cfg, GridResolution.SAMPLE, "net", "ls")

    assert full_ratio.shape[0] == 12
    assert sample_ratio.shape[0] == 3  # 12 // 4, per sample_grid_points()'s [::4]


def test_quench_level_ls_is_a_noop_on_already_pre_shaped_bundled_data(tmp_path, monkeypatch, cfg):
    # A bundled example (base_dir given) is already pre-shaped to
    # SAMPLE resolution (see scripts/generate_example_data.py) - both
    # tau_adv and tau_ch4 already share this 3-level grid, so
    # subset_to_grid_of() must find nothing to do rather than slicing
    # an already-3-level profile with [::4] a second time.
    base_dir = tmp_path / "example_data"
    monkeypatch.setattr(io, "path_to_results", base_dir)

    level_coord = DimCoord([1000.0, 2000.0, 3000.0], long_name="level_height", units="m")
    # Not io.save_tau_adv() - it always writes the real pipeline's own
    # FULL-tagged filename (tau_adv has no genuine SAMPLE variant
    # there). A bundled example's own SAMPLE-tagged copy is written
    # directly here instead, mirroring what
    # scripts/generate_example_data.py's own _save_subsampled_tau_adv()
    # actually does.
    ls_out_dir = base_dir / "taus" / "sample" / "tau_adv_ls"
    ls_out_dir.mkdir(parents=True)
    ls_fname = io.tau_adv_filename(cfg, "ls", grid_resolution=GridResolution.SAMPLE)
    ls_cubes = io._cubes_with_shared_coords(
        _ls_family_result(level_coord), io._TAU_ADV_LS_KEYS, "pres_mean"
    )
    iris.save(ls_cubes, ls_out_dir / ls_fname)
    tau_ch4_cube = _tau_ch4_cube("net", level_coord)
    io.save_tau_chem(tau_ch4_cube, cfg, GridResolution.SAMPLE, "net")

    ratio = quench_level(cfg, GridResolution.SAMPLE, "net", "ls", base_dir=base_dir)

    assert ratio.shape[0] == 3


def test_quench_level_zahnle_ls_divides_tco_by_tau_adv_ls(tmp_path, monkeypatch, cfg):
    monkeypatch.setattr(io, "path_to_results", tmp_path)
    monkeypatch.setattr(io, "_TAU_ADV_LS_DIR", tmp_path / "tau_adv_ls")

    level_coord = DimCoord([1000.0, 2000.0], long_name="level_height", units="m")
    io.save_tau_adv(_ls_family_result(level_coord), cfg, "ls")
    pres, temp = _pres_temp_cubes(level_coord)

    ratio = quench_level_zahnle_ls(cfg, pres, temp, m=3.0)

    tco = zahnle_marley_2014_tco(pres, temp, m=3.0)
    tau_adv_data = io.load_tau_adv(cfg, "ls")["tau_adv_ls_dom"].data
    expected = tco.data / tau_adv_data[:, None, None]
    np.testing.assert_allclose(ratio.data, expected)


@pytest.mark.parametrize("conversion", [SpeciesConversion.CH4_CO, SpeciesConversion.CO_CH4])
@pytest.mark.parametrize("chem_kind", ["rls", "net"])
def test_quench_level_qn_divides_its_own_tau_ch4_by_tau_adv_qn(
    tmp_path, monkeypatch, cfg, chem_kind, conversion
):
    monkeypatch.setattr(io, "path_to_results", tmp_path)
    monkeypatch.setattr(io, "_TAU_ADV_QN_DIR", tmp_path / "tau_adv_qn")
    # tau_adv_qn only ever exists for an "equ" run - cfg itself stays
    # "kin" throughout, so this also exercises quench_level() correctly
    # overriding cfg.exp to look it up. Parametrized over both
    # conversions - tau_adv_qn is genuinely computed per-conversion
    # (CH4's own mole fraction for CH4_CO, CO's own for CO_CH4 - see
    # quenchpath.stages.stage_07_taus.calc_and_save_adv_qn()'s own
    # docstring), not CH4-only.
    equ_cfg = RunConfig(planet=cfg.planet, exp="equ", mdh=cfg.mdh, tK=cfg.tK)

    level_coord = DimCoord([1000.0, 2000.0], long_name="level_height", units="m")
    tau_ch4_cube = _tau_ch4_cube(chem_kind, level_coord)
    io.save_tau_chem(tau_ch4_cube, cfg, GridResolution.FULL, chem_kind, conversion=conversion)
    # save_tau_adv()/load_tau_adv() take a plain species string, not a
    # SpeciesConversion (tau_adv_qn doesn't care about pathway-search
    # direction, only which single species it's about) - quench_level()
    # itself does the conversion.source translation internally.
    io.save_tau_adv(_qn_family_result(level_coord), equ_cfg, "qn", species=conversion.source)

    ratio = quench_level(cfg, GridResolution.FULL, chem_kind, "qn", conversion=conversion)

    tau_adv_data = io.load_tau_adv(equ_cfg, "qn", species=conversion.source)["tau_adv_qn"].data
    expected = tau_ch4_cube.data / tau_adv_data
    np.testing.assert_allclose(ratio.data, expected)


def test_quench_level_rejects_unknown_adv_kind(cfg):
    with pytest.raises(ValueError, match="adv_kind"):
        quench_level(cfg, GridResolution.FULL, "net", "bogus")


def test_quench_level_rejects_unknown_chem_kind(tmp_path, monkeypatch, cfg):
    # Not checked by quench_level() itself - deferred to
    # load_tau_chem()/tau_chem_dir()'s own existing check, so this is
    # AssertionError (matching test_tau_chem_dir_rejects_unknown_chem_kind
    # in tests/test_io.py), not ValueError like an unknown adv_kind.
    monkeypatch.setattr(io, "path_to_results", tmp_path)
    with pytest.raises(AssertionError, match="chem_kind"):
        quench_level(cfg, GridResolution.FULL, "bogus", "ls")


def test_quench_level_zahnle_qn_divides_tco_by_tau_adv_qn(tmp_path, monkeypatch, cfg):
    monkeypatch.setattr(io, "path_to_results", tmp_path)
    monkeypatch.setattr(io, "_TAU_ADV_QN_DIR", tmp_path / "tau_adv_qn")
    equ_cfg = RunConfig(planet=cfg.planet, exp="equ", mdh=cfg.mdh, tK=cfg.tK)

    level_coord = DimCoord([1000.0, 2000.0], long_name="level_height", units="m")
    # species="CO" - quench_level_zahnle_qn()'s own default is CO_CH4
    # (Zahnle & Marley's own t_CO fit is calibrated to CO->CH4 quench
    # points specifically - see its own docstring), not CH4_CO.
    io.save_tau_adv(_qn_family_result(level_coord), equ_cfg, "qn", species="CO")
    pres, temp = _pres_temp_cubes(level_coord)

    ratio = quench_level_zahnle_qn(cfg, pres, temp, m=3.0)

    tco = zahnle_marley_2014_tco(pres, temp, m=3.0)
    tau_adv_data = io.load_tau_adv(equ_cfg, "qn", species="CO")["tau_adv_qn"].data
    expected = tco.data / tau_adv_data
    np.testing.assert_allclose(ratio.data, expected)


@pytest.mark.parametrize("conversion", [SpeciesConversion.CH4_CO, SpeciesConversion.CO_CH4])
def test_quench_level_zahnle_qn_conversion_picks_the_right_species(
    tmp_path, monkeypatch, cfg, conversion
):
    # quenchpath-apps issue #22 (fixed): quench_level_zahnle_qn() used to have no
    # conversion parameter at all, always loading CH4's own tau_adv_qn
    # regardless of what a caller was actually analysing. Only
    # `conversion.source`'s own species file is ever saved here - the
    # pre-fix code would try to load CH4's file for conversion=CO_CH4
    # (never saved in this test), failing loudly instead of silently
    # returning a species-mismatched ratio.
    monkeypatch.setattr(io, "path_to_results", tmp_path)
    monkeypatch.setattr(io, "_TAU_ADV_QN_DIR", tmp_path / "tau_adv_qn")
    equ_cfg = RunConfig(planet=cfg.planet, exp="equ", mdh=cfg.mdh, tK=cfg.tK)

    level_coord = DimCoord([1000.0, 2000.0], long_name="level_height", units="m")
    io.save_tau_adv(_qn_family_result(level_coord), equ_cfg, "qn", species=conversion.source)
    pres, temp = _pres_temp_cubes(level_coord)

    ratio = quench_level_zahnle_qn(cfg, pres, temp, m=3.0, conversion=conversion)

    tco = zahnle_marley_2014_tco(pres, temp, m=3.0)
    tau_adv_data = io.load_tau_adv(equ_cfg, "qn", species=conversion.source)["tau_adv_qn"].data
    expected = tco.data / tau_adv_data
    np.testing.assert_allclose(ratio.data, expected)
