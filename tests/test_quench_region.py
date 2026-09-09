"""Tests for quenchpath.analysis.quench_region - quench_region() (same 4
combinations as quenchpath.analysis.quench_level.quench_level() itself,
minus the Zahnle ones).

Real save/load I/O is exercised via quenchpath.io's own directory
constants monkeypatched to a pytest tmp_path (same convention as
tests/test_io.py).
"""

import numpy as np
import pytest
from iris.coords import AuxCoord, DimCoord
from iris.cube import Cube

from quenchpath import io
from quenchpath.analysis.quench_level import quench_level
from quenchpath.analysis.quench_region import quench_region
from quenchpath.config import GridResolution, RunConfig


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


def test_quench_region_is_ln_of_quench_level(tmp_path, monkeypatch, cfg):
    # Not parametrized over chem_kind/adv_kind/conversion - quench_region()
    # just forwards them to quench_level() (already covered, combination by
    # combination, by tests/test_quench_level.py's own tests) and takes
    # ln() of the result; the log-wrapping itself doesn't vary with any of
    # the three, so one combination is all this needs to check.
    monkeypatch.setattr(io, "path_to_results", tmp_path)
    monkeypatch.setattr(io, "_TAU_ADV_LS_DIR", tmp_path / "tau_adv_ls")

    level_coord = DimCoord([1000.0, 2000.0], long_name="level_height", units="m")
    tau_ch4_cube = _tau_ch4_cube("net", level_coord)
    io.save_tau_chem(tau_ch4_cube, cfg, GridResolution.FULL, "net")
    io.save_tau_adv(_ls_family_result(level_coord), cfg, "ls")

    log_ratio = quench_region(cfg, GridResolution.FULL, "net", "ls")

    expected = np.log(quench_level(cfg, GridResolution.FULL, "net", "ls").data)
    np.testing.assert_allclose(log_ratio.data, expected)
    assert log_ratio.units == "1"
