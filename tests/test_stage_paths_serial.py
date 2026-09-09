"""Tests for quenchpath.stages.stage_05_paths_serial.select_grid() - the grid-resolution
selection/indexing logic used by calc_paths_serial(). The location-loop
driver itself needs real network+rates data and isn't unit-tested here
(see tests/test_stage_rates.py's module docstring for the same
convention elsewhere in this codebase).
"""

import numpy as np
from iris.coords import DimCoord
from iris.cube import Cube

from quenchpath.config import GridResolution
from quenchpath.stages.stage_05_paths_serial import select_grid


def _temp_cube(n_levels=8, n_lats=6, n_lons=9):
    hgts = np.linspace(1.0e4, 1.0e6, n_levels)
    lats = np.linspace(-80.0, 80.0, n_lats)
    lons = np.linspace(0.0, 350.0, n_lons)
    return Cube(
        np.zeros((n_levels, n_lats, n_lons)),
        dim_coords_and_dims=[
            (DimCoord(hgts, long_name="level_height", units="m"), 0),
            (DimCoord(lats, standard_name="latitude", units="degrees"), 1),
            (DimCoord(lons, standard_name="longitude", units="degrees"), 2),
        ],
    )


def test_select_grid_full_returns_every_native_point():
    cube = _temp_cube(n_levels=8, n_lats=6, n_lons=9)

    hgts, lats, lons, hgt_idx, lat_idx, lon_idx = select_grid(cube, GridResolution.FULL)

    np.testing.assert_array_equal(hgts, cube.coord("level_height").points)
    np.testing.assert_array_equal(lats, cube.coord("latitude").points)
    np.testing.assert_array_equal(lons, cube.coord("longitude").points)
    assert hgt_idx == list(range(8))
    assert lat_idx == list(range(6))
    assert lon_idx == list(range(9))


def test_select_grid_sample_indices_round_trip_to_the_selected_points():
    cube = _temp_cube(n_levels=12, n_lats=90, n_lons=144)

    hgts, lats, lons, hgt_idx, lat_idx, lon_idx = select_grid(cube, GridResolution.SAMPLE)

    full_hgts = cube.coord("level_height").points
    full_lats = cube.coord("latitude").points
    full_lons = cube.coord("longitude").points

    # every selected point's index must index back into the exact value
    # select_grid() reported for it - this is the property calc_paths_serial()
    # actually relies on to index temp_arr/k_eq_arr/etc.
    np.testing.assert_array_equal(full_hgts[hgt_idx], hgts)
    np.testing.assert_array_equal(full_lats[lat_idx], lats)
    np.testing.assert_array_equal(full_lons[lon_idx], lons)

    # sample is every 4th level, per config.sample_grid_points()
    assert len(hgts) == len(full_hgts[::4])
