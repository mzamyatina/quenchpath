"""Regression tests for quenchpath.stages.stage_07_taus's derivative helpers
(deriv()/d_dphi()/d_dlambda()/d_dz()) against known analytic derivatives
on small synthetic cubes, independent of the real-data verification
already performed for calc_and_save_adv_ls()/
calc_and_save_adv_qn() themselves. Plus a direct
regression test for _load_chem_rate_cubes()'s own species selection
(quenchpath-apps issue #15's stage_07 fix: CH4 number density for CH4-to-CO, CO's
for CO-to-CH4 - a real physical requirement, not a relabeling, and the
one call site assert_number_density()'s species= check exists to guard).
"""

import numpy as np
import pytest
from iris.coords import DimCoord
from iris.cube import Cube, CubeList

from quenchpath import io
from quenchpath.config import GridResolution, RunConfig, SpeciesConversion
from quenchpath.stages.stage_07_taus import (
    _load_chem_rate_cubes,
    d_dlambda,
    d_dphi,
    d_dz,
    deriv,
)


class _FakeModel:
    """Stand-in for aeolus.model.um - just the coordinate-name attributes
    deriv()/d_dphi()/d_dlambda()/d_dz() actually use."""

    x = "longitude"
    y = "latitude"
    z = "height"


class _FakeConst:
    """Stand-in for an aeolus planet-constants object - just
    planet_radius, a scalar Cube (matching the real thing's type)."""

    def __init__(self, radius_m):
        self.planet_radius = Cube(radius_m, units="m")


def test_deriv_linear_function():
    # d(2x + 3)/dx == 2 everywhere.
    x = np.linspace(0.0, 10.0, 21)
    data = 2.0 * x + 3.0
    cube = Cube(data, dim_coords_and_dims=[(DimCoord(x, long_name="x"), 0)])

    result = deriv(cube, "x")

    assert result.shape == cube.shape
    np.testing.assert_allclose(result.data, 2.0, rtol=1e-6)


def test_deriv_quadratic_function():
    # d(x^2)/dx == 2x - checked away from the boundary, where the
    # differentiate-then-interpolate-back approach is least accurate.
    x = np.linspace(0.0, 10.0, 101)
    data = x**2
    cube = Cube(data, dim_coords_and_dims=[(DimCoord(x, long_name="x"), 0)])

    result = deriv(cube, "x")

    interior = slice(5, -5)
    np.testing.assert_allclose(result.data[interior], 2 * x[interior], rtol=1e-3)


def test_d_dz_matches_deriv_on_the_z_coordinate():
    z = np.linspace(1000.0, 5000.0, 21)
    data = 3.0 * z - 7.0
    cube = Cube(data, dim_coords_and_dims=[(DimCoord(z, long_name="height"), 0)])

    result = d_dz(cube, const=_FakeConst(1.0), model=_FakeModel)

    np.testing.assert_allclose(result.data, 3.0, rtol=1e-6)


def test_d_dphi_linear_in_latitude_radians():
    # data = slope * lat_radians -> (1/r) * d(data)/d(phi) == slope / r,
    # constant everywhere.
    slope, radius = 5.0, 2.0e6
    lat_deg = np.linspace(-80.0, 80.0, 33)
    lat_rad = np.radians(lat_deg)
    data = slope * lat_rad
    # units="1" (dimensionless synthetic field), not units-less - d_dphi()'s
    # own convert_units() call needs real (non-"unknown") input units.
    cube = Cube(
        data,
        units="1",
        dim_coords_and_dims=[(DimCoord(lat_deg, standard_name="latitude", units="degrees"), 0)],
    )

    result = d_dphi(cube, const=_FakeConst(radius), model=_FakeModel)

    interior = slice(3, -3)
    np.testing.assert_allclose(result.data[interior], slope / radius, rtol=1e-2)
    # The input cube's own latitude coordinate must be left in degrees,
    # not radians, after the call (the original script explicitly
    # converts back - see d_dphi()'s body).
    assert str(cube.coord("latitude").units) == "degrees"


def test_d_dlambda_linear_in_longitude_radians():
    # data = slope * lon_radians (constant across latitude) ->
    # (1/(r*cos(phi))) * d(data)/d(lambda) == slope / (r * cos(phi)),
    # varying only with latitude.
    slope, radius = 5.0, 2.0e6
    lat_deg = np.array([-30.0, 0.0, 30.0])
    lon_deg = np.linspace(0.0, 350.0, 36)
    lon_rad = np.radians(lon_deg)
    data = np.tile(slope * lon_rad, (len(lat_deg), 1))
    # units="1" - see test_d_dphi_linear_in_latitude_radians()'s own comment.
    cube = Cube(
        data,
        units="1",
        dim_coords_and_dims=[
            (DimCoord(lat_deg, standard_name="latitude", units="degrees"), 0),
            (DimCoord(lon_deg, standard_name="longitude", units="degrees", circular=True), 1),
        ],
    )

    result = d_dlambda(cube, const=_FakeConst(radius), model=_FakeModel)

    # A linear-in-longitude ramp isn't actually periodic (it jumps from
    # ~2*pi back to 0 at the wrap point) - excluding the two boundary
    # points where the circular longitude coordinate wraps avoids that
    # artifact, same rationale as test_deriv_quadratic_function's boundary
    # exclusion.
    interior = slice(1, -1)
    for i, lat in enumerate(lat_deg):
        expected = slope / (radius * np.cos(np.radians(lat)))
        np.testing.assert_allclose(result.data[i, interior], expected, rtol=5e-2)
    assert str(cube.coord("latitude").units) == "degrees"
    assert str(cube.coord("longitude").units) == "degrees"


def test_d_dphi_of_constant_field_is_near_zero():
    lat_deg = np.linspace(-80.0, 80.0, 17)
    # units="1" - see test_d_dphi_linear_in_latitude_radians()'s own comment.
    cube = Cube(
        np.full_like(lat_deg, 42.0),
        units="1",
        dim_coords_and_dims=[(DimCoord(lat_deg, standard_name="latitude", units="degrees"), 0)],
    )
    result = d_dphi(cube, const=_FakeConst(1.0), model=_FakeModel)
    np.testing.assert_allclose(result.data, 0.0, atol=1e-10)


# ---------------------------------------------------------------------
# _load_chem_rate_cubes(): species selection by conversion (quenchpath-apps issue #15)
# ---------------------------------------------------------------------


def _merged_pathway_cubes():
    level_coord = DimCoord([1000.0, 2000.0], long_name="level_height", units="m")
    dims = [(level_coord.copy(), 0)]
    return (
        Cube([200.0, 300.0], dim_coords_and_dims=dims, var_name="temp", units="K"),
        Cube([1e5, 1e4], dim_coords_and_dims=dims, var_name="pres", units="Pa"),
        Cube([1, 1], dim_coords_and_dims=dims, var_name="path_index", units="1"),
        Cube([10.0, 20.0], dim_coords_and_dims=dims, var_name="net_rate_inv", units="cm3 s"),
        Cube([5.0, 8.0], dim_coords_and_dims=dims, var_name="max_rate_inv", units="cm3 s"),
    )


@pytest.mark.parametrize(
    ("conversion", "expected_species", "expected_data"),
    [
        (SpeciesConversion.CH4_CO, "CH4", [1e10, 2e10]),
        (SpeciesConversion.CO_CH4, "CO", [3e10, 4e10]),
    ],
)
def test_load_chem_rate_cubes_picks_conversions_own_species(
    tmp_path, monkeypatch, conversion, expected_species, expected_data
):
    monkeypatch.setattr(io, "path_to_results", tmp_path)
    monkeypatch.setattr(io, "_CONVERTED_DIR", tmp_path)
    cfg = RunConfig(planet="tinyplanet", exp="kin")

    io.save_merged_paths(
        *_merged_pathway_cubes(),
        {1: {"eqns": ["X"], "directions": ["forward"], "nodes": ["CH4", "CO"]}},
        cfg,
        GridResolution.SAMPLE,
        conversion=conversion,
    )
    level_coord = DimCoord([1000.0, 2000.0], long_name="level_height", units="m")
    converted = CubeList(
        [
            Cube(
                [1e10, 2e10],
                dim_coords_and_dims=[(level_coord.copy(), 0)],
                long_name="CH4 number density",
                units="cm-3",
            ),
            Cube(
                [3e10, 4e10],
                dim_coords_and_dims=[(level_coord.copy(), 0)],
                long_name="CO number density",
                units="cm-3",
            ),
        ]
    )
    io.save_converted(converted, cfg)

    net_rate_inv_cube, max_rate_inv_cube, source_nd = _load_chem_rate_cubes(
        cfg, GridResolution.SAMPLE, conversion=conversion
    )

    assert source_nd.name() == f"{expected_species} number density"
    np.testing.assert_array_equal(source_nd.data, expected_data)
