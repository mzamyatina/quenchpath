"""Regression test for quenchpath.stages.stage_03_convert.

tests/data/tiny_raw_run.nc and tests/data/tiny_processed_golden.nc are a
real (not synthetic) 2-latitude x 2-longitude x all-66-native-height-levels
slice of hd189733b's actual raw model output (CH4 + CO mole fraction,
air_temperature, air_pressure) and mole_fraction_to_number_density()'s
exact output on that slice.

The full (non-fixture) verification - mole_fraction_to_number_density()
against hd189733b's real output, all 31 cubes, full native grid - was
also done and is bit-for-bit identical to this fixture's coverage; not
re-run here since it needs the multi-GB real raw run file this repo
doesn't ship.
"""

from pathlib import Path

import iris
import numpy as np
from aeolus.const import init_const

from quenchpath.stages.stage_03_convert import mole_fraction_to_number_density

DATA_DIR = Path(__file__).parent / "data"


def test_mole_fraction_to_number_density_matches_golden_output():
    # Golden fixture was captured on macOS/its BLAS build; CI runs on
    # Linux. rtol=1e-10 (not exact equality) accommodates tiny last-few-ULP
    # differences across platforms/BLAS builds, while still being tight
    # enough to catch a real computational error (which would differ by
    # many orders of magnitude more).
    consts = init_const()
    raw_cubes = iris.load(DATA_DIR / "tiny_raw_run.nc")

    gases = mole_fraction_to_number_density(raw_cubes, consts.molar_gas_constant, consts.avogadro)

    golden = iris.load(DATA_DIR / "tiny_processed_golden.nc")
    assert {c.name() for c in gases} == {c.name() for c in golden}
    # CH4 number density, CO number density, air_temperature, air_pressure
    assert len(gases) == 4

    for cube in gases:
        golden_cube = golden.extract_cube(cube.name())
        assert cube.shape == golden_cube.shape
        np.testing.assert_allclose(cube.data, golden_cube.data, rtol=1e-10)


def test_mole_fraction_to_number_density_ignores_non_gas_cubes():
    # air_temperature/air_pressure themselves don't match "<gas> mole
    # fraction" and must not be misinterpreted as a gas to convert - no
    # "air_pressure number density"/"air_temperature number density"
    # should ever appear.
    consts = init_const()
    raw_cubes = iris.load(DATA_DIR / "tiny_raw_run.nc")

    gases = mole_fraction_to_number_density(raw_cubes, consts.molar_gas_constant, consts.avogadro)

    names = {c.name() for c in gases}
    assert "air_pressure number density" not in names
    assert "air_temperature number density" not in names


def test_mole_fraction_to_number_density_passes_through_air_pressure_unchanged():
    # air_pressure is carried through as-is (needed downstream now that
    # nothing relevels it away - see issue #11) - not recomputed, not
    # relabeled onto a different grid.
    consts = init_const()
    raw_cubes = iris.load(DATA_DIR / "tiny_raw_run.nc")
    raw_pressure = raw_cubes.extract_cube("air_pressure")

    gases = mole_fraction_to_number_density(raw_cubes, consts.molar_gas_constant, consts.avogadro)

    out_pressure = gases.extract_cube("air_pressure")
    np.testing.assert_array_equal(out_pressure.data, raw_pressure.data)


def test_mole_fraction_to_number_density_preserves_input_grid_exactly():
    # The whole point of dropping the relevel (issue #11): every output
    # cube must stay on exactly the same native vertical/lat/lon grid the
    # input cubes were already on - points AND bounds, not just shape.
    consts = init_const()
    raw_cubes = iris.load(DATA_DIR / "tiny_raw_run.nc")
    raw_temp = raw_cubes.extract_cube("air_temperature")

    gases = mole_fraction_to_number_density(raw_cubes, consts.molar_gas_constant, consts.avogadro)

    for cube in gases:
        for coord_name in ["level_height", "latitude", "longitude"]:
            out_coord = cube.coord(coord_name)
            in_coord = raw_temp.coord(coord_name)
            np.testing.assert_array_equal(out_coord.points, in_coord.points)
            np.testing.assert_array_equal(out_coord.bounds, in_coord.bounds)
