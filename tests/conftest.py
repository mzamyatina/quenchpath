"""Shared pytest fixtures for quenchpath's test suite.

These are small, synthetic-data factories - no real NetCDF files, no
multi-GB inputs. The goal is fixtures cheap enough to run in CI on every
push (see .github/workflows/tests.yml), not fixtures that reproduce real
science at scale. Stage-specific golden-output fixtures (e.g. a tiny
synthetic raw run for stage 03, a captured-known-good processed cube for
stage 04) live alongside their stage's own tests; this file holds the
general-purpose building blocks those are built from.
"""

import iris
import iris.cube
import pytest


@pytest.fixture
def make_temp_cube():
    """Factory fixture: build a scalar air_temperature Cube (units K) for
    a given temperature value, matching the Cube shape quenchpath.chemistry's
    functions expect (a Cube with .core_data()/.copy(), not a bare float).
    """

    def _make(value):
        return iris.cube.Cube(value, long_name="air_temperature", units="K")

    return _make


@pytest.fixture
def make_pres_cube():
    """Factory fixture: build a scalar air_pressure Cube (units bar) for
    a given pressure value - same convention as make_temp_cube() above."""

    def _make(value):
        return iris.cube.Cube(value, long_name="air_pressure", units="bar")

    return _make


@pytest.fixture
def bath_gas_mixture():
    """A CubeList of the 9 bath-gas species quenchpath.chemistry's
    k_kooij_low_pressure_limit and k_troe extract by name (CO, CO2, H2O, CH4,
    H2, N2, He, O-3P, NH3), each a scalar number-density Cube [cm-3].
    Values are arbitrary but equal (1e10 cm-3) so hand-computed expected
    values in tests stay simple.
    """
    species = ["CO", "CO2", "H2O", "CH4", "H2", "N2", "He", "O-3P", "NH3"]
    return iris.cube.CubeList(
        iris.cube.Cube(1e10, long_name=name, units="cm-3") for name in species
    )


@pytest.fixture
def unit_effcs():
    """22 third-body efficiencies, all 1.0 - matches the length k_troe/
    k_kooij_low_pressure_limit index into (effcs[0]..effcs[21]), with every
    efficiency set to 1 so M (the weighted bath-gas number density) is
    just a plain sum of the 9 species in `bath_gas_mixture`.
    """
    return [1.0] * 22
