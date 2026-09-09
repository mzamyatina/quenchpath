import dataclasses

import numpy as np
import pytest

from quenchpath.config import GridResolution, RunConfig, SpeciesConversion, sample_grid_points


def test_run_config_defaults():
    cfg = RunConfig(planet="hd189733b")
    assert cfg.exp == "kin"
    assert cfg.mdh == "0"
    assert cfg.tK == "1"


def test_run_config_is_frozen():
    cfg = RunConfig(planet="hd189733b")
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.planet = "hd189733b"


def test_grid_resolution_values():
    assert GridResolution.SAMPLE.value == "sample"
    assert GridResolution.FULL.value == "full"
    assert GridResolution("full") is GridResolution.FULL


def test_path_direction_values_and_endpoints():
    assert SpeciesConversion.CH4_CO.value == "ch4_co"
    assert SpeciesConversion.CO_CH4.value == "co_ch4"
    assert SpeciesConversion("co_ch4") is SpeciesConversion.CO_CH4

    assert SpeciesConversion.CH4_CO.source == "CH4"
    assert SpeciesConversion.CH4_CO.target == "CO"
    assert SpeciesConversion.CH4_CO.label == "CH4-to-CO"

    assert SpeciesConversion.CO_CH4.source == "CO"
    assert SpeciesConversion.CO_CH4.target == "CH4"
    assert SpeciesConversion.CO_CH4.label == "CO-to-CH4"


def test_sample_grid_points_hgts_is_grid_size_agnostic():
    # Unlike lat/lon below, the vertical stride ([::4]) makes no
    # assumption about how many levels there are.
    hgts = np.linspace(1.0e4, 1.0e6, 23)
    assert sample_grid_points(full_hgts=hgts)["hgts"].tolist() == hgts[::4].tolist()


def test_sample_grid_points_rejects_a_non_native_latitude_count():
    # lats/lons' offsets (0:45:6, 44:46, 47::6, [::9]) are hardcoded to
    # a 90-latitude/144-longitude grid - silently applying them to any
    # other size would select a meaningless subset instead of erroring.
    lats = np.linspace(-90.0, 90.0, 91)
    with pytest.raises(AssertionError, match="90"):
        sample_grid_points(full_lats=lats)


def test_sample_grid_points_rejects_a_non_native_longitude_count():
    lons = np.linspace(0.0, 360.0, 143, endpoint=False)
    with pytest.raises(AssertionError, match="144"):
        sample_grid_points(full_lons=lons)
