"""Tests for quenchpath.stages.stage_06_merge_paths.

Only merge_paths_to_cubes() is unit tested here, with small
synthetic location dicts - matching this project's established
convention of testing pure logic directly rather than mocking
quenchpath.io's real directories (see e.g. tests/test_stage_rates.py's
module docstring). eqn_sequence_of()/assign_path_indices_from_locations() themselves are
tested in tests/test_path_index.py now (they moved to
quenchpath.analysis.path_index - stage_06_merge_paths.py just imports them).
load_paths_from_disk()/merge_and_save_paths() (the I/O
driver half) are verified against real data separately, not unit
tested here.
"""

import iris.coord_systems
import iris.coords
import iris.cube
import numpy as np
import pytest

from quenchpath.stages.stage_06_merge_paths import merge_paths_to_cubes

# Stand-in for io.load_converted(cfg)'s own native grid - real
# coord_system (unlike a coordinate reconstructed from a location
# dict's bare hgt/lat/lon values) is what merge_paths_to_cubes()
# should inherit here. Matches every _location() call below exactly
# (hgt in {1000.0, 2000.0}, lat in {-10.0, 10.0}, lon in {0.0, 180.0}).
_REFERENCE_CUBE = iris.cube.Cube(
    np.zeros((2, 2, 2)),
    dim_coords_and_dims=[
        (iris.coords.DimCoord([1000.0, 2000.0], long_name="level_height", units="m"), 0),
        (
            iris.coords.DimCoord(
                [-10.0, 10.0],
                standard_name="latitude",
                units="degrees",
                coord_system=iris.coord_systems.GeogCS(6371229.0),
            ),
            1,
        ),
        (
            iris.coords.DimCoord(
                [0.0, 180.0],
                standard_name="longitude",
                units="degrees",
                coord_system=iris.coord_systems.GeogCS(6371229.0),
            ),
            2,
        ),
    ],
    var_name="reference",
)


def _segment(eqn, seg_rate_inv, direction="forward"):
    # No "from"/"to" keys - recoverable as zip(nodes, nodes[1:]) if
    # needed; eqn_sequence_of()/merge_paths_to_cubes() never look at
    # either.
    return {"eqn": eqn, "seg_rate_inv": seg_rate_inv, "direction": direction}


def _location(pres, lat, lon, temp, segments, hgt_idx=0, hgt=0.0, nodes=None):
    # net_rate_inv/max_rate_inv mirror find_pathway()/
    # calc_paths_serial()'s own convention: both NaN if segments is empty
    # (no path found) - computed by stage 05 now, not recomputed here,
    # so this helper mirrors that rather than merge_paths_to_cubes()
    # deriving them itself. hgt_idx/hgt default to 0 - eqn_sequence_of()/
    # assign_path_indices_from_locations() don't look at either; only the
    # merge_paths_to_cubes() tests below need real, distinct
    # values, passed explicitly there. `nodes` defaults to one
    # placeholder species per segment boundary - locations sharing the
    # same segments (hence the same path_index) get the same nodes this
    # way automatically, which path_index_map_from_locations() requires. No
    # "planet"/"exp" - this is the shape after
    # load_paths_from_disk() has folded a file's header-line
    # hgt_idx/hgt back in, which never includes them (every caller
    # already has cfg).
    net_rate_inv = sum(seg["seg_rate_inv"] for seg in segments) if segments else float("nan")
    max_rate_inv = max((seg["seg_rate_inv"] for seg in segments), default=float("nan"))
    if nodes is None:
        nodes = [f"N{i}" for i in range(len(segments) + 1)]
    return {
        "hgt_idx": hgt_idx,
        "hgt": hgt,
        "pres": pres,
        "lat": lat,
        "lon": lon,
        "temp": temp,
        "net_rate_inv": net_rate_inv,
        "max_rate_inv": max_rate_inv,
        "nodes": nodes,
        "segments": segments,
    }


def test_merge_paths_to_cubes_builds_expected_grid():
    # A complete 2 (pres) x 2 (lat) x 2 (lon) grid, all sharing one
    # pathway except one distinct location, so both path_index and
    # max_rate_inv vary across the cube.
    eqn_common = "X+Y↔Z+W"
    eqn_other = "A+B↔C+D"
    # bottom (high pres, low hgt) -> level 0, top -> level 1
    hgt_idx_of = {1e5: 0, 1e2: 1}
    hgt_of = {1e5: 1000.0, 1e2: 2000.0}
    locations = []
    for pres in [1e5, 1e2]:
        for lat in [-10.0, 10.0]:
            for lon in [0.0, 180.0]:
                is_special = pres == 1e2 and lat == 10.0 and lon == 180.0
                eqn = eqn_other if is_special else eqn_common
                seg_rate_inv = 99.0 if is_special else 5.0
                temp = 500.0 if pres == 1e5 else 1500.0
                locations.append(
                    _location(
                        pres,
                        lat,
                        lon,
                        temp,
                        [_segment(eqn, seg_rate_inv)],
                        hgt_idx=hgt_idx_of[pres],
                        hgt=hgt_of[pres],
                    )
                )

    (
        temp_cube,
        pres_cube,
        path_index_cube,
        net_rate_inv_cube,
        max_rate_inv_cube,
        seq_to_index,
        identity_map,
    ) = merge_paths_to_cubes(locations, _REFERENCE_CUBE)

    assert temp_cube.shape == (2, 2, 2)
    assert pres_cube.shape == (2, 2, 2)
    assert path_index_cube.shape == (2, 2, 2)
    assert net_rate_inv_cube.shape == (2, 2, 2)
    assert max_rate_inv_cube.shape == (2, 2, 2)
    # ascending, per sort
    assert temp_cube.coord("level_height").points.tolist() == [1000.0, 2000.0]
    assert set(temp_cube.data.ravel().tolist()) == {500.0, 1500.0}
    assert set(pres_cube.data.ravel().tolist()) == {1e5, 1e2}
    # Real coordinate metadata inherited from _REFERENCE_CUBE (a
    # coord_system a bare DimCoord built straight from the location
    # dicts' own lat/lon values would never have) - not just matching
    # points.
    assert temp_cube.coord("latitude").coord_system == iris.coord_systems.GeogCS(6371229.0)
    assert temp_cube.coord("longitude").coord_system == iris.coord_systems.GeogCS(6371229.0)

    assert len(seq_to_index) == 2
    assert seq_to_index[(eqn_common,)] == 1  # 7 of 8 locations -> most common
    assert seq_to_index[(eqn_other,)] == 2
    # The one special location's path_index/max_rate_inv must differ from
    # every other (identical) location.
    unique_indices = set(path_index_cube.data.ravel().tolist())
    assert len(unique_indices) == 2
    assert max_rate_inv_cube.data.max() == 99.0
    assert (max_rate_inv_cube.data == 5.0).sum() == 7  # 8 locations, 1 special
    # Each location here has exactly one segment, so net_rate_inv (the
    # summed seg_rate_inv) and max_rate_inv (the max) must agree exactly.
    assert (net_rate_inv_cube.data == max_rate_inv_cube.data).all()

    assert identity_map[seq_to_index[(eqn_common,)]] == {
        "eqns": [eqn_common],
        "directions": ["forward"],
        "nodes": ["N0", "N1"],
    }
    assert identity_map[seq_to_index[(eqn_other,)]] == {
        "eqns": [eqn_other],
        "directions": ["forward"],
        "nodes": ["N0", "N1"],
    }


def test_merge_paths_to_cubes_handles_empty_segments():
    # A location with segments=[] (no path found there) must not crash
    # max_rate_inv's max()-over-segments - it should get NaN instead.
    # net_rate_inv is already NaN by the time it gets here (_location()
    # mirrors calc_paths_serial()'s own no-path convention), so this
    # also checks that NaN survives the DataFrame/cube round-trip.
    import math

    eqn = "X+Y↔Z+W"
    hgt_idx_of = {1e5: 0, 1e2: 1}
    hgt_of = {1e5: 1000.0, 1e2: 2000.0}
    locations = [
        _location(
            pres,
            lat,
            lon,
            1000.0,
            [_segment(eqn, 1.0)],
            hgt_idx=hgt_idx_of[pres],
            hgt=hgt_of[pres],
        )
        for pres in [1e5, 1e2]
        for lat in [-10.0, 10.0]
        for lon in [0.0, 180.0]
    ]
    # no path found at this one location - matches calc_paths_serial()'s
    # own NetworkXNoPath convention (segments/nodes/max_rate_inv -> []/
    # []/NaN, alongside net_rate_inv already NaN).
    locations[0]["segments"] = []
    locations[0]["nodes"] = []
    locations[0]["net_rate_inv"] = float("nan")
    locations[0]["max_rate_inv"] = float("nan")

    _, _, path_index_cube, net_rate_inv_cube, max_rate_inv_cube, seq_to_index, identity_map = (
        merge_paths_to_cubes(locations, _REFERENCE_CUBE)
    )

    assert () in seq_to_index  # the no-path location gets its own pathway "sequence"
    assert identity_map[seq_to_index[()]] == {"eqns": [], "directions": [], "nodes": []}
    assert math.isnan(net_rate_inv_cube.data.ravel()[0])
    assert math.isnan(max_rate_inv_cube.data.ravel()[0])
    assert not any(math.isnan(v) for v in net_rate_inv_cube.data.ravel()[1:])
    assert not any(math.isnan(v) for v in max_rate_inv_cube.data.ravel()[1:])


def test_merge_paths_to_cubes_raises_on_incomplete_grid():
    eqn = "X+Y↔Z+W"
    hgt_idx_of = {1e5: 0, 1e2: 1}
    hgt_of = {1e5: 1000.0, 1e2: 2000.0}
    locations = [
        _location(
            pres,
            lat,
            lon,
            1000.0,
            [_segment(eqn, 1.0)],
            hgt_idx=hgt_idx_of[pres],
            hgt=hgt_of[pres],
        )
        for pres in [1e5, 1e2]
        for lat in [-10.0, 10.0]
        for lon in [0.0, 180.0]
    ][:-1]  # drop one location - now an incomplete 2x2x2 grid

    with pytest.raises(ValueError, match="incomplete run"):
        merge_paths_to_cubes(locations, _REFERENCE_CUBE)
