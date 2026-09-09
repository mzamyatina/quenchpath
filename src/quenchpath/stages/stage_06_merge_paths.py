"""What: merge a completed calc_paths_serial()/calc_paths_parallel()
run's per-level pathway JSON Lines (one file per level) into 5 gridded
cubes - temp, pres, path_index, net_rate_inv, max_rate_inv - saved as
one .nc file: a single frozen, shareable artifact with the path-index
assignment pinned, for reproducible cross-run comparisons and figure
provenance.

Why not just a load-speed optimisation: reloading the raw per-level
JSON Lines directly is comparably fast - the point is the frozen,
pinned artifact, not speed.

What: a location's "pathway" is the ordered tuple of reaction eqn
strings along its path (CH4-to-CO by default, or CO-to-CH4 - see
`conversion` below); path_index 1 is whichever sequence is most common
among the locations being merged, 2 the second most common, etc. -
fresh per call, not anchored to any previous run's numbering (see
quenchpath.analysis.path_index for the eqn_sequence_of()/
assign_path_indices_from_locations() functions that do this, and for
pooling multiple datasets into consistent numbering - a comparison
across two datasets, not just one). The {index: {"eqns", "directions",
"nodes"}} mapping (see
quenchpath.analysis.path_index.path_index_map_from_locations()) is
embedded as a JSON-string attribute on the saved path_index cube.

Why embed that mapping: so the full pathway identity, not just
path_index itself, survives the merge.
"""

import iris.cube
import numpy as np
import pandas as pd

from quenchpath import io
from quenchpath.analysis.path_index import (
    assign_path_indices_from_locations,
    eqn_sequence_of,
    path_index_map_from_locations,
)
from quenchpath.config import SpeciesConversion

__all__ = [
    "eqn_sequence_of",
    "assign_path_indices_from_locations",
    "merge_paths_to_cubes",
    "merge_and_save_paths",
]


def merge_paths_to_cubes(locations, reference_cube):
    """What: merge already-loaded CH4-to-CO path location dicts into 5
    gridded iris Cubes on (level_height, latitude, longitude): temp
    [K], pres [Pa - that location's own true modelled pressure, not a
    shared axis, see issue #11], path_index [dimensionless - see
    assign_path_indices_from_locations()], net_rate_inv [cm3 s - the
    summed 1/rate across that location's whole path, i.e.
    find_pathway()'s own dijkstra-reported total - what path-finding
    itself minimizes], max_rate_inv [cm3 s - the slowest (bottleneck)
    segment's 1/rate along that same path, also find_pathway()'s own].
    Pure merging/gridding only.

    Why no recompute: net_rate_inv/max_rate_inv are both already
    computed by find_pathway() (stage 05).

    Parameters
    ----------
    locations: list[dict]
        One location dict per grid location (must together form a
        complete regular (hgt_idx, lat, lon) grid - e.g. as loaded from
        every JSON file a single calc_paths_serial()/calc_paths_parallel()
        run saved, via quenchpath.io.load_paths_from_disk()). Each must
        already have "hgt_idx"/"hgt" (folded in from its file's header
        line by load_paths_from_disk() - not present on either
        function's own yielded dicts), "nodes" (see path_index_map_from_locations()), and
        "pres"/"net_rate_inv"/"max_rate_inv" keys.
    reference_cube: iris.cube.Cube
        Any cube on the run's own native (level_height, latitude,
        longitude) grid (e.g. `io.load_converted(cfg).extract_cube(
        "CH4 number density")`) - only its 3 coordinates are used, never
        its data. The 5 returned cubes inherit real coordinates from
        `reference_cube` instead of getting bare reconstructed ones -
        after checking (not assuming) `locations`' own hgt/lat/lon
        values are exactly `reference_cube`'s own points, or a subset of
        them (e.g. a GridResolution.SAMPLE run's coarser grid).

        Why inherit from reference_cube: a location dict only carries
        bare hgt/lat/lon values - real coordinate metadata (var_name,
        coord_system, attributes, bounds) doesn't survive the
        per-location JSON round-trip.

    Returns
    -------
    (temp_cube, pres_cube, path_index_cube, net_rate_inv_cube,
    max_rate_inv_cube, seq_to_index, identity_map):
        The 5 cubes (var_name "temp"/"pres"/"path_index"/"net_rate_inv"/
        "max_rate_inv"), assign_path_indices_from_locations()'s {sequence: index}
        mapping, and path_index_map_from_locations()'s {index: {"eqns",
        "directions", "nodes"}} mapping.

    Raises
    ------
    ValueError
        If `locations` isn't a complete grid (e.g. an incomplete run),
        if `locations`' own hgt/lat/lon values aren't (a subset of)
        `reference_cube`'s own points, or (see path_index_map_from_locations())
        a location is missing "nodes" or two locations sharing a
        path_index disagree on nodes/direction.
    """
    seq_to_index = assign_path_indices_from_locations(locations)
    identity_map = path_index_map_from_locations(locations)

    # What: sorted by hgt_idx (robust int identity), ascending =
    # bottom-to-top.
    # Why hgt_idx is safe to sort/use for the vertical DimCoord (unlike
    # "pres"): hgt genuinely is one shared value per hgt_idx (see issue
    # #11).
    df = pd.DataFrame(locations).sort_values(
        ["hgt_idx", "lat", "lon"], ascending=[True, True, True]
    )

    hgts = df.groupby("hgt_idx").hgt.first().to_numpy()
    lats = df.lat.unique()
    lons = df.lon.unique()
    shape = (len(hgts), len(lats), len(lons))
    expected_n = shape[0] * shape[1] * shape[2]
    if len(df) != expected_n:
        raise ValueError(f"Have {len(df)} locations but expected {expected_n} - incomplete run?")

    # What: long_name="level_height" (not "hgt").
    # Why: matches this same coordinate's name everywhere else in the
    # pipeline - it's the raw UM/aeolus convention (aeolus.model.um.z),
    # not quenchpath's own choice.
    #
    # What: no manual reordering of the 3 extracted coords below -
    # checked explicitly, not assumed.
    # Why that's safe: reference_cube's own native grid is itself
    # stored ascending along all 3 axes (matching df's own ascending
    # sort above), and Constraint-based extraction preserves that
    # original order, so the 3 extracted coords come out already in
    # df's own hgt/lat/lon order.
    reference_subset = reference_cube.extract(
        iris.Constraint(
            level_height=lambda z: z.point in hgts,
            latitude=lambda la: la.point in lats,
            longitude=lambda lo: lo.point in lons,
        )
    )
    if reference_subset is None or reference_subset.shape != shape:
        got_shape = None if reference_subset is None else reference_subset.shape
        raise ValueError(
            f"locations' own hgt/lat/lon values aren't (a subset of) reference_cube's own "
            f"native grid points - matched shape {got_shape}, expected {shape}."
        )
    level_coord = reference_subset.coord("level_height")
    lat_coord = reference_subset.coord("latitude")
    lon_coord = reference_subset.coord("longitude")
    assert np.array_equal(level_coord.points, hgts), "level_height order doesn't match hgt_idx"
    assert np.array_equal(lat_coord.points, lats), "latitude order doesn't match df's own sort"
    assert np.array_equal(lon_coord.points, lons), "longitude order doesn't match df's own sort"

    dim_coords_and_dims = [(level_coord, 0), (lat_coord, 1), (lon_coord, 2)]
    temp_cube = iris.cube.Cube(
        df.temp.values.astype(float).reshape(shape),
        dim_coords_and_dims=dim_coords_and_dims,
        var_name="temp",
        units="K",
    )
    pres_cube = iris.cube.Cube(
        df.pres.values.astype(float).reshape(shape),
        dim_coords_and_dims=dim_coords_and_dims,
        var_name="pres",
        units="Pa",
    )
    path_index_cube = iris.cube.Cube(
        df.path_index.values.reshape(shape),
        dim_coords_and_dims=dim_coords_and_dims,
        var_name="path_index",
        units="1",  # a categorical index (1 = most dominant, 2 = next, ...), not physical
    )
    net_rate_inv_cube = iris.cube.Cube(
        df.net_rate_inv.values.astype(float).reshape(shape),
        dim_coords_and_dims=dim_coords_and_dims,
        var_name="net_rate_inv",
        units="cm3 s",
    )
    max_rate_inv_cube = iris.cube.Cube(
        df.max_rate_inv.values.astype(float).reshape(shape),
        dim_coords_and_dims=dim_coords_and_dims,
        var_name="max_rate_inv",
        units="cm3 s",
    )
    return (
        temp_cube,
        pres_cube,
        path_index_cube,
        net_rate_inv_cube,
        max_rate_inv_cube,
        seq_to_index,
        identity_map,
    )


def merge_and_save_paths(cfg, grid_resolution, conversion=SpeciesConversion.CH4_CO):
    """I/O driver: load every saved per-level pathway JSON Lines for
    `cfg`/`grid_resolution`/`conversion` from disk (quenchpath.io.
    load_paths_from_disk()), merge into gridded cubes, and save as
    one .nc file. Pure merging - no conversion-specific logic of its own,
    just threads `conversion` through to whichever saved files it loads/
    writes.

    Returns
    -------
    (temp_cube, pres_cube, path_index_cube, net_rate_inv_cube,
    max_rate_inv_cube, seq_to_index, identity_map):
    as merge_paths_to_cubes().
    """
    conversion = SpeciesConversion(conversion)
    locations = io.load_paths_from_disk(cfg, grid_resolution, conversion=conversion)
    # What: CH4 number density (stage 03's saved output) is the merged
    # cubes' coordinate reference here - see merge_paths_to_cubes()'s
    # own docstring, only its coordinates are used, never its data.
    # Why CH4 number density specifically: purely convenient to load,
    # not physically significant.
    reference_cube = io.load_converted(cfg).extract_cube("CH4 number density")
    (
        temp_cube,
        pres_cube,
        path_index_cube,
        net_rate_inv_cube,
        max_rate_inv_cube,
        seq_to_index,
        identity_map,
    ) = merge_paths_to_cubes(locations, reference_cube)
    io.save_merged_paths(
        temp_cube,
        pres_cube,
        path_index_cube,
        net_rate_inv_cube,
        max_rate_inv_cube,
        identity_map,
        cfg,
        grid_resolution,
        conversion=conversion,
    )
    return (
        temp_cube,
        pres_cube,
        path_index_cube,
        net_rate_inv_cube,
        max_rate_inv_cube,
        seq_to_index,
        identity_map,
    )
