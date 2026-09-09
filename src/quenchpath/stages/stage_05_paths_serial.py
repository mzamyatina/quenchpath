"""Stage 05 (serial): per-location CH4-to-CO shortest-path search across
a planet/experiment's grid.

What: the per-location graph-construction/dijkstra logic itself lives
in quenchpath.graph.shortest_path (pure, iris/grid-free); this module
handles I/O, the grid-resolution toggle, and the
precomputed-array/integer-index pattern.

Why that pattern: lets the location loop index plain numpy arrays
instead of repeated iris.Constraint-based .extract() calls.

quenchpath.stages.stage_05_paths_parallel is a separate, multiprocessing
implementation of this same stage (reusing load_eqns_with_rates()/
select_grid() from here rather than duplicating them) - see its module
docstring. Recommended pairing: this module for SAMPLE resolution
(~4896 locations, already fast serially), paths_parallel for FULL
(where multiprocessing actually pays off) - not an enforced restriction,
either works at either resolution.
"""

import json
import warnings

import networkx as nx
import numpy as np

from quenchpath import io
from quenchpath.config import GridResolution, SpeciesConversion, sample_grid_points
from quenchpath.graph.shortest_path import (
    EQNS_W_FOUR_C,
    build_bidirect_graph,
    build_unidirect_graph,
    find_pathway,
    prune_to_carbon_subgraph,
    species_order,
)
from quenchpath.mypaths import path_to_networks_processed


def load_eqns_with_rates(cfg):
    """What: load the chemical network plus stage 04's saved
    k_eq/rate_f/rate_r cubes for `cfg`, merged into one eqns list (each
    entry gets "k_eq"/"rate_f"/"rate_r" Cube fields attached, matched by
    the cubes' "eqn" attribute). The CH4(+M)<->CH3+H(+M) entry is
    dropped.

    Why dropped: already folded into H+CH3(+M)<->CH4(+M) by stage 04's
    special case.
    """
    with open(path_to_networks_processed / "venot2019_network.json") as f:
        eqns = json.load(f)
    eqns = [eqn for eqn in eqns if eqn["eqn"] != "CH4(+M)↔CH3+H(+M)"]

    cl_all = io.load_rate_cubes(cfg)
    cl_ks_eq = [c for c in cl_all if c.name() == "equilibrium_constant"]
    cl_rates_f = [c for c in cl_all if c.name() == "forward_reaction_rate"]
    cl_rates_r = [c for c in cl_all if c.name() == "reverse_reaction_rate"]

    for eqn in eqns:
        for cb_k_eq in cl_ks_eq:
            if eqn["eqn"] == cb_k_eq.attributes["eqn"]:
                eqn["k_eq"] = cb_k_eq
        for cb_rate_f in cl_rates_f:
            if eqn["eqn"] == cb_rate_f.attributes["eqn"]:
                eqn["rate_f"] = cb_rate_f
        for cb_rate_r in cl_rates_r:
            if eqn["eqn"] == cb_rate_r.attributes["eqn"]:
                eqn["rate_r"] = cb_rate_r
    return eqns


def select_grid(temp_cube, grid_resolution):
    """What: select the hgt/lat/lon points to iterate for
    `grid_resolution`, and precompute each point's integer index into
    `temp_cube`'s full native grid.

    Why: enables indexing plain numpy arrays instead of repeated
    iris.Constraint-based .extract() calls in the location loop.

    What: the vertical axis is `temp_cube`'s "level_height" coordinate
    (stage 03's native grid); pressure is looked up per location
    instead (`pres_arr[hgt_idx, lat_idx, lon_idx]` in calc_paths_serial()).

    Why not pressure as the vertical axis: real per-column pressure
    isn't a shared axis (see issue #11).

    Returns
    -------
    hgts, lats, lons: numpy arrays of the selected height/latitude/
        longitude coordinate values.
    hgt_idx, lat_idx, lon_idx: lists of each selected point's integer
        index into `temp_cube`'s full (native) grid, in the same order as
        hgts/lats/lons.
    """

    def grid_index(full_points, value):
        """Index of `value` within `full_points` (exact match required)."""
        matches = np.where(full_points == value)[0]
        assert len(matches) == 1, f"expected exactly one match for {value!r}, found {len(matches)}"
        return int(matches[0])

    full_hgts = temp_cube.coord("level_height").points
    full_lats = temp_cube.coord("latitude").points
    full_lons = temp_cube.coord("longitude").points

    grid_resolution = GridResolution(grid_resolution)
    if grid_resolution == GridResolution.FULL:
        hgts, lats, lons = full_hgts, full_lats, full_lons
    else:
        sample = sample_grid_points(full_hgts, full_lats, full_lons)
        hgts, lats, lons = sample["hgts"], sample["lats"], sample["lons"]

    hgt_idx = [grid_index(full_hgts, hgt) for hgt in hgts]
    lat_idx = [grid_index(full_lats, lat) for lat in lats]
    lon_idx = [grid_index(full_lons, lon) for lon in lons]
    return hgts, lats, lons, hgt_idx, lat_idx, lon_idx


def assert_shared_grid(eqns, temp_cube, pres_cube):
    """What: assert every reaction's k_eq cube and pres_cube share
    temp_cube's exact vertical/latitude/longitude grid - checked once,
    up front, by both calc_paths_serial() and
    stage_05_paths_parallel.calc_paths_parallel().
    Why shared here: reusing this rather than duplicating the checks,
    same as load_eqns_with_rates()/select_grid() above."""
    full_hgts = temp_cube.coord("level_height").points
    full_lats = temp_cube.coord("latitude").points
    full_lons = temp_cube.coord("longitude").points
    assert all((eqn["k_eq"].coord("level_height").points == full_hgts).all() for eqn in eqns)
    assert all((eqn["k_eq"].coord("latitude").points == full_lats).all() for eqn in eqns)
    assert all((eqn["k_eq"].coord("longitude").points == full_lons).all() for eqn in eqns)
    assert (pres_cube.coord("level_height").points == full_hgts).all()
    assert (pres_cube.coord("latitude").points == full_lats).all()
    assert (pres_cube.coord("longitude").points == full_lons).all()


def calc_paths_serial(
    cfg, grid_resolution=GridResolution.SAMPLE, conversion=SpeciesConversion.CH4_CO
):
    """I/O driver: load `cfg`'s network+rates, iterate the chosen grid,
    find each location's pathway (CH4-to-CO by default; `conversion` -
    see quenchpath.config.SpeciesConversion - picks source/target species,
    passed straight to find_pathway() below).

    CH4_CO stays on quenchpath.graph.shortest_path.build_unidirect_graph();
    CO_CH4 uses build_bidirect_graph() instead - see this function's own
    `build_graph` dispatch below and build_bidirect_graph()'s own
    docstring for why the two conversions need different graphs.

    Yields one location dict per location ("pres" is that location's own
    true modelled pressure - this can differ substantially between
    locations at the same hgt_idx, see issue #11); each level's locations
    are also saved to disk together as one JSON Lines file
    (quenchpath.io.save_paths_level(), whose header line carries
    that level's hgt_idx/hgt) once that level completes.
    """
    conversion = SpeciesConversion(conversion)
    eqns = load_eqns_with_rates(cfg)

    converted_cubes = io.load_converted(cfg)
    temp_cube = converted_cubes.extract_cube("air_temperature")
    temp_cube.attributes = {}
    pres_cube = converted_cubes.extract_cube("air_pressure")
    assert_shared_grid(eqns, temp_cube, pres_cube)

    hgts, lats, lons, hgt_idx, lat_idx, lon_idx = select_grid(temp_cube, grid_resolution)

    temp_arr = temp_cube.data
    pres_arr = pres_cube.data
    for eqn in eqns:
        eqn["k_eq_arr"] = eqn["k_eq"].data
        eqn["rate_f_arr"] = eqn["rate_f"].data
        eqn["rate_r_arr"] = eqn["rate_r"].data

    # What: compute nodes once, outside the location loop.
    # Why: the node list is the same at every location (only
    # k_eq/rate_f/rate_r vary) - rebuilding it per location would mean
    # up to n_levels*90*144 redundant rebuilds.
    nodes = species_order(eqns)
    nodes_with_carbon = [i for i in nodes if "C" in i]
    # Which graph-construction function this search uses - see this
    # function's own docstring for why CH4_CO/CO_CH4 need different ones.
    build_graph = (
        build_bidirect_graph if conversion == SpeciesConversion.CO_CH4 else build_unidirect_graph
    )

    for hgt, li in zip(hgts, hgt_idx):
        hgt_dicts = []
        for lat, lai in zip(lats, lat_idx):
            for lon, loi in zip(lons, lon_idx):
                temp = float(temp_arr[li, lai, loi])
                eqns_at_location = [
                    {
                        "eqn": eqn["eqn"],
                        "reacs": eqn["reacs"],
                        "prods": eqn["prods"],
                        "k_eq": eqn["k_eq_arr"][li, lai, loi],
                        "rate_f": eqn["rate_f_arr"][li, lai, loi],
                        "rate_r": eqn["rate_r_arr"][li, lai, loi],
                    }
                    for eqn in eqns
                ]
                G = build_graph(eqns_at_location, nodes=nodes)
                SG_tidy = prune_to_carbon_subgraph(
                    G, EQNS_W_FOUR_C, nodes_with_carbon=nodes_with_carbon
                )
                try:
                    net_rate_inv, max_rate_inv, path_nodes, segments = find_pathway(
                        SG_tidy, source=conversion.source, target=conversion.target
                    )
                except nx.NetworkXNoPath:
                    # What: save it with no segments instead
                    # (merge_paths_to_cubes() handles that) - rare (see
                    # graph.shortest_path's EQNS_W_FOUR_C docstring) but
                    # real.
                    # Why: don't abort a multi-hour run over one
                    # disconnected location.
                    warnings.warn(
                        f"No {conversion.label} path found for {cfg} at "
                        f"hgt_idx={li} (hgt={hgt:.1f} m), lat={lat}, "
                        f"lon={lon} - saving with no segments.",
                        stacklevel=2,
                    )
                    net_rate_inv, max_rate_inv, path_nodes, segments = (
                        float("nan"),
                        float("nan"),
                        [],
                        [],
                    )

                location_dict = {
                    "lat": float(lat),
                    "lon": float(lon),
                    "pres": float(pres_arr[li, lai, loi]),
                    "temp": temp,
                    "net_rate_inv": net_rate_inv,
                    "max_rate_inv": max_rate_inv,
                    "nodes": path_nodes,
                    "segments": segments,
                }
                yield location_dict
                hgt_dicts.append(location_dict)
        io.save_paths_level(hgt_dicts, cfg, grid_resolution, li, hgt, conversion=conversion)
