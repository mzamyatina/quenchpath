"""Stage 05 (parallel): per-location CH4-to-CO shortest-path search,
multiprocessing across vertical levels.

What: this module is a deliberately separate implementation of the
same stage (not a drop-in monkeypatch of it) - this one carries the
multiprocessing bookkeeping. Both this module and
quenchpath.stages.stage_05_paths_serial.calc_paths_serial() (which runs the
exact same search in a single serial loop) produce byte-identical
output for the same input (see tests/test_stage_paths_parallel.py) and
share their I/O/grid-selection helpers
(quenchpath.stages.stage_05_paths_serial.load_eqns_with_rates()/select_grid())
rather than duplicating those too.

Why separate rather than a monkeypatch: so the plain, easy-to-read
serial code stays untouched and easy to reason about.

What: parallelization granularity is one vertical level per task, not
one grid location.

Why: at FULL resolution that's ~66 tasks instead of ~855,360, so
process-pool scheduling/pickling overhead is negligible, while each
task is still cheap to hand off (only that level's 2D lat/lon slices of
temp/pres/k_eq/rate_f/rate_r travel to the worker - kilobytes to low
megabytes, not the multi-GB full 3D arrays). This granularity also
preserves calc_paths_serial()'s own per-level checkpointing
(quenchpath.io.save_paths_level() - one file per level, as each
level's worker task completes).

Every location within one level is still processed serially inside
that level's worker (same inner loop as the serial version) - levels
are the unit of parallelism, not individual locations.
"""

import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import networkx as nx

from quenchpath import io
from quenchpath.config import GridResolution, SpeciesConversion
from quenchpath.graph.shortest_path import (
    EQNS_W_FOUR_C,
    build_bidirect_graph,
    build_unidirect_graph,
    find_pathway,
    prune_to_carbon_subgraph,
    species_order,
)
from quenchpath.stages.stage_05_paths_serial import (
    assert_shared_grid,
    load_eqns_with_rates,
    select_grid,
)

# What: set once per worker process by _init_worker(), not resent with
# every task.
# Why: the network topology (nodes/nodes_with_carbon), each reaction's
# static fields (eqn/reacs/prods), and the species conversion are the
# same at every level - only the numeric
# k_eq/rate_f/rate_r/temp/pres slices passed to _process_one_level()
# vary.
_worker_state = {}


def _init_worker(nodes, nodes_with_carbon, eqn_meta, conversion=SpeciesConversion.CH4_CO):
    """ProcessPoolExecutor initializer - runs once per worker process."""
    _worker_state["nodes"] = nodes
    _worker_state["nodes_with_carbon"] = nodes_with_carbon
    _worker_state["eqn_meta"] = eqn_meta
    _worker_state["conversion"] = SpeciesConversion(conversion)


def _process_one_level(
    cfg, li, hgt, lats, lat_idx, lons, lon_idx, temp_slice, pres_slice, rate_slices
):
    """Worker task: find every (lat, lon) location's CH4-to-CO path at
    one vertical level (index `li`, height `hgt`) - the parallel
    counterpart of calc_paths_serial()'s inner two loops for a single
    level, using this level's already-sliced 2D (lat, lon) arrays
    instead of indexing into the full 3D ones.

    Parameters
    ----------
    cfg: quenchpath.config.RunConfig
        Only used to name the run in the NetworkXNoPath warning below -
        passed whole since it's a small, trivially picklable frozen
        dataclass.
    li: int
        This level's index into the run's full native grid.
    hgt: float
        This level's true level_height value.
    lats, lat_idx, lons, lon_idx: as returned by select_grid() - the
        chosen grid_resolution's latitude/longitude points and their
        indices into `temp_slice`/`pres_slice`/`rate_slices`.
    temp_slice, pres_slice: 2D (lat, lon) numpy arrays
        This level's air_temperature/air_pressure slice.
    rate_slices: list[dict]
        One dict per reaction (same order as `_worker_state["eqn_meta"]`
        - zipped against it below), this level's own 2D "k_eq"/
        "rate_f"/"rate_r" array slices. "eqn"/"reacs"/"prods" aren't
        repeated here - they're identical at every level, so they only
        ever travel once, via `_init_worker()`'s `eqn_meta`.

    Returns
    -------
    (li, hgt_dicts): (int, list[dict])
        `li` again (so the caller can label/order results without
        relying on task-submission order, since as_completed() doesn't
        preserve it) and this level's list of location dicts, in the
        same shape calc_paths_serial() yields.
    """
    nodes = _worker_state["nodes"]
    nodes_with_carbon = _worker_state["nodes_with_carbon"]
    eqn_meta = _worker_state["eqn_meta"]
    conversion = _worker_state["conversion"]
    # Which graph-construction function this search uses - see
    # calc_paths_serial()'s own docstring (quenchpath.stages.
    # stage_05_paths_serial) for why CH4_CO/CO_CH4 need different ones.
    build_graph = (
        build_bidirect_graph if conversion == SpeciesConversion.CO_CH4 else build_unidirect_graph
    )

    hgt_dicts = []
    for lat, lai in zip(lats, lat_idx):
        for lon, loi in zip(lons, lon_idx):
            temp = float(temp_slice[lai, loi])
            eqns_at_location = [
                {
                    "eqn": meta["eqn"],
                    "reacs": meta["reacs"],
                    "prods": meta["prods"],
                    "k_eq": r["k_eq"][lai, loi],
                    "rate_f": r["rate_f"][lai, loi],
                    "rate_r": r["rate_r"][lai, loi],
                }
                for meta, r in zip(eqn_meta, rate_slices)
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
                # What: save with no segments - rare (see
                # graph.shortest_path's EQNS_W_FOUR_C docstring) but
                # real, matches calc_paths_serial()'s own handling.
                # Why: don't abort the whole level over one
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

            hgt_dicts.append(
                {
                    "lat": float(lat),
                    "lon": float(lon),
                    "pres": float(pres_slice[lai, loi]),
                    "temp": temp,
                    "net_rate_inv": net_rate_inv,
                    "max_rate_inv": max_rate_inv,
                    "nodes": path_nodes,
                    "segments": segments,
                }
            )
    return li, hgt_dicts


def calc_paths_parallel(
    cfg,
    grid_resolution=GridResolution.SAMPLE,
    max_workers=None,
    conversion=SpeciesConversion.CH4_CO,
):
    """Parallel (multiprocessing, one task per vertical level) counterpart
    of quenchpath.stages.stage_05_paths_serial.calc_paths_serial() - identical inputs,
    saved files, and yielded dicts (see that function's own docstring),
    computed across an OS process pool instead of one serial loop. See
    this module's own docstring for why "one level per task" is the
    parallelization granularity.

    Parameters
    ----------
    cfg: quenchpath.config.RunConfig
    grid_resolution: quenchpath.config.GridResolution, optional
    max_workers: int, optional
        Passed straight to concurrent.futures.ProcessPoolExecutor -
        defaults to its own default (os.process_cpu_count(), i.e. every
        available core). Pass a smaller number to leave headroom for
        other work on the same machine.

        Why worker count, not RAM, is the limiting factor: each task
        only carries one level's 2D slices (~54 MB for the full
        181-reaction network - see this module's own docstring), so
        worker count is CPU-, not memory-, bound.

        Example: on a 16-core Apple M4 Max (12 performance + 4
        efficiency cores, 48 GB RAM - the machine this pipeline was
        developed on), 10 is a reasonable choice - the 4 efficiency
        cores contribute much less to this CPU-bound work than the 12
        performance ones, and leaving 2 performance cores free keeps
        the machine usable for other work while a run is in progress.
    conversion: quenchpath.config.SpeciesConversion, optional
        See calc_paths_serial()'s own docstring - sent to every worker
        once via _init_worker()'s initargs, not resent per level/task.

    Yields
    ------
    dict
        One location dict per location, same shape as
        calc_paths_serial() - not in any guaranteed lat/lon/level order
        (levels complete as their worker finishes, not submission order).
    """
    grid_resolution = GridResolution(grid_resolution)
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

    nodes = species_order(eqns)
    nodes_with_carbon = [i for i in nodes if "C" in i]
    eqn_meta = [{"eqn": eqn["eqn"], "reacs": eqn["reacs"], "prods": eqn["prods"]} for eqn in eqns]

    # as_completed() below only gets `li` back per task, not `hgt` - kept
    # here so save_paths_level()'s header line can still record it.
    hgt_of = dict(zip(hgt_idx, hgts))

    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_worker,
        initargs=(nodes, nodes_with_carbon, eqn_meta, conversion),
    ) as executor:
        futures = []
        for hgt, li in zip(hgts, hgt_idx):
            # Only this level's 2D slices travel to the worker - see
            # module docstring for why that's cheap even at FULL
            # resolution, unlike sending the full 3D arrays per task.
            # "eqn"/"reacs"/"prods" aren't repeated here - already sent
            # once via eqn_meta above (same order as `eqns`, so zippable
            # against rate_slices in _process_one_level()).
            rate_slices = [
                {
                    "k_eq": eqn["k_eq_arr"][li],
                    "rate_f": eqn["rate_f_arr"][li],
                    "rate_r": eqn["rate_r_arr"][li],
                }
                for eqn in eqns
            ]
            futures.append(
                executor.submit(
                    _process_one_level,
                    cfg,
                    li,
                    hgt,
                    lats,
                    lat_idx,
                    lons,
                    lon_idx,
                    temp_arr[li],
                    pres_arr[li],
                    rate_slices,
                )
            )

        for future in as_completed(futures):
            li, hgt_dicts = future.result()
            io.save_paths_level(
                hgt_dicts, cfg, grid_resolution, li, hgt_of[li], conversion=conversion
            )
            yield from hgt_dicts
