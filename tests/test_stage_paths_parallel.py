"""Tests for quenchpath.stages.stage_05_paths_parallel.

_process_one_level()/_init_worker() are pure (plain dicts/arrays, no
iris/grid, no quenchpath.io) - synthetic-data-only, same convention as
tests/test_graph_shortest_path.py. calc_paths_parallel() itself
(the I/O driver) needs real network+rates data and isn't unit-tested
here, matching tests/test_stage_paths_serial.py's own module docstring
- see commit e4cb6ac's message for its real-data verification
(byte-for-byte match against calc_paths_serial()'s own output for the
same real run).
"""

import math
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pytest

from quenchpath.config import RunConfig
from quenchpath.stages.stage_05_paths_parallel import (
    _init_worker,
    _process_one_level,
    _worker_state,
)


@pytest.fixture(autouse=True)
def _clear_worker_state():
    # _process_one_level() reads module-level _worker_state, normally
    # populated once per process by ProcessPoolExecutor's initializer -
    # tests calling it directly (no pool) must set/clear it themselves
    # so state from one test can't leak into the next.
    yield
    _worker_state.clear()


def _rate_slices(eqns_2x2):
    """Build _process_one_level()'s `rate_slices` argument: one dict per
    reaction, with 2x2 (lat, lon) k_eq/rate_f/rate_r arrays - eqn/reacs/
    prods aren't repeated here, they come from `eqn_meta` instead (see
    _eqn_meta())."""
    return [
        {
            "k_eq": np.array(eqn["k_eq"]),
            "rate_f": np.array(eqn["rate_f"]),
            "rate_r": np.array(eqn["rate_r"]),
        }
        for eqn in eqns_2x2
    ]


def _eqn_meta(eqns_2x2):
    """Build _init_worker()'s `eqn_meta` argument: one dict per reaction
    (same order as _rate_slices(), zipped against it in
    _process_one_level()), just the eqn/reacs/prods fields that are
    identical at every level."""
    return [{"eqn": eqn["eqn"], "reacs": eqn["reacs"], "prods": eqn["prods"]} for eqn in eqns_2x2]


def test_process_one_level_finds_paths_at_every_lat_lon():
    # find_pathway() defaults to source="CH4"/target="CO" - a
    # straightforward two-reaction CH4->B->CO chain, same k_eq/rate_f/
    # rate_r at all 4 (lat, lon) points in this 2x2 slice.
    eqns_2x2 = [
        {
            "eqn": "CH4↔B",
            "reacs": ["CH4"],
            "prods": ["B"],
            "k_eq": [[2.0, 2.0], [2.0, 2.0]],
            "rate_f": [[10.0, 10.0], [10.0, 10.0]],
            "rate_r": [[5.0, 5.0], [5.0, 5.0]],
        },
        {
            "eqn": "B↔CO",
            "reacs": ["B"],
            "prods": ["CO"],
            "k_eq": [[2.0, 2.0], [2.0, 2.0]],
            "rate_f": [[4.0, 4.0], [4.0, 4.0]],
            "rate_r": [[1.0, 1.0], [1.0, 1.0]],
        },
    ]
    nodes = ["CH4", "B", "CO"]
    _init_worker(nodes=nodes, nodes_with_carbon=nodes, eqn_meta=_eqn_meta(eqns_2x2))

    li, hgt_dicts = _process_one_level(
        cfg=RunConfig(planet="testplanet", exp="kin"),
        li=3,
        hgt=1234.5,
        lats=[-10.0, 10.0],
        lat_idx=[0, 1],
        lons=[0.0, 90.0],
        lon_idx=[0, 1],
        temp_slice=np.array([[500.0, 501.0], [502.0, 503.0]]),
        pres_slice=np.array([[1e5, 1e5], [1e5, 1e5]]),
        rate_slices=_rate_slices(eqns_2x2),
    )

    assert li == 3
    assert len(hgt_dicts) == 4  # 2 lats x 2 lons
    for d in hgt_dicts:
        # hgt_idx/hgt/planet/exp aren't in the per-location dict - the
        # caller (calc_paths_parallel()) already has cfg/li/hgt in
        # scope, and save_paths_level() puts them in the file's
        # header line instead of repeating them on every location.
        assert "hgt_idx" not in d
        assert "hgt" not in d
        assert "planet" not in d
        assert "exp" not in d
        assert d["nodes"] == ["CH4", "B", "CO"]
        assert [seg["eqn"] for seg in d["segments"]] == ["CH4↔B", "B↔CO"]
        # "from"/"to" aren't on segments - already recoverable as
        # zip(d["nodes"], d["nodes"][1:]).
        assert "from" not in d["segments"][0]
        assert "to" not in d["segments"][0]
        assert d["net_rate_inv"] == pytest.approx(1 / 10.0 + 1 / 4.0)
        assert d["max_rate_inv"] == pytest.approx(1 / 4.0)  # slowest (bottleneck) segment
    # lat/lon/temp/pres actually vary per point, not just copy-pasted
    assert {d["lat"] for d in hgt_dicts} == {-10.0, 10.0}
    assert {d["lon"] for d in hgt_dicts} == {0.0, 90.0}
    assert {d["temp"] for d in hgt_dicts} == {500.0, 501.0, 502.0, 503.0}


def test_process_one_level_no_path_saves_empty_segments_and_warns():
    # A and B never connect - matches calc_paths_serial()'s own
    # NetworkXNoPath handling (see quenchpath.stages.stage_05_paths_serial).
    eqns_2x2 = [
        {
            "eqn": "A↔A2",
            "reacs": ["A"],
            "prods": ["A2"],
            "k_eq": [[2.0]],
            "rate_f": [[10.0]],
            "rate_r": [[5.0]],
        },
    ]
    nodes = ["CH4", "CO", "A", "A2"]
    _init_worker(nodes=nodes, nodes_with_carbon=nodes, eqn_meta=_eqn_meta(eqns_2x2))

    with pytest.warns(UserWarning, match="No CH4-to-CO path found"):
        li, hgt_dicts = _process_one_level(
            cfg=RunConfig(planet="testplanet", exp="kin"),
            li=0,
            hgt=1.0,
            lats=[0.0],
            lat_idx=[0],
            lons=[0.0],
            lon_idx=[0],
            temp_slice=np.array([[500.0]]),
            pres_slice=np.array([[1e5]]),
            rate_slices=_rate_slices(eqns_2x2),
        )

    assert len(hgt_dicts) == 1
    assert hgt_dicts[0]["nodes"] == []
    assert hgt_dicts[0]["segments"] == []
    assert math.isnan(hgt_dicts[0]["net_rate_inv"])
    assert math.isnan(hgt_dicts[0]["max_rate_inv"])


def test_calc_paths_parallel_dispatches_through_a_real_process_pool():
    # Not calc_paths_parallel() itself (needs real network/rates
    # data - see module docstring) - just proves _init_worker()/
    # _process_one_level() actually survive real pickling/dispatch
    # through ProcessPoolExecutor (the thing a purely in-process test of
    # _process_one_level() alone can't catch: unpicklable arguments,
    # wrong initializer signature, etc.).
    eqns_2x2 = [
        {
            "eqn": "CH4↔CO",
            "reacs": ["CH4"],
            "prods": ["CO"],
            "k_eq": [[2.0]],
            "rate_f": [[10.0]],
            "rate_r": [[5.0]],
        },
    ]
    cfg = RunConfig(planet="testplanet", exp="kin")
    with ProcessPoolExecutor(
        max_workers=2,
        initializer=_init_worker,
        initargs=(["CH4", "CO"], ["CH4", "CO"], _eqn_meta(eqns_2x2)),
    ) as executor:
        futures = [
            executor.submit(
                _process_one_level,
                cfg,
                li,
                float(li),
                [0.0],
                [0],
                [0.0],
                [0],
                np.array([[500.0]]),
                np.array([[1e5]]),
                _rate_slices(eqns_2x2),
            )
            for li in range(3)
        ]
        results = {f.result()[0]: f.result()[1] for f in futures}

    assert set(results) == {0, 1, 2}
    for li, hgt_dicts in results.items():
        assert "hgt_idx" not in hgt_dicts[0]
        assert hgt_dicts[0]["segments"][0]["eqn"] == "CH4↔CO"
