"""Tests for quenchpath.analysis.path_index.

Small synthetic location dicts - matching this project's established
convention of testing pure logic directly rather than mocking
quenchpath.io's real directories (see e.g. tests/test_stage_rates.py's
module docstring).
"""

import numpy as np
import pytest

from quenchpath.analysis.path_index import (
    assign_path_indices_from_locations,
    assign_path_indices_from_maps,
    eqn_sequence_of,
    path_index_map_from_locations,
    path_index_remap_lookup,
)


def _segment(eqn, seg_rate_inv, direction="forward"):
    return {"eqn": eqn, "seg_rate_inv": seg_rate_inv, "direction": direction}


def _location(segments, nodes=None):
    return {"segments": segments, "nodes": [] if nodes is None else nodes}


def test_eqn_sequence_of_returns_ordered_eqn_tuple():
    loc = _location(
        [
            _segment("CH4+H↔CH3+H2", 10.0),
            _segment("CH3+O↔CO+H2", 20.0),
        ]
    )
    assert eqn_sequence_of(loc) == ("CH4+H↔CH3+H2", "CH3+O↔CO+H2")


def test_eqn_sequence_of_empty_segments_is_empty_tuple():
    assert eqn_sequence_of(_location([])) == ()


def test_assign_path_indices_ranks_by_decreasing_frequency():
    common_eqn = ["X+Y↔Z+W"]
    rare_eqn = ["A+B↔C+D"]
    locs = [
        _location([_segment(common_eqn[0], 1.0)]),
        _location([_segment(common_eqn[0], 1.0)]),
        _location([_segment(rare_eqn[0], 1.0)]),
    ]

    seq_to_index = assign_path_indices_from_locations(locs)

    assert seq_to_index[tuple(common_eqn)] == 1  # more frequent -> index 1
    assert seq_to_index[tuple(rare_eqn)] == 2
    assert locs[0]["path_index"] == 1
    assert locs[1]["path_index"] == 1
    assert locs[2]["path_index"] == 2


def test_assign_path_indices_breaks_ties_deterministically_by_sequence():
    # Two sequences tied at 1 location each - the tie-break (sequence's
    # own tuple ordering) must not depend on `locations`' iteration
    # order, so swapping the two locations' order must give the same
    # index assignment both times.
    seq_a = ("A+B↔C+D",)
    seq_b = ("X+Y↔Z+W",)
    loc_a = _location([_segment(seq_a[0], 1.0)])
    loc_b = _location([_segment(seq_b[0], 1.0)])

    forward = assign_path_indices_from_locations([loc_a, loc_b])
    backward = assign_path_indices_from_locations([loc_b, loc_a])

    assert forward == backward


def test_assign_path_indices_from_maps_pools_and_ranks_across_datasets():
    # Two datasets, each already merged with their own local numbering:
    # dataset "a" discovered ("common",) first (local index 1), "rare_a"
    # second (local index 2); dataset "b" discovered ("common",) and
    # "rare_b" too, with different local indices. Pooled across both,
    # ("common",) has 3+2=5 locations (most frequent -> pooled index 1);
    # "rare_a"/"rare_b" have 1 location each, tied -> broken by their own
    # tuple ordering.
    path_index_maps = {
        "a": {1: {"eqns": ["common"]}, 2: {"eqns": ["rare_a"]}},
        "b": {1: {"eqns": ["rare_b"]}, 2: {"eqns": ["common"]}},
    }
    cube_data_by_dataset = {
        "a": np.array([1, 1, 1, 2]),
        "b": np.array([2, 2, 1]),
    }

    seq_to_index, seq_counts, seq_found_in, remap_by_dataset = assign_path_indices_from_maps(
        cube_data_by_dataset, path_index_maps
    )

    assert seq_to_index[("common",)] == 1  # 5 locations total, most common
    assert seq_counts[("common",)] == 5
    assert seq_counts[("rare_a",)] == 1
    assert seq_counts[("rare_b",)] == 1
    assert seq_found_in[("common",)] == {"a", "b"}
    # "rare_a"/"rare_b" tied at 1 location each - tuple order breaks the tie
    assert {("rare_a",), ("rare_b",)} == {seq for seq, idx in seq_to_index.items() if idx != 1}
    assert sorted(
        [seq for seq in seq_to_index if seq != ("common",)],
        key=lambda s: seq_to_index[s],
    ) == sorted([("rare_a",), ("rare_b",)])

    # Remapping each dataset's own cube data through its lookup array
    # must recover the pooled index at every location.
    remapped_a = remap_by_dataset["a"][cube_data_by_dataset["a"]]
    remapped_b = remap_by_dataset["b"][cube_data_by_dataset["b"]]
    assert (remapped_a == 1).sum() == 3  # dataset a's 3 "common" locations
    assert (remapped_b == 1).sum() == 2  # dataset b's 2 "common" locations
    assert remapped_a[remapped_a != 1][0] == seq_to_index[("rare_a",)]
    assert remapped_b[remapped_b != 1][0] == seq_to_index[("rare_b",)]


def test_path_index_remap_lookup_uses_an_externally_supplied_ranking():
    # Unlike assign_path_indices_from_maps() (which derives seq_to_index
    # itself, pooled fresh from the datasets given), this is the
    # "remap onto someone else's already-decided ranking" case - e.g. a
    # ranking pooled from raw, pre-merge locations that never went
    # through merge-paths at all. seq_to_index here deliberately
    # includes a sequence ("only_elsewhere",) that this dataset's own
    # path_index_map never uses, to confirm the lookup only cares about
    # sequences it's actually asked to remap.
    path_index_map = {1: {"eqns": ["common"]}, 2: {"eqns": ["rare"]}}
    seq_to_index = {("common",): 2, ("rare",): 5, ("only_elsewhere",): 1}
    cube_data = np.array([1, 1, 2, 1])

    lookup = path_index_remap_lookup(path_index_map, seq_to_index)
    remapped = lookup[cube_data]

    assert list(remapped) == [2, 2, 5, 2]


def test_path_index_remap_lookup_raises_on_a_sequence_missing_from_seq_to_index():
    path_index_map = {1: {"eqns": ["common"]}, 2: {"eqns": ["unranked"]}}
    seq_to_index = {("common",): 1}  # "unranked" deliberately missing

    with pytest.raises(KeyError):
        path_index_remap_lookup(path_index_map, seq_to_index)


def test_assign_path_indices_from_maps_remap_matches_path_index_remap_lookup():
    # assign_path_indices_from_maps() now builds its own remap_by_dataset
    # using path_index_remap_lookup() internally - confirm the two stay
    # in agreement (not just each internally self-consistent) by
    # recomputing dataset "a"'s lookup independently and comparing.
    path_index_maps = {
        "a": {1: {"eqns": ["common"]}, 2: {"eqns": ["rare_a"]}},
        "b": {1: {"eqns": ["rare_b"]}, 2: {"eqns": ["common"]}},
    }
    cube_data_by_dataset = {
        "a": np.array([1, 1, 1, 2]),
        "b": np.array([2, 2, 1]),
    }

    seq_to_index, _counts, _found_in, remap_by_dataset = assign_path_indices_from_maps(
        cube_data_by_dataset, path_index_maps
    )
    independent_lookup = path_index_remap_lookup(path_index_maps["a"], seq_to_index)

    assert list(remap_by_dataset["a"]) == list(independent_lookup)


def test_path_index_map_from_locations_captures_eqns_directions_nodes_per_index():
    locs = [
        _location(
            [_segment("CH4+H↔CH3+H2", 1.0, "forward"), _segment("CH3+O↔CO+H2", 2.0, "reverse")],
            nodes=["CH4", "CH3", "CO"],
        ),
        _location(
            [_segment("CH4+H↔CH3+H2", 5.0, "forward"), _segment("CH3+O↔CO+H2", 6.0, "reverse")],
            nodes=["CH4", "CH3", "CO"],
        ),
        _location([_segment("A+B↔C+D", 1.0, "forward")], nodes=["CH4", "CO"]),
    ]
    assign_path_indices_from_locations(locs)

    identity_map = path_index_map_from_locations(locs)

    common_idx = locs[0]["path_index"]
    rare_idx = locs[2]["path_index"]
    assert identity_map[common_idx] == {
        "eqns": ["CH4+H↔CH3+H2", "CH3+O↔CO+H2"],
        "directions": ["forward", "reverse"],
        "nodes": ["CH4", "CH3", "CO"],
    }
    assert identity_map[rare_idx] == {
        "eqns": ["A+B↔C+D"],
        "directions": ["forward"],
        "nodes": ["CH4", "CO"],
    }


def test_path_index_map_from_locations_raises_when_nodes_missing():
    loc = {"segments": [_segment("A+B↔C+D", 1.0)], "path_index": 1}  # no "nodes" key
    with pytest.raises(ValueError, match="no 'nodes' key"):
        path_index_map_from_locations([loc])


def test_path_index_map_from_locations_raises_on_direction_disagreement_within_one_index():
    # Same eqn_sequence_of() sequence (both just "A+B↔C+D"), but the two
    # locations disagree on direction - path_index_map_from_locations() must
    # catch this rather than silently keeping whichever came first.
    loc_a = _location([_segment("A+B↔C+D", 1.0, "forward")], nodes=["CH4", "CO"])
    loc_b = _location([_segment("A+B↔C+D", 1.0, "reverse")], nodes=["CH4", "CO"])
    loc_a["path_index"] = loc_b["path_index"] = 1  # force them into the same index

    with pytest.raises(ValueError, match="disagree on nodes/direction"):
        path_index_map_from_locations([loc_a, loc_b])
