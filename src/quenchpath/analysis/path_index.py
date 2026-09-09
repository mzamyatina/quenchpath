"""What: pathway identity and path_index assignment/ranking.
stage_06_merge_paths.py uses it for one run's own per-location grid.

Why kept separate (reusable outside the merge-paths pipeline stage):
the same ranking problem shows up again when multiple already-merged
datasets need consistent numbering (e.g. a separate repo building
cross-planet figures) - the same reasoning
quenchpath.graph.shortest_path is kept separate from its calling
stage.
"""

from collections import Counter

import numpy as np


def eqn_sequence_of(location):
    """The ordered tuple of reaction eqn strings along one location's
    pathway - the "identity" used to group locations by shared pathway
    (see assign_path_indices_from_locations()). `location` is one
    location dict (calc_paths_serial()/calc_paths_parallel()'s
    yielded shape, or either's saved JSON loaded back).
    """
    return tuple(segment["eqn"] for segment in location.get("segments", []))


def assign_path_indices_from_locations(locations):
    """What: assign every dict in `locations` a "path_index" int
    (mutates each dict in place): 1 for the most common
    eqn_sequence_of() sequence among `locations`, 2 for the second most
    common, etc. Ties are broken by the sequence's own tuple ordering,
    not by dict/iteration order.

    Why tuple ordering, not iteration order: dict/iteration order can
    vary across filesystems/runs, so tuple ordering keeps this
    deterministic given the same locations regardless of the order
    they're passed in.

    What: a comparison across two datasets that needs consistent
    numbering should pool both into one
    assign_path_indices_from_locations() call (concatenate their
    location lists first), then split the result back apart - each
    location still knows which dataset it came from via whatever other
    keys it carries. If the datasets are already merged (only each
    one's own path_index_map survives, not the raw locations - see
    stage_06_merge_paths.py's module docstring), use
    assign_path_indices_from_maps() instead.

    Returns the {sequence: index} mapping used - its length is the total
    number of distinct pathways found.
    """
    seq_counts = Counter(eqn_sequence_of(loc) for loc in locations)
    ranked_seqs = sorted(seq_counts, key=lambda s: (-seq_counts[s], s))
    seq_to_index = {seq: i + 1 for i, seq in enumerate(ranked_seqs)}

    for loc in locations:
        loc["path_index"] = seq_to_index[eqn_sequence_of(loc)]
    return seq_to_index


def path_index_map_from_locations(locations):
    """What: {path_index: {"eqns": [...], "directions": [...], "nodes":
    [...]}} for every distinct path_index among `locations` (which must
    already have "path_index" set - see
    assign_path_indices_from_locations()) - embedded in the saved .nc's
    "path_index_map" attribute (see
    quenchpath.stages.stage_06_merge_paths).

    Why the fuller identity, not just the eqn sequence:
    assign_path_indices_from_locations() itself only needs
    eqn_sequence_of()'s bare eqn sequence to rank pathways, but a
    merged/saved dataset should carry the fuller identity too, so a
    downstream consumer can tell which reactions/species a path_index
    represents without re-running the pathway search.

    What: every location sharing one path_index must agree on
    nodes/directions too, not just the eqn_sequence_of() sequence used
    to group them - checked here, not assumed (raises ValueError on
    disagreement).

    Why check this: a reaction's forward/reverse direction is itself
    temperature-dependent, so two locations sharing an eqn sequence
    could in principle favour opposite directions at some hop even
    though grouping is by eqn alone.

    Raises
    ------
    ValueError
        If a location has no "nodes" key (every location dict
        find_pathway()/calc_paths_serial()/calc_paths_parallel() yields
        has one - see their own docstrings; this is a precondition
        check for a caller passing hand-built locations, not something
        real pipeline output should ever trigger), or if two locations
        sharing a path_index disagree on nodes/directions.
    """
    identity_by_index = {}
    for loc in locations:
        if "nodes" not in loc:
            raise ValueError(
                "location has no 'nodes' key - real pipeline output always has one "
                "(see find_pathway()/calc_paths_serial()/calc_paths_parallel())"
            )
        idx = loc["path_index"]
        entry = {
            "eqns": [seg["eqn"] for seg in loc["segments"]],
            "directions": [seg["direction"] for seg in loc["segments"]],
            "nodes": list(loc["nodes"]),
        }
        existing = identity_by_index.setdefault(idx, entry)
        if existing != entry:
            raise ValueError(
                f"path_index {idx}: locations sharing the same eqn_sequence_of() "
                f"sequence disagree on nodes/direction - {existing} vs {entry}"
            )
    return identity_by_index


def path_index_remap_lookup(path_index_map, seq_to_index):
    """{local_index: pooled_index} lookup array for one dataset, built by
    matching each local index's own reaction sequence
    (`path_index_map[local_index]["eqns"]`) against `seq_to_index` - the
    single building block for going from a merge-paths dataset's own
    local path_index numbering to *any* externally-decided pooled
    numbering, whether that numbering came from
    assign_path_indices_from_maps() itself (pooled fresh across these
    same datasets - which uses this function internally, see below) or
    from somewhere else entirely (e.g. a ranking pooled from raw,
    pre-merge locations that never went through merge-paths at all).

    Parameters
    ----------
    path_index_map: dict[int, dict]
        One dataset's own {local_index: {"eqns": [...], ...}} mapping
        (see path_index_map_from_locations()) - only "eqns" is used here.
    seq_to_index: dict[tuple, int]
        {sequence: pooled_index} - the target numbering. Every sequence
        `path_index_map` actually uses must be a key of this (raises
        KeyError, not a silent miss, if one isn't - a mismatched
        `seq_to_index` almost certainly means it wasn't pooled from a
        superset of this dataset's own sequences).

    Returns
    -------
    numpy.ndarray
        Indexed by local_index -> pooled_index. Apply via
        `lookup[path_index_cube.data]` (or, for a genuine iris Cube,
        `path_index_cube.copy(data=lookup[path_index_cube.data])`).
    """
    lookup = np.zeros(max(path_index_map) + 1, dtype=int)
    for local_idx, entry in path_index_map.items():
        lookup[local_idx] = seq_to_index[tuple(entry["eqns"])]
    return lookup


def assign_path_indices_from_maps(cube_data_by_dataset, path_index_maps):
    """What: assign_path_indices_from_locations()'s pooling problem,
    worked from already-merged datasets instead of raw locations - each
    dataset only has its own path_index cube data and its own
    {local_index: sequence} path_index_map (as loaded back via
    quenchpath.io.load_merged_paths()), not the original locations.

    Why work from already-merged datasets: the original locations may
    be far slower to reload at FULL resolution.

    Parameters
    ----------
    cube_data_by_dataset: dict[str, numpy.ndarray]
        {dataset_key: path_index_cube.data} - one array per dataset,
        holding each grid location's own *local* path_index (an int
        keyed into that dataset's own `path_index_maps[dataset_key]`,
        not yet comparable across datasets).
    path_index_maps: dict[str, dict[int, dict]]
        {dataset_key: path_index_map} - same keys as
        `cube_data_by_dataset`; each dataset's own {local_index:
        {"eqns": [...], "directions": [...], "nodes": [...]}} mapping
        (see path_index_map_from_locations()) - only "eqns" is used here for the
        pooling identity itself.

    Returns
    -------
    (seq_to_index, seq_counts, seq_found_in, remap_by_dataset):
        seq_to_index: {sequence: pooled_index} - ranked by total
        location count summed across every dataset (tie-break: the
        sequence's own tuple ordering, same rule as
        assign_path_indices_from_locations()).
        seq_counts: {sequence: total location count}, summed across
        every dataset - same keys as seq_to_index, for reporting (e.g.
        each pathway's share of all locations).
        seq_found_in: {sequence: set of dataset_key} - which datasets
        that sequence actually occurs in, for reporting/labelling.
        remap_by_dataset: {dataset_key: numpy.ndarray} - one lookup
        array per dataset (indexed by that dataset's own local index),
        so `remap_by_dataset[key][cube_data_by_dataset[key]]` recolors
        that dataset's path_index cube data onto the shared pooled
        numbering in one vectorized step - see path_index_remap_lookup(),
        used here to build each one.
    """
    seq_counts = Counter()
    seq_found_in = {}
    for key, cube_data in cube_data_by_dataset.items():
        path_index_map = path_index_maps[key]
        local_indices, counts = np.unique(cube_data, return_counts=True)
        for local_idx, count in zip(local_indices, counts):
            seq = tuple(path_index_map[int(local_idx)]["eqns"])
            seq_counts[seq] += int(count)
            seq_found_in.setdefault(seq, set()).add(key)

    ranked_seqs = sorted(seq_counts, key=lambda s: (-seq_counts[s], s))
    seq_to_index = {seq: i + 1 for i, seq in enumerate(ranked_seqs)}

    remap_by_dataset = {
        key: path_index_remap_lookup(path_index_maps[key], seq_to_index)
        for key in cube_data_by_dataset
    }

    return seq_to_index, dict(seq_counts), seq_found_in, remap_by_dataset
