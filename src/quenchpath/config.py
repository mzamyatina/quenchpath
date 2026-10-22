"""Run configuration shared across pipeline stages."""

from dataclasses import dataclass
from enum import Enum

import numpy as np


class GridResolution(str, Enum):
    """Spatial resolution for the per-location CH4-to-CO pathway search.
    SAMPLE: ~4896 locations/planet. FULL: every native grid point
    (n_levels x 90 lat x 144 lon - n_levels is the run's own native
    level count, typically 66 but not fixed across planets).
    """

    SAMPLE = "sample"
    FULL = "full"


class SpeciesConversion(str, Enum):
    """What: which species the per-location pathway search runs from/to
    - CH4_CO (the default/original conversion) or CO_CH4.

    Why CO_CH4 exists: added for a like-for-like comparison against
    Zahnle & Marley (2014)'s CO->CH4-fit literature quench timescale
    (see quenchpath.analysis.zahnle_marley_2014's own module docstring;
    quenchpath-apps issue #15).

    Why not named *Direction: a real, separate "direction" concept
    already exists one level down (a single reaction's own
    forward/reverse orientation, e.g. graph.shortest_path.
    build_unidirect_graph()'s edge attribute, or find_pathway()'s
    segments) - conflating the two under one name made call sites
    genuinely ambiguous (a function could have both a `conversion=`
    parameter picking CH4_CO vs CO_CH4 AND, two lines later, a
    `segment["direction"]` meaning forward/reverse of one specific
    reaction). This class is the coarser, system-level choice; the
    other is per-edge and never renamed to match this one.

    What: a directory/filename axis alongside GridResolution - not a
    RunConfig field. `CH4_CO`'s own `.value` is deliberately the
    literal every saved ch4_co_paths/tau_ch4_co_* path already uses
    (see quenchpath.io.paths_dir()/tau_chem_dir()) - passing it
    explicitly, or omitting `conversion=` entirely, reproduces every
    existing path byte-for-byte.

    Why not a RunConfig field: for the same reason grid_resolution
    isn't one - RunConfig's fields (planet/exp/mdh/tK) name the shared
    stage 03/04 inputs (converted number densities, reaction rates),
    which don't depend on which species conversion a later stage
    searches for. Why `CH4_CO`'s `.value` matches the pre-existing
    literal: so no already-saved output/notebook needs to change for
    this to exist.

    What: CH4_CO and CO_CH4 do NOT search the same graph - CH4_CO uses
    `graph.shortest_path.build_unidirect_graph()` (one edge per
    reaction, oriented by local k_eq); CO_CH4 uses
    `build_bidirect_graph()` instead (both directions, each weighted by
    its own real rate).

    Why: see build_bidirect_graph()'s own docstring for why CH4_CO
    can't use it too and why CO_CH4 needs it (quenchpath-apps issue #16), and
    docs/uni_vs_bidirectional_pathway_search.md for the fuller
    methodology writeup.
    """

    CH4_CO = "ch4_co"
    CO_CH4 = "co_ch4"

    @property
    def source(self):
        """Pathway-search start species."""
        return "CH4" if self is SpeciesConversion.CH4_CO else "CO"

    @property
    def target(self):
        """Pathway-search end species."""
        return "CO" if self is SpeciesConversion.CH4_CO else "CH4"

    @property
    def label(self):
        """What: human-readable "X-to-Y" form, e.g. for warnings/echoes
        - `SpeciesConversion.CH4_CO.label == "CH4-to-CO"`.
        Why this exact format: matches every such string already
        hardcoded in the codebase before this existed."""
        return f"{self.source}-to-{self.target}"


def sample_grid_points(full_hgts=None, full_lats=None, full_lons=None):
    """The GridResolution.SAMPLE grid's points along each dimension,
    derived from that dimension's full native coordinate array. Pass only
    the dimensions you need; only the corresponding keys are returned.

    - height: every 4th level.
    - latitude: coarse, symmetric sample around the equator (~10 degrees).
    - longitude: every 9th point (4 per 10 degrees, plus the terminators).

    Returns
    -------
    dict with "hgts"/"lats"/"lons" keys, one per non-None argument.
    """
    points = {}
    if full_hgts is not None:
        points["hgts"] = full_hgts[::4]
    if full_lats is not None:
        assert len(full_lats) == 90, (
            f"got {len(full_lats)} latitude points, expected 90 - the offsets below "
            f"(0:45:6, 44:46, 47::6) are hardcoded to that grid size and would silently "
            f"select a meaningless subset on any other one."
        )
        points["lats"] = np.concatenate(
            (full_lats[0 : 40 + 5 : 6], full_lats[44:46], full_lats[47::6])
        )
    if full_lons is not None:
        assert len(full_lons) == 144, (
            f"got {len(full_lons)} longitude points, expected 144 - [::9] below is "
            f"hardcoded to that grid size."
        )
        points["lons"] = full_lons[::9]  # 4 for every 10 degrees, 9 for terminators
    return points


@dataclass(frozen=True)
class RunConfig:
    """Identifies one planet/experiment/metallicity/diffusion
    combination to process."""

    planet: str
    exp: str = "kin"
    mdh: str = "0"
    tK: str = "1"
