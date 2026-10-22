"""Shared plotting primitives for pres-vs-latitude "slice" figures and
pathway-index color assignment.

Requires matplotlib - install the `viz` extra (`pip install -e .[viz]`).
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from quenchpath.mypaths import path_to_data_root, path_to_plots

# What: set mathtext.default="regular" once here, at import time.
#
# Why: mathtext ($...$) defaults to an italic font - species/subscript
# labels like $CH_4$ or $\tau^{dom}_{CH_4-to-CO}$ would otherwise
# render slanted, unlike the same text written plainly. "regular"
# matches mathtext to the surrounding (upright) font instead.
#
# Why at import time, not per-call: so every script that imports
# anything from this module gets it for free, not just calls that go
# through plot_field_slices()/save_figure().
plt.rcParams["mathtext.default"] = "regular"

# What: pathway-index palette - up to 16 distinct colors (1-8
# ColorBrewer Set2, 9-16 Set1, more saturated). Colors follow whatever
# order assign_path_indices_from_locations() ranks pathways in for the
# dataset being plotted - no fixed color mapping across different
# datasets/runs. NOT machine-validated for CVD/contrast.
#
# Why Set1 (more saturated) for 9-16: so >8 pathways still read as one
# coherent palette.
SET2_COLORS = [
    "#66c2a5",
    "#fc8d62",
    "#8da0cb",
    "#e78ac3",
    "#a6d854",
    "#ffd92f",
    "#e5c494",
    "#b3b3b3",
]
EXTENSION_COLORS = [
    "#e41a1c",
    "#377eb8",
    "#4daf4a",
    "#984ea3",
    "#ff7f00",
    "#a65628",
    "#f781bf",
    "#999999",
]
ALL_COLORS = SET2_COLORS + EXTENSION_COLORS

LON_TITLES = ["Antistellar", "Morning", "Substellar", "Evening"]


def representative_lon_indices(lons):
    """What: indices into sorted `lons` [degrees] for the 4
    representative longitudes LON_TITLES names -
    antistellar/morning/substellar/evening = 0/90/180/270 degrees
    (substellar=180 convention, see quenchpath.stages.stage_07_taus's
    own docstring). Value-based, not a fixed index list.

    Why value-based: works for whatever longitude grid `lons` actually
    is (native, sample, or anything else) instead of assuming one
    specific point count.

    What: on a cell-centered grid (grid points offset from 0/90/180/270
    by half a cell, e.g. UM output starting at 1.25 rather than 0
    degrees), each target sits exactly between two grid points - picks
    the one at or just past the target (ceiling), not
    nearest-with-an-arbitrary-tie-break.

    Why ceiling, not nearest: agrees with stage_07_taus.py's own dayside
    boundary convention (`lons_all >= 90`) - "Morning" lands just
    inside the dayside range, "Evening" just outside it (already
    night), not the reverse.
    """
    lons = np.asarray(lons)
    return [int(np.searchsorted(lons, target)) % len(lons) for target in (0, 90, 180, 270)]


def save_figure(fig, figname):
    """Save `fig` as a png under `path_to_plots`, creating it (and any
    subdirectory `figname` itself names, e.g. `"full/slices_..."`) if
    needed.

    What: `bbox_inches="tight"`.

    Why: constrained_layout (set in plot_field_slices()'s own
    `plt.subplots()` call) can push row/column titles off-canvas once
    the combined legend (contour_legend + extra_legend_handles) grows
    large; a bigger `figsize` doesn't fix this, but expanding the saved
    PNG to fit the actual rendered content does.

    What: prints the saved path relative to `path_to_data_root` (in full
    if `path_to_plots` has been pointed outside it, as tests do).

    Why: notebooks keep this line in their saved outputs, and an absolute
    path would publish the local machine's directory layout.
    """
    out_path = path_to_plots / f"{figname}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    if out_path.is_relative_to(path_to_data_root):
        out_path = out_path.relative_to(path_to_data_root)
    print(f"Saved {out_path}")


def pathway_index_cmap(n_indices):
    """ListedColormap for a pathway-index plot with `n_indices` distinct
    pathways, drawn from ALL_COLORS. Raises ValueError past 16 distinct
    pathways rather than silently cycling/reusing colors.
    """
    if n_indices > len(ALL_COLORS):
        raise ValueError(
            f"Discovered {n_indices} distinct pathways but only {len(ALL_COLORS)} colors are "
            "prepared - add more colors to EXTENSION_COLORS (and validate them) before proceeding."
        )
    return ListedColormap(ALL_COLORS[:n_indices])


def pathway_index_contourf_kw(n_indices):
    """`(kw, cbar_ticks)` for a `contourf`-plotted pathway-index field with
    `n_indices` distinct pathways (1-indexed data: path_index runs
    1..n_indices).

    What: levels are centered on each integer index (bin i covers
    [i-0.5, i+0.5)), so index v is drawn in `ALL_COLORS[v-1]` and the
    colorbar tick at position v is labeled str(v).

    Why not `levels=list(range(n_indices + 1))`: that puts integer
    value v in bin v (colored `ALL_COLORS[v]`, the *next* pathway's
    color) while still labeling that same bin "v+1".
    """
    levels = [i - 0.5 for i in range(1, n_indices + 2)]
    kw = dict(levels=levels, cmap=pathway_index_cmap(n_indices), vmin=0.5, vmax=n_indices + 0.5)
    cbar_ticks = (list(range(1, n_indices + 1)), [str(i) for i in range(1, n_indices + 1)])
    return kw, cbar_ticks


def pathway_index_legend_handles(path_index_map):
    """One colored-patch Legend handle per distinct path_index in
    `path_index_map` (see quenchpath.analysis.path_index.path_index_map_from_locations()),
    colored to match `pathway_index_cmap()`'s own index->color assignment
    and labeled with that index's full reaction-equation chain.

    What: the full `eqns` chain, not just the shorter species-only
    `nodes` chain.

    Why: two distinct path_index values can visit the exact same
    species by different specific reactions (e.g. a different collision
    partner or H-abstraction vs M-assisted dissociation at the same
    step) - `nodes` alone can be identical across entries that still
    need distinguishing here; `eqns` cannot
    (assign_path_indices_from_locations() ranks pathways by their
    `eqns` sequence, so distinct indices are guaranteed distinct
    `eqns`).

    What: meant for `plot_field_slices()`'s `extra_legend_handles` -
    pass alongside any quench-level `contour_legend` so both end up in
    one combined figure legend, not two separate ones.

    Why one combined legend: constrained_layout doesn't stack multiple
    "outside" legends - the second one drawn displaces the first rather
    than appearing below it.

    What: each eqn's own "↔" is replaced with "→"/"←" per that
    segment's `directions` entry (reactants/products keep the network
    file's own left/right order - only the arrow flips).

    Why replace it: a bare "↔" doesn't say which way the reaction
    actually went for this pathway, and `directions` already carries
    that.

    What: "⇒" (not "→", already used for a single reaction's own
    direction) chains successive reactions together.

    Why "⇒", not "→": so the two never look like the same kind of
    arrow.

    What: an entry with an empty `eqns` (quenchpath.graph.shortest_path.
    find_pathway() raised NetworkXNoPath at every location sharing
    this index - see quenchpath.stages.stage_05_paths_serial's own
    NetworkXNoPath handling) gets a fixed "(no path found)" label
    rather than joining zero segments into an empty string.

    Why not an empty string: a blank legend entry would silently look
    like a missing/broken color, not a real, named category of
    location.
    """
    return [
        Patch(
            color=ALL_COLORS[int(idx) - 1],
            label=f"{idx}: "
            + (
                " ⇒ ".join(
                    eqn.replace("↔", "→" if direction == "forward" else "←")
                    for eqn, direction in zip(entry["eqns"], entry["directions"])
                )
                or "(no path found)"
            ),
        )
        for idx, entry in sorted(path_index_map.items(), key=lambda kv: int(kv[0]))
    ]


def quench_level_contour_panels(
    lon_idx,
    ratio_net_ls=None,
    ratio_net_qn=None,
    ratio_rls_ls=None,
    ratio_rls_qn=None,
    ratio_zahnle_ls=None,
    ratio_zahnle_qn=None,
    ratio_net_ls_label=None,
    ratio_net_qn_label=None,
    ratio_rls_ls_label=None,
    ratio_rls_qn_label=None,
    ratio_zahnle_ls_label=None,
    ratio_zahnle_qn_label=None,
    zahnle_color="tab:red",
    conversion="ch4_co",
    region_bounds=None,
):
    """What: build `plot_field_slices()`-ready `contour_panels` (and a
    matching legend) for up to 6 ratio=1 "quench level" overlays, plus
    any number of `region_bounds` (see that parameter's own paragraph
    below) - one per tau_ch4/tau_adv combination
    quenchpath.analysis.quench_level defines (net_ls dashed, net_qn
    solid, rls_ls dashdot, rls_qn dotted - matplotlib's 4 named
    linestyles, one each, all drawn in `plot_field_slices()`'s own
    `contour_color`, black by default), plus 2 more pairing the same
    tau_adv_<ls|qn> against Zahnle & Marley (2014)'s literature
    timescale instead (zahnle_ls dashed, zahnle_qn solid) - one set per
    longitude index in `lon_idx`.

    Why zahnle_ls/qn reuse net_ls/net_qn's own linestyles rather than
    adding new ones: ls-vs-qn is what a linestyle already encodes here.

    Why `zahnle_color`, not black: marks them as the literature
    comparison rather than a 5th quenchpath-internal combination.

    `ratio_net_ls`/`ratio_net_qn`/`ratio_rls_ls`/`ratio_rls_qn`/
    `ratio_zahnle_ls`/`ratio_zahnle_qn` are already-divided ratio arrays
    (see quenchpath.analysis.quench_level's own quench_level() - called
    once per net/rls x ls/qn combination, e.g.
    quench_level(cfg, grid_resolution, "net", "ls") for ratio_net_ls -
    plus quench_level_zahnle_ls()/quench_level_zahnle_qn()), each full
    (level_height, latitude, longitude) shape - pass whichever you have,
    the ones you omit (None) simply aren't drawn.

    Returns `(contour_panels, legend)` - both `None` if none are given.
    `legend` is a list of (label, linestyle) pairs or (label, linestyle,
    color) triples - same plain-vs-override convention as
    `contour_panels`'s own tuples above (net/rls entries are plain
    pairs, following `plot_field_slices()`'s own `contour_color`;
    zahnle entries are triples, always `zahnle_color`). The `*_label`
    keywords override that combination's default label text.

    What: `conversion` - `"ch4_co"` (default) or `"co_ch4"` (accepts
    either a plain string or a `quenchpath.config.SpeciesConversion`
    value) - which species the quenchpath-derived ratios' default
    labels (net_ls/net_qn/rls_ls/rls_qn only, not the zahnle ones -
    Zahnle & Marley (2014)'s own fit is always framed as t_CO) name as
    source-to-target. Only affects the default text; the `*_label`
    overrides above still take precedence regardless of `conversion`.

    Why not import SpeciesConversion here: keeps this module
    iris/config-free.

    What: every ratio here is a full (level, latitude, longitude) array,
    even the `_ls` combinations (`net_ls`/`rls_ls`/`zahnle_ls`) -
    dividing by `tau_adv_ls_dom` (a day-side, level-only spatial mean
    with no per-location pressure of its own - see
    quenchpath.analysis.quench_level's own module docstring) still
    broadcasts against the numerator's own full lat/lon grid, so the
    result varies by location the same as any `_qn` combination.

    Why this matters for plotting: every one of these ratios lands on
    the exact same (level, latitude, longitude) grid
    `plot_field_slices()`'s own color field (e.g. path_index_cube) is
    drawn on - both come from the same merge-paths .nc file (see
    quenchpath.analysis.quench_level.quench_level()'s own docstring) -
    so `plot_field_slices()`'s real local pressure is a valid y-axis
    for all 6 combinations here, not just the `_qn` ones, and is what
    it defaults to (`pres_mode="local"`) when `get_contour_coords`
    isn't given.

    What: `region_bounds`, if given, is a list of `(ratio, low, high,
    label, linestyle, color)` tuples - each drawn as 2 more contour
    lines (at `low`/`high`, same ratio array as one of the 6
    combinations above, e.g. `ratio_net_qn`) marking a quench REGION's
    lower/upper bound rather than a single ratio=1 crossing (see
    quenchpath.analysis.quench_region.quench_region() for how such a
    bound gets built - `low`/`high` there are
    `exp(-bound)`/`exp(bound)`). Unlike `combinations`, each entry needs
    its own explicit label/linestyle/color.

    Why exp(-bound)/exp(bound): this function's own `combinations` list
    above already establishes that a "ratio=X" line is drawn on the
    *linear* ratio array, not a log one.

    Why each entry needs its own label/linestyle/color: there's no
    fixed convention for how many of these a caller passes or what each
    one means.
    """
    if (
        ratio_net_ls is None
        and ratio_net_qn is None
        and ratio_rls_ls is None
        and ratio_rls_qn is None
        and ratio_zahnle_ls is None
        and ratio_zahnle_qn is None
        and not region_bounds
    ):
        return None, None

    # What: plain (non-LaTeX) "source-to-target" fragment, spliced into
    # each default label below via string concatenation rather than an
    # f-string. Duck-typed on conversion == "co_ch4" instead of
    # importing SpeciesConversion.source/target/label.
    #
    # Why string concatenation, not an f-string: avoids escaping every
    # literal "{"/"}" the surrounding mathtext already needs.
    #
    # Why duck-typed, not imported: keeps this module iris/config-free.
    source_tex, target_tex = ("CO", "CH_4") if conversion == "co_ch4" else ("CH_4", "CO")
    x_to_y = source_tex + "-to-" + target_tex

    # (ratio, label override, default label, linestyle, color) - one
    # entry per combination this function supports, in the fixed
    # net_ls/net_qn/rls_ls/rls_qn/zahnle_ls/zahnle_qn order every
    # legend/contour-panel list below follows. color=None means "follow
    # plot_field_slices()'s own contour_color" (the quenchpath-internal
    # 4, a plain 3-tuple); the zahnle pair always gets zahnle_color (a
    # 4-tuple) - see this function's own docstring for both conventions.
    combinations = [
        (
            ratio_net_ls,
            ratio_net_ls_label,
            r"$\tau^{net}_{" + x_to_y + r"}/\tau^{LS,dom}_{adv}=1$",
            "dashed",
            None,
        ),
        (
            ratio_net_qn,
            ratio_net_qn_label,
            r"$\tau^{net}_{" + x_to_y + r"}/\tau^{QN}_{adv}=1$",
            "solid",
            None,
        ),
        (
            ratio_rls_ls,
            ratio_rls_ls_label,
            r"$\tau^{RLS}_{" + x_to_y + r"}/\tau^{LS,dom}_{adv}=1$",
            "dashdot",
            None,
        ),
        (
            ratio_rls_qn,
            ratio_rls_qn_label,
            r"$\tau^{RLS}_{" + x_to_y + r"}/\tau^{QN}_{adv}=1$",
            "dotted",
            None,
        ),
        (
            ratio_zahnle_ls,
            ratio_zahnle_ls_label,
            r"Zahnle & Marley (2014) $t_{CO}/\tau^{LS,dom}_{adv}=1$",
            "dashed",
            zahnle_color,
        ),
        (
            ratio_zahnle_qn,
            ratio_zahnle_qn_label,
            r"Zahnle & Marley (2014) $t_{CO}/\tau^{QN}_{adv}=1$",
            "solid",
            zahnle_color,
        ),
    ]

    legend = []
    contour_panels = [[] for _ in lon_idx]
    for ratio, label_override, default_label, ls, color in combinations:
        if ratio is None:
            continue
        style = (ls,) if color is None else (ls, color)
        legend.append((label_override or default_label, *style))
        for panels, lo in zip(contour_panels, lon_idx):
            panels.append((ratio[..., lo], [1], *style))

    for ratio, low, high, label, ls, color in region_bounds or []:
        style = (ls,) if color is None else (ls, color)
        legend.append((label, *style))
        for panels, lo in zip(contour_panels, lon_idx):
            panels.append((ratio[..., lo], [low, high], *style))

    return contour_panels, legend


def slice_panel_coords(lats, pres, pres_mode, panel_data):
    """Per-column (X, Y) coordinate arrays for one row's 4 panels, plus
    that row's own (pres_min, pres_max) actually present - shared by
    `plot_field_slices()` and any caller building its own grid instead
    (e.g. a figure needing a per-row colorbar, which that primitive
    doesn't support).

    `pres_mode="mean"`: `pres` is one 1D `(level,)` profile shared by
    all 4 columns (e.g. `tau_adv_ls_dom`'s own day-side spatial-mean
    pressure - see quenchpath.analysis.quench_level's own module
    docstring for why it's not genuine local pressure) - each
    `panel_data[col]` must be `(len(pres), len(lats))`.

    `pres_mode="local"`: `pres` is a list of 4 arrays, `pres[col]` that
    column's own true local pressure, `(level, lat)` - genuinely
    different per column - each `panel_data[col]` must match
    `pres[col]`'s own shape exactly.

    Why genuinely different per column: real air_pressure at fixed
    height varies with latitude AND longitude, not just level.

    Returns `(x_cols, y_cols, pres_min, pres_max)` - `x_cols`/`y_cols`
    each a list of 4 arrays ready for `pcolormesh`/`contourf` (`x_cols`
    already broadcast to 2D in `"local"` mode, where matplotlib needs
    matching-dimensionality X/Y once either one is 2D).

    What: shapes are checked, not just assumed.

    Why: a mismatch would otherwise surface as a cryptic matplotlib
    shape error, or - if shapes happen to be broadcastable - a silently
    wrong plot.
    """
    assert pres_mode in ("mean", "local"), (
        f"pres_mode must be 'mean' or 'local', not {pres_mode!r}."
    )
    x_cols, y_cols = [], []
    pres_min, pres_max = np.inf, -np.inf
    for col, data in enumerate(panel_data):
        if pres_mode == "local":
            y = pres[col]
            assert np.ndim(y) == 2, (
                f"pres_mode='local' expects one 2D (level, lat) array per column, got a "
                f"{np.ndim(y)}D array for column {col} - did you mean pres_mode='mean'?"
            )
            assert data.shape == y.shape, (
                f"column {col}: panel data shape {data.shape} doesn't match its own local "
                f"pressure's shape {y.shape} - not on the same grid."
            )
            x = np.broadcast_to(lats, y.shape)
        else:
            y = pres
            assert np.ndim(y) == 1, (
                f"pres_mode='mean' expects one 1D (level,) profile shared by all 4 columns, "
                f"got a {np.ndim(y)}D array - did you mean pres_mode='local' (a list of 4 "
                f"per-column 2D arrays)?"
            )
            assert data.shape == (len(y), len(lats)), (
                f"column {col}: panel data shape {data.shape} doesn't match "
                f"(len(pres), len(lats)) = ({len(y)}, {len(lats)}) - not on the same grid."
            )
            x = lats
        x_cols.append(x)
        y_cols.append(y)
        pres_min = min(pres_min, np.min(y))
        pres_max = max(pres_max, np.max(y))
    return x_cols, y_cols, pres_min, pres_max


def check_contour_coords(contour_lats, contour_pres, contour_panels, pres_min, pres_max, planet=""):
    """Validate a `get_contour_coords()` return - against the contour
    data it's paired with, and against the field's own already-computed
    pressure range (`pres_min`/`pres_max`, from `slice_panel_coords()`).

    What: `contour_pres`'s own shape is detected here (a plain 1D array
    = shared across all 4 columns; a list of 4 arrays = one per
    column), independent of the field's own `pres_mode`.

    Why independent: a contour pairing with a genuinely different
    pressure convention from the color field's own is a legitimate
    combination (e.g. a coarse mean contour over a local-pressure
    field), not something to force into lockstep with `pres_mode` -
    the common case (both share the field's own local pressure) is
    just what `plot_field_slices()` defaults `contour_pres` to when
    `get_contour_coords` isn't given (see its own docstring).

    What: the range-overlap check exists.

    Why: a contour silently drawn off-panel (a unit mismatch, e.g. Pa
    vs bar, or simply the wrong array passed in) produces no error at
    all - just a quench line that's quietly missing from the saved
    figure, easy to not notice.

    What: deliberately a range *intersection* test, not an equality
    test.

    Why intersection, not equality: a color field and its contour
    legitimately using two different pressure conventions on the same
    axes is fine; one of them being off the visible axis entirely is
    not.
    """
    contour_pres_is_local = isinstance(contour_pres, (list, tuple))
    pres_cols = list(contour_pres) if contour_pres_is_local else [contour_pres] * 4
    if contour_panels is not None:
        for col, (col_panels, y) in enumerate(zip(contour_panels, pres_cols)):
            expected_shape = y.shape if contour_pres_is_local else (len(y), len(contour_lats))
            for data, _levels, _ls, *_color in col_panels or []:
                assert data.shape == expected_shape, (
                    f"{planet}: column {col} contour data shape {data.shape} doesn't match "
                    f"{expected_shape} - not on the same grid."
                )
    c_min = min(np.min(y) for y in pres_cols)
    c_max = max(np.max(y) for y in pres_cols)
    assert c_max >= pres_min and c_min <= pres_max, (
        f"{planet}: contour pressure range [{c_min:.3g}, {c_max:.3g}] Pa doesn't overlap the "
        f"field's own plotted range [{pres_min:.3g}, {pres_max:.3g}] Pa - the quench line would "
        f"be drawn entirely off-panel (a unit mismatch, e.g. Pa vs bar, or the wrong array, are "
        f"the likely causes)."
    )


# What: fixed inter-column gap (figure-fraction, tuned for this
# module's own 14in-wide default canvas - see plot_field_slices()'s own
# figsize= default) for pack_columns_tightly() below.
#
# Why this exact value: measured directly off
# notebooks/plt_slices.ipynb's own hatp11b_mdh0/sample/CH4-to-CO
# figure (13 combined legend/colorbar rows - see that figure's own
# `n_indices`+quench-line-legend count) - the figure whose spacing a
# 2026-09 review found preferable and asked to be applied everywhere.
# Deliberately NOT re-derived per figure: see pack_columns_tightly()'s
# own docstring for why a fixed target is needed at all.
_SLICES_COLUMN_GAP = 0.022737635742918982


def pack_columns_tightly(fig, col_gap=_SLICES_COLUMN_GAP):
    """Repositions every panel's own x0 left to right (y0/height
    untouched), anchored at column 0's own natural left edge and using
    row 0/column 0's own natural panel width, so every column ends up
    exactly `col_gap` (figure-fraction) apart - closing plot_field_slices()'s
    own inter-column gap to one fixed, predictable value instead of
    whatever `box_aspect=1` (see the `set_box_aspect(1)` call below)
    happened to leave over.

    Why the gap isn't already fixed by default: `box_aspect=1` forces
    every panel square in *display inches*, computed by
    constrained_layout as the largest square that fits the available
    room - and the panel row's own available *height* shrinks with
    however much legend/colorbar content a given figure's own
    `contour_legend`/`extra_legend_handles` need (more known pathways,
    more quench-line overlays -> more legend rows -> less height ->
    (via box_aspect) a smaller/narrower square panel -> more leftover
    *width*). constrained_layout dumps that leftover width as *extra*
    inter-column gap rather than outer margin, regardless of any
    "wspace"/"w_pad" rcParam - so two figures at the identical figsize
    can end up with visibly different column spacing for no reason
    other than how many pathways/overlays that particular run happens
    to have (confirmed empirically: hatp11b_mdh0/sample's own
    CH4-to-CO vs CO-to-CH4 figures - 6 vs 2 known pathways, 13 vs 8
    total legend rows - solve to a ~0.32in vs ~0.13in gap at the exact
    same 14in-wide canvas).

    Why repositioning (not resizing) is the fix: this leaves each
    panel's own natural (box_aspect-solved) size alone entirely -
    fixing the *gap* alone, at a value taken from one specific
    already-good reference figure, needs no colorbar/legend
    repositioning to match (unlike shrinking panels down to a common
    reference size would - that would leave a visible gap between a
    smaller panel row and colorbar/legend still placed as if for a
    bigger one). Any leftover width this reclaims from the gap becomes
    outer right margin instead - unused, but not a coherence problem
    the way an inconsistent gap is.

    Freezes the layout engine (`fig.set_layout_engine(None)`) - must
    run after every other artist (colorbar, legend, row/column titles)
    is already placed, since those all still need constrained_layout
    active to reserve their own room; freezing only stops instructions
    to slide the panels back apart, and does not disturb their
    already-decided sizes/positions."""
    fig.canvas.draw()
    main_axes = [a for a in fig.axes if a.get_subplotspec() is not None]
    row0 = [a for a in main_axes if a.get_subplotspec().rowspan.start == 0]
    p00 = min(row0, key=lambda a: a.get_subplotspec().colspan.start).get_position()
    panel_w, left0 = p00.width, p00.x0

    fig.set_layout_engine(None)
    for a in main_axes:
        col = a.get_subplotspec().colspan.start
        pos = a.get_position()
        new_x0 = left0 + col * (panel_w + col_gap)
        a.set_position([new_x0, pos.y0, panel_w, pos.height])


def plot_field_slices(
    planets,
    get_row_data,
    kw,
    cbar_label,
    figname,
    *,
    pres_mode="local",
    get_contour_panels=None,
    get_contour_coords=None,
    contour_color="k",
    get_hatch_panels=None,
    hatch_pattern="///",
    cbar_ticks=None,
    ptype="pcolormesh",
    contour_legend=None,
    extra_legend_handles=None,
    legend_title=None,
    planet_names=None,
    figsize=None,
    lon_titles=None,
):
    """`len(planets)` (rows) x 4 (representative-longitude columns:
    antistellar/morning/substellar/evening) pres-vs-latitude slice grid.
    `len(planets) == 1` is the single-planet layout; `len(planets) > 1`
    is the multi-planet layout - same loop produces both.

    `pres_mode="local"` (default): `get_row_data(planet)` -> `(lats,
    pres_cols, panel_data)`, where `pres_cols` is a list of 4 arrays,
    `pres_cols[col]` that column's own true local pressure - genuinely
    different per column (level, latitude, AND longitude).

    Why local by default: it's the physically correct pressure for
    every field this module plots except the tau_adv_ls_* cubes
    themselves (a day-side, level-only spatial mean with no
    per-location pressure of its own - see
    quenchpath.analysis.quench_level's own module docstring); pass
    `pres_mode="mean"` for one of those instead.

    `pres_mode="mean"`: `get_row_data(planet)` -> `(lats, pres,
    panel_data)`, where `pres` is one 1D `(level,)` profile shared by
    all 4 longitude columns (see `slice_panel_coords()`'s own
    docstring) and `panel_data` is a list of 4 2D `(level, lat)`
    arrays, one per representative longitude.

    What: in both modes, the shared y-axis range is the full pressure
    range actually present across every row's own pressure (not a
    fixed guess).

    Why: each planet/row's native grid reaches a different true
    pressure at its top/bottom level (see quenchpath-apps issue #11).

    What: `get_contour_panels(planet)`, if given, -> a list of 4
    entries (one per longitude panel), each either None or a list of
    (data, levels, ls) or (data, levels, ls, color) tuples (see
    `quench_level_contour_panels()`) - the plain 3-tuple form draws in
    `contour_color` (default `"k"`, black - pass a different color for
    a caller that wants its own convention, e.g. grey); the 4-tuple
    form overrides that per-line.

    Why the 4-tuple form exists: e.g. for a literature-comparison
    overlay drawn in a different color from quenchpath's own lines
    within the same panel. No additional styling beyond color (e.g. no
    path-effects stroke/halo) - keep it that way in any future change
    here.

    `get_contour_coords(planet)`, if given, -> `(contour_lats,
    contour_pres)` for that overlay - `contour_pres`'s own shape
    follows `pres_mode`, same convention as `pres` above (a single 1D
    array in `"mean"` mode, a list of 4 per-column 2D arrays in
    `"local"` mode). Defaults to the row's own `lats`/`pres` (from
    `get_row_data` above) when omitted, in either mode - the common
    case, since every quench-level ratio this module computes shares
    its color field's exact grid (see
    `quench_level_contour_panels()`'s own docstring). Pass this only
    when a contour genuinely needs a *different* pressure convention
    from the field it's drawn over (e.g. `pres_mode="mean"`'s own
    tau_adv_ls_dom-only case).

    What: `get_hatch_panels(planet)`, if given, -> a list of 4 entries
    (one per longitude panel), each either None or a 2D boolean array
    (True = hatch that cell) matching that panel's own
    `panel_data[col]` shape - e.g. a literature timescale's own
    calibration-envelope mask (see
    quenchpath.analysis.zahnle_marley_2014.zahnle_marley_2014_invalid_mask()).
    Drawn as an unfilled (`colors="none"`) `contourf` with
    `hatch_pattern` (default `"///"`) on top of the main field but
    underneath any contour lines. Uses `panel_data[col]`'s own (x, y)
    coordinates, not `contour_lats`/`contour_pres`.

    Why underneath contour lines: so a quench line crossing a hatched
    region still reads clearly.

    Why the field's own coordinates, not contour_lats/pres: a validity
    mask is a per-gridpoint property of the field itself, not a derived
    quench-line quantity, so it belongs on the field's own true grid
    regardless of `pres_mode`.

    What: `contour_legend`, if given, is a list of (label, linestyle)
    or (label, linestyle, color) tuples drawn as a small figure-level
    legend (plain 2-tuples default to black, matching `contour_color`'s
    own default). `extra_legend_handles`, if given, is a list of
    already-built Legend handles (e.g. Patch instances - see
    `pathway_index_legend_handles()`) appended to that same legend, not
    a second one.

    Why one legend, not two: constrained_layout doesn't stack multiple
    "outside" legends cleanly, so anything that needs to appear
    alongside `contour_legend` has to go through this one call.

    `legend_title`, if given, is that legend's title text (e.g.
    explaining a pathway-index ranking convention) - ignored if neither
    `contour_legend` nor `extra_legend_handles` is given (no legend to
    title).

    `planet_names`, if given, is a {planet_key: display_name} mapping
    used for each row's title (e.g. loaded from
    quenchpath.mypaths.path_to_planets_yaml) - falls back to the raw
    planet key when not given or when a planet is missing from it.

    What: `figsize`, if given, overrides this function's own default
    canvas size (`(14, 8)` for a single planet, `(14, 4 * len(planets))`
    otherwise).

    Why a caller would override it: wants the saved figure to already
    be close to its final printed size/aspect ratio rather than relying
    on a downstream `\\includegraphics` scale factor to shrink a much
    larger canvas (which shrinks its text by the same factor) - such a
    caller should pair this with its own `rcParams["font.size"]`, not
    something this function sets itself, since the "right" font size
    depends on that caller's own target figsize, not on this function's
    default one.

    What: `lon_titles`, if given, overrides this function's own default
    per-column titles (`LON_TITLES` - "Antistellar"/"Morning"/
    "Substellar"/"Evening"). Must still be 4 entries, same
    antistellar/morning/substellar/evening order.

    Why a caller would override it: panels too narrow for the full
    names - row 0's `loc="right"` column title and column 0's
    `loc="left"` row title share the same Axes and can visually collide
    there once panels get small enough - shorter titles are the direct
    fix.
    """
    if isinstance(planets, str):
        raise TypeError("planets must be a list of planet names, not a bare string")
    if pres_mode not in ("mean", "local"):
        raise ValueError(f"pres_mode must be 'mean' or 'local', not {pres_mode!r}.")

    if figsize is None:
        figsize = (14, 8) if len(planets) == 1 else (14, 4 * len(planets))
    if lon_titles is None:
        lon_titles = LON_TITLES
    elif len(lon_titles) != 4:
        raise ValueError(f"lon_titles must have exactly 4 entries, got {len(lon_titles)}.")
    # What: rasterize the main field (pcolormesh/contourf, one per
    # panel - by far the heaviest artist here) by default. A caller
    # that explicitly passes rasterized=False in `kw` still wins (dict
    # unpacking after this default overrides it).
    #
    # Why: so that a PDF/SVG export of this figure (if one is ever
    # added - save_figure() only writes PNG today, where this flag is a
    # no-op) stays a manageable size instead of embedding every one of
    # its thousands of individual grid cells as its own vector path.
    kw = {"rasterized": True, **kw}
    fig, ax = plt.subplots(
        nrows=len(planets),
        ncols=4,
        figsize=figsize,
        sharex=True,
        sharey=True,
        constrained_layout=True,
        squeeze=False,
    )
    p0 = None
    pres_all_min, pres_all_max = np.inf, -np.inf
    for row, planet in enumerate(planets):
        lats, pres, panel_data = get_row_data(planet)
        x_cols, y_cols, pres_min, pres_max = slice_panel_coords(lats, pres, pres_mode, panel_data)
        pres_all_min = min(pres_all_min, pres_min)
        pres_all_max = max(pres_all_max, pres_max)
        contour_panels = get_contour_panels(planet) if get_contour_panels is not None else None
        hatch_panels = get_hatch_panels(planet) if get_hatch_panels is not None else None
        if get_contour_coords is not None:
            contour_lats, contour_pres = get_contour_coords(planet)
        else:
            contour_lats, contour_pres = lats, pres
        check_contour_coords(contour_lats, contour_pres, contour_panels, pres_min, pres_max, planet)
        contour_pres_is_local = isinstance(contour_pres, (list, tuple))
        for col, (lo_title, iax) in enumerate(zip(lon_titles, ax[row])):
            plot_fn = iax.contourf if ptype == "contourf" else iax.pcolormesh
            p0 = plot_fn(x_cols[col], y_cols[col], panel_data[col], **kw)
            if hatch_panels is not None and hatch_panels[col] is not None:
                mask = hatch_panels[col]
                assert mask.shape == panel_data[col].shape, (
                    f"{planet}: hatch mask shape {mask.shape} doesn't match panel data shape "
                    f"{panel_data[col].shape} - not on the same grid."
                )
                # levels=[0.5, 1.5] around the boolean-as-int mask's only
                # "on" value (1) - drawn even when `mask` is all-False
                # (an empty, harmless contourf) so a caller doesn't need
                # to special-case "no invalid cells this panel" itself.
                iax.contourf(
                    x_cols[col],
                    y_cols[col],
                    mask.astype(int),
                    levels=[0.5, 1.5],
                    colors="none",
                    hatches=[hatch_pattern],
                )
            if contour_panels is not None and contour_panels[col] is not None:
                # What: contour_x/contour_y computed per column here,
                # not once for the whole row - same broadcast as
                # x_cols/y_cols above (slice_panel_coords()), needed
                # whenever contour_pres is itself a per-column 2D array
                # (contour_pres_is_local, independent of pres_mode -
                # see check_contour_coords()'s own docstring).
                if contour_pres_is_local:
                    contour_x = np.broadcast_to(contour_lats, contour_pres[col].shape)
                    contour_y = contour_pres[col]
                else:
                    contour_x, contour_y = contour_lats, contour_pres
                for entry in contour_panels[col]:
                    data, levels, ls, *rest = entry
                    color = rest[0] if rest else contour_color
                    iax.contour(
                        contour_x,
                        contour_y,
                        data,
                        levels=levels,
                        colors=color,
                        linestyles=ls,
                    )
            iax.set_yscale("log")
            if row == 0:
                iax.set_title(lo_title, loc="right")
            if row == len(planets) - 1:
                iax.set_xlabel("Latitude [degrees]")
        ax[row, 0].set_ylabel("Pressure [Pa]")
        row_title = planet_names.get(planet, planet) if planet_names else planet
        ax[row, 0].set_title(row_title, loc="left")
    # Full pressure range actually present across every row plotted here
    # (not a fixed guess) - each planet/row's own native grid reaches a
    # different true pressure at its top/bottom level (see quenchpath-apps issue #11), so
    # a shared fixed range risks silently clipping some rows' data.
    for iax in ax.ravel():
        iax.set_ylim(pres_all_max, pres_all_min)
        # What: fixed box aspect (not data-unit equal aspect) - square
        # panels, not wide rectangles, holds regardless of x/y range.
        # Why not data-unit equal aspect: would be meaningless here -
        # the y-axis is log-pressure, the x-axis is plain-degrees
        # latitude, two unrelated units.
        iax.set_box_aspect(1)
    # What: a second bar axis on just the last (bottom-right) panel (Pa
    # is every panel's own axis, shared, set above) - a secondary_yaxis
    # (not a second data axis) so it tracks the shared log-scale limits
    # automatically, 1 bar = 1e5 Pa.
    # Why just the last panel: enough for a quick unit cross-reference
    # without repeating it on every column.
    secax = ax[-1, -1].secondary_yaxis("right", functions=(lambda p: p / 1e5, lambda b: b * 1e5))
    secax.set_ylabel("Pressure [bar]")
    cbar = fig.colorbar(p0, ax=ax, orientation="horizontal", label=cbar_label, shrink=0.6)
    if cbar_ticks is not None:
        cbar.ax.set_xticks(cbar_ticks[0], cbar_ticks[1])
    if contour_legend is not None or extra_legend_handles is not None:
        handles = list(extra_legend_handles or [])
        for entry in contour_legend or []:
            label, ls, *rest = entry
            color = rest[0] if rest else "k"
            handles.append(Line2D([0], [0], color=color, linestyle=ls, label=label))
        # What: "outside lower center" location.
        # Why: a constrained_layout-aware location (mpl >=3.7) that
        # reserves its own space, so it survives the tight-bbox PNG
        # save intact.
        #
        # What: one column when extra_legend_handles entries are
        # present; otherwise side-by-side as before.
        # Why: extra_legend_handles entries have long text (e.g. a full
        # reaction-equation chain - needs its own line each);
        # contour_legend entries alone are short.
        ncol = 1 if extra_legend_handles else len(handles)
        fig.legend(
            handles=handles,
            loc="outside lower center",
            ncol=ncol,
            fontsize="medium",
            title=legend_title,
            title_fontproperties={"weight": "bold", "size": "medium"},
            alignment="left",
        )
    # What: after every other artist (colorbar, legend, titles) is
    # already placed - see pack_columns_tightly()'s own docstring for
    # why the ordering matters and why this is needed at all.
    pack_columns_tightly(fig)
    save_figure(fig, figname)
    return fig
