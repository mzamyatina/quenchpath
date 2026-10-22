"""Tests for quenchpath.viz.plot_field_slices()'s two pres_mode's
(`"mean"`: one shared 1D pressure profile per row; `"local"`: one
genuine 2D per-column local-pressure array each - see that function's
own docstring), the validation both modes are checked against, and the
quench-line contour's own no-stroke styling constraint.

Synthetic tiny arrays throughout - no real NetCDF/pipeline data needed,
matching this test suite's own convention (see conftest.py's module
docstring). Requires the `viz` extra (`pip install -e .[viz]`) -
skipped entirely if matplotlib isn't installed, same condition that
gates quenchpath.viz's own usability.
"""

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")  # headless - no display needed/available in CI

import matplotlib.colors  # noqa: E402
import matplotlib.contour  # noqa: E402
import matplotlib.figure  # noqa: E402

from quenchpath import viz  # noqa: E402

LATS = np.array([-60.0, -20.0, 20.0, 60.0])
PRES_MEAN = np.array([1e5, 1e4, 1e3])  # 3 levels, decreasing = higher altitude
_RNG = np.random.default_rng(0)


def _field_data():
    """4 panels' worth of (level, lat) field data, one per representative
    longitude - shape matches PRES_MEAN x LATS."""
    return [_RNG.random((len(PRES_MEAN), len(LATS))) for _ in range(4)]


def _local_pres_cols():
    """4 columns' own local pressure, each a distinct (level, lat) array -
    same level range as PRES_MEAN but genuinely varying by latitude too,
    unlike it."""
    return [PRES_MEAN[:, None] * (1 + 0.1 * i + 0.01 * np.arange(len(LATS))) for i in range(4)]


def _contour_panels(pres):
    """One dashed ratio=1 contour panel per column, all on `pres` (1D) -
    values deliberately straddle 1 so a real line gets drawn, not an
    empty QuadContourSet."""
    ratio = np.linspace(0.5, 2.0, len(pres))[:, None] * np.ones((1, len(LATS)))
    return [[(ratio, [1], "dashed")] for _ in range(4)]


def _get_row_data_mean(planet):
    return LATS, PRES_MEAN, _field_data()


def _get_row_data_local(planet):
    return LATS, _local_pres_cols(), _field_data()


def _get_contour_panels(planet):
    return _contour_panels(PRES_MEAN)


def _get_contour_coords(planet):
    return LATS, PRES_MEAN


def test_mean_mode_smoke(tmp_path, monkeypatch):
    """pres_mode="mean" (quenchpath's original behavior, kept for the
    tau_adv_ls_* family that has no local pressure of its own - no
    longer the default) - still saves a real figure."""
    monkeypatch.setattr(viz, "path_to_plots", tmp_path)
    fig = viz.plot_field_slices(
        ["planetA"], _get_row_data_mean, {}, "cbar label", "test_mean", pres_mode="mean"
    )
    assert (tmp_path / "test_mean.png").exists()
    assert len(fig.axes) > 0


def test_local_mode_smoke(tmp_path, monkeypatch):
    """pres_mode="local" (the default) with an explicit
    get_contour_coords override - the per-column true-local-pressure
    path."""
    monkeypatch.setattr(viz, "path_to_plots", tmp_path)
    fig = viz.plot_field_slices(
        ["planetA"],
        _get_row_data_local,
        {},
        "cbar label",
        "test_local",
        pres_mode="local",
        get_contour_panels=_get_contour_panels,
        get_contour_coords=_get_contour_coords,
    )
    assert (tmp_path / "test_local.png").exists()
    assert len(fig.axes) > 0


def test_local_mode_without_contour_coords_reuses_the_row_own_pressure(tmp_path, monkeypatch):
    """pres_mode="local" with no get_contour_coords given - defaults to
    the row's own local pres_cols (from get_row_data), the common case
    for every quench-level ratio this module computes (see
    quench_level_contour_panels()'s own docstring: they share their
    color field's exact grid) - no longer a TypeError (that hard
    requirement is gone now that a real per-quantity default exists)."""
    monkeypatch.setattr(viz, "path_to_plots", tmp_path)
    fig = viz.plot_field_slices(
        ["planetA"],
        _get_row_data_local,
        {},
        "cbar label",
        "test_local_default_contour",
        pres_mode="local",
        get_contour_panels=_get_contour_panels,
    )
    contour_sets = [
        artist
        for ax in fig.axes
        for artist in ax.collections
        if isinstance(artist, matplotlib.contour.QuadContourSet)
    ]
    assert contour_sets, "expected at least one contour artist to check"


def test_save_figure_uses_a_tight_bbox(tmp_path, monkeypatch):
    """save_figure() must save with bbox_inches="tight" - regression
    guard for a real bug this session: constrained_layout can push row/
    column titles below the visible canvas entirely once the "outside
    lower center" legend grows large (a 6-pathway-index run's own
    combined legend did this for real), and a bare (non-tight) save
    writes exactly that broken, title-less canvas to disk with no error
    at all. A wider/taller explicit figsize doesn't fix this reliably
    (tried first, made it worse); bbox_inches="tight" does, by saving
    whatever was actually rendered rather than trusting the nominal
    figsize to already contain it."""
    monkeypatch.setattr(viz, "path_to_plots", tmp_path)
    calls = []
    real_savefig = matplotlib.figure.Figure.savefig

    def fake_savefig(self, *args, **kwargs):
        calls.append(kwargs)
        return real_savefig(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", fake_savefig)

    viz.plot_field_slices(
        ["planetA"], _get_row_data_mean, {}, "cbar label", "test_tight_bbox", pres_mode="mean"
    )

    assert len(calls) == 1
    assert calls[0].get("bbox_inches") == "tight"


def test_save_figure_creates_figname_own_subdirectory(tmp_path, monkeypatch):
    """A `figname` containing "/" (e.g. "full/slices_...") saves under
    its own subdirectory of path_to_plots, created on demand - not just
    path_to_plots itself (a bare `path_to_plots.mkdir()` would leave the
    subdirectory missing and the save would raise)."""
    monkeypatch.setattr(viz, "path_to_plots", tmp_path)
    fig = matplotlib.figure.Figure()
    viz.save_figure(fig, "full/subdir_test")
    assert (tmp_path / "full" / "subdir_test.png").exists()


def test_figsize_override(tmp_path, monkeypatch):
    """`figsize=` overrides the (14, 8)/(14, 4 * len(planets)) default -
    e.g. a caller rendering straight to its own print size."""
    monkeypatch.setattr(viz, "path_to_plots", tmp_path)
    fig = viz.plot_field_slices(
        ["planetA"],
        _get_row_data_mean,
        {},
        "cbar label",
        "test_figsize",
        figsize=(7.0, 7.4),
        pres_mode="mean",
    )
    assert tuple(fig.get_size_inches()) == (7.0, 7.4)


def test_figsize_default_unchanged(tmp_path, monkeypatch):
    """No `figsize=` given - still the original default, unaffected by
    the new parameter's presence."""
    monkeypatch.setattr(viz, "path_to_plots", tmp_path)
    fig = viz.plot_field_slices(
        ["planetA"], _get_row_data_mean, {}, "cbar label", "test_default", pres_mode="mean"
    )
    assert tuple(fig.get_size_inches()) == (14.0, 8.0)


def test_lon_titles_override(tmp_path, monkeypatch):
    """`lon_titles=` overrides the default LON_TITLES column titles -
    e.g. a caller with narrower panels that need shorter labels."""
    monkeypatch.setattr(viz, "path_to_plots", tmp_path)
    fig = viz.plot_field_slices(
        ["planetA"],
        _get_row_data_mean,
        {},
        "cbar label",
        "test_lon_titles",
        lon_titles=["AS", "M", "SS", "E"],
        pres_mode="mean",
    )
    row0 = fig.axes[0]
    assert row0.get_title(loc="right") == "AS"


def test_lon_titles_wrong_length_raises():
    with pytest.raises(ValueError, match="lon_titles"):
        viz.plot_field_slices(
            ["planetA"], _get_row_data_mean, {}, "cbar", "x", lon_titles=["only", "two"]
        )


def test_invalid_pres_mode_raises():
    with pytest.raises(ValueError, match="pres_mode"):
        viz.plot_field_slices(["planetA"], _get_row_data_mean, {}, "cbar", "x", pres_mode="bogus")


def test_mean_mode_rejects_per_column_pressure(tmp_path, monkeypatch):
    """A list of 4 arrays passed where pres_mode="mean" expects one
    shared 1D profile - the "did you mean pres_mode='local'?" hint."""
    monkeypatch.setattr(viz, "path_to_plots", tmp_path)
    with pytest.raises(AssertionError, match="pres_mode='local'"):
        viz.plot_field_slices(["planetA"], _get_row_data_local, {}, "cbar", "x", pres_mode="mean")


def test_local_mode_rejects_shared_pressure(tmp_path, monkeypatch):
    """The mirror image - a single 1D profile passed where pres_mode=
    "local" expects one 2D array per column."""
    monkeypatch.setattr(viz, "path_to_plots", tmp_path)
    with pytest.raises(AssertionError, match="pres_mode='mean'"):
        viz.plot_field_slices(
            ["planetA"],
            _get_row_data_mean,
            {},
            "cbar",
            "x",
            pres_mode="local",
            get_contour_coords=_get_contour_coords,
        )


def test_contour_pressure_range_must_overlap_field(tmp_path, monkeypatch):
    """The quench line's own pressure range must overlap the field's
    plotted range - a contour pressure array entirely outside it (e.g.
    from a unit mismatch, or simply the wrong array) would otherwise
    draw silently off-panel with no error at all."""
    monkeypatch.setattr(viz, "path_to_plots", tmp_path)

    def get_contour_coords_off_panel(planet):
        return LATS, PRES_MEAN * 1e10  # far outside the field's own range

    with pytest.raises(AssertionError, match="doesn't overlap"):
        viz.plot_field_slices(
            ["planetA"],
            _get_row_data_mean,
            {},
            "cbar",
            "x",
            get_contour_panels=_get_contour_panels,
            get_contour_coords=get_contour_coords_off_panel,
            pres_mode="mean",
        )


def test_quench_line_has_no_path_effects(tmp_path, monkeypatch):
    """Locks in the no-stroke constraint: the quench-level contour is a
    plain colors="k" line, nothing else - see plot_field_slices()'s own
    docstring. A white-stroke halo technique was tried elsewhere and
    removed as never belonging here."""
    monkeypatch.setattr(viz, "path_to_plots", tmp_path)
    fig = viz.plot_field_slices(
        ["planetA"],
        _get_row_data_mean,
        {},
        "cbar",
        "test_no_stroke",
        get_contour_panels=_get_contour_panels,
        get_contour_coords=_get_contour_coords,
        pres_mode="mean",
    )
    contour_sets = [
        artist
        for ax in fig.axes
        for artist in ax.collections
        if isinstance(artist, matplotlib.contour.QuadContourSet)
    ]
    assert contour_sets, "expected at least one contour artist to check"
    for cs in contour_sets:
        # get_path_effects() returns None, not [], when nothing's ever
        # been set on an artist - both mean "no stroke/halo applied".
        assert not cs.get_path_effects()


def test_contour_color_is_configurable(tmp_path, monkeypatch):
    """contour_color= lets a caller (e.g. one unifying its own
    quench-line convention across figures) draw the quench line in
    something other than the default black, without touching
    plot_field_slices() itself again."""
    monkeypatch.setattr(viz, "path_to_plots", tmp_path)
    fig = viz.plot_field_slices(
        ["planetA"],
        _get_row_data_mean,
        {},
        "cbar",
        "test_contour_color",
        get_contour_panels=_get_contour_panels,
        get_contour_coords=_get_contour_coords,
        contour_color="grey",
        pres_mode="mean",
    )
    contour_sets = [
        artist
        for ax in fig.axes
        for artist in ax.collections
        if isinstance(artist, matplotlib.contour.QuadContourSet)
    ]
    assert contour_sets, "expected at least one contour artist to check"
    grey_rgba = matplotlib.colors.to_rgba("grey")
    for cs in contour_sets:
        colors = cs.get_edgecolor()
        assert all(tuple(c) == grey_rgba for c in colors)


def test_per_line_contour_color_overrides_shared_contour_color(tmp_path, monkeypatch):
    """A (data, levels, ls, color) 4-tuple entry overrides contour_color
    for just that line - e.g. quench_level_contour_panels()'s own
    zahnle_ls/zahnle_qn overlays, drawn in a different color from
    quenchpath's own (plain 3-tuple, contour_color-following) lines
    within the same panel."""
    monkeypatch.setattr(viz, "path_to_plots", tmp_path)

    def _get_contour_panels_mixed(planet):
        base = _contour_panels(PRES_MEAN)  # plain 3-tuples, follow contour_color
        return [[*panels, (panels[0][0], [1], "solid", "tab:red")] for panels in base]

    fig = viz.plot_field_slices(
        ["planetA"],
        _get_row_data_mean,
        {},
        "cbar",
        "test_mixed_colors",
        get_contour_panels=_get_contour_panels_mixed,
        get_contour_coords=_get_contour_coords,
        contour_color="k",
        pres_mode="mean",
    )
    contour_sets = [
        artist
        for ax in fig.axes
        for artist in ax.collections
        if isinstance(artist, matplotlib.contour.QuadContourSet)
    ]
    black_rgba = matplotlib.colors.to_rgba("k")
    red_rgba = matplotlib.colors.to_rgba("tab:red")
    seen = {tuple(c) for cs in contour_sets for c in cs.get_edgecolor()}
    assert black_rgba in seen
    assert red_rgba in seen


def test_contour_legend_plain_pairs_default_to_black_not_none(tmp_path, monkeypatch):
    """A plain (label, linestyle) 2-tuple in contour_legend must draw a
    black legend swatch, the same as the actual (contour_color-following)
    line it labels - regression test for a real bug where building these
    as (label, linestyle, None) triples upstream made Line2D(color=None)
    fall back to matplotlib's color-cycle default (blue) instead."""
    monkeypatch.setattr(viz, "path_to_plots", tmp_path)
    fig = viz.plot_field_slices(
        ["planetA"],
        _get_row_data_mean,
        {},
        "cbar",
        "test_legend_default_color",
        get_contour_panels=_get_contour_panels,
        get_contour_coords=_get_contour_coords,
        contour_legend=[("plain pair", "dashed"), ("overridden", "solid", "tab:red")],
        pres_mode="mean",
    )
    [legend] = fig.legends
    colors = [line.get_color() for line in legend.get_lines()]
    assert colors == ["k", "tab:red"]


def test_hatch_panels_drawn_for_flagged_cells(tmp_path, monkeypatch):
    """get_hatch_panels() draws without error whether or not any cell is
    actually flagged - a caller shouldn't need to special-case an
    all-False mask itself."""
    monkeypatch.setattr(viz, "path_to_plots", tmp_path)

    def _get_hatch_panels_some_flagged(planet):
        mask = np.zeros((len(PRES_MEAN), len(LATS)), dtype=bool)
        mask[0, :] = True
        return [mask.copy() for _ in range(4)]

    def _get_hatch_panels_none_flagged(planet):
        return [np.zeros((len(PRES_MEAN), len(LATS)), dtype=bool) for _ in range(4)]

    for get_hatch_panels, name in [
        (_get_hatch_panels_some_flagged, "test_hatch_some"),
        (_get_hatch_panels_none_flagged, "test_hatch_none"),
    ]:
        fig = viz.plot_field_slices(
            ["planetA"],
            _get_row_data_mean,
            {},
            "cbar",
            name,
            get_hatch_panels=get_hatch_panels,
            pres_mode="mean",
        )
        assert (tmp_path / f"{name}.png").exists()
        assert len(fig.axes) > 0


def test_hatch_panel_shape_mismatch_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(viz, "path_to_plots", tmp_path)

    def _get_hatch_panels_wrong_shape(planet):
        return [np.zeros((1, 1), dtype=bool) for _ in range(4)]

    with pytest.raises(AssertionError, match="hatch mask shape"):
        viz.plot_field_slices(
            ["planetA"],
            _get_row_data_mean,
            {},
            "cbar",
            "test_hatch_bad_shape",
            get_hatch_panels=_get_hatch_panels_wrong_shape,
            pres_mode="mean",
        )


class TestPathwayIndexLegendHandles:
    """pathway_index_legend_handles() directly - in particular the
    empty-eqns case (a "no path found" pathway index - see
    quenchpath.analysis.path_index.eqn_sequence_of()'s own empty-tuple
    return - now real, not just theoretical, once a CO-to-CH4 run's
    per-location NetworkXNoPath rate stopped being ~0% - see GitHub
    quenchpath-apps issue #15)."""

    def test_labels_a_real_pathway_by_its_eqn_chain(self):
        path_index_map = {
            1: {"eqns": ["A+B↔C"], "directions": ["forward"], "nodes": ["A", "C"]},
        }
        [handle] = viz.pathway_index_legend_handles(path_index_map)
        assert handle.get_label() == "1: A+B→C"

    def test_empty_eqns_gets_a_placeholder_not_a_blank_label(self):
        path_index_map = {
            1: {"eqns": ["A+B↔C"], "directions": ["forward"], "nodes": ["A", "C"]},
            2: {"eqns": [], "directions": [], "nodes": []},
        }
        handles = {
            h.get_label().split(":")[0]: h.get_label()
            for h in viz.pathway_index_legend_handles(path_index_map)
        }
        assert handles["2"] == "2: (no path found)"


class TestQuenchLevelContourPanels:
    """quench_level_contour_panels() directly - the (data, levels, ls[,
    color]) tuple/legend-entry builder, not plot_field_slices() itself."""

    def _ratio(self):
        return np.ones((len(PRES_MEAN), len(LATS), 4))  # 4 longitudes

    def test_no_ratios_given_returns_none(self):
        assert viz.quench_level_contour_panels([0, 1]) == (None, None)

    def test_zahnle_lines_get_their_own_color_quenchpath_lines_dont(self):
        contour_panels, legend = viz.quench_level_contour_panels(
            [0],
            ratio_net_ls=self._ratio(),
            ratio_zahnle_ls=self._ratio(),
            ratio_zahnle_qn=self._ratio(),
        )
        (net_ls_entry, zahnle_ls_entry, zahnle_qn_entry) = contour_panels[0]
        assert len(net_ls_entry) == 3  # plain 3-tuple - follows the shared contour_color
        assert net_ls_entry[2] == "dashed"
        assert zahnle_ls_entry[2:] == ("dashed", "tab:red")
        assert zahnle_qn_entry[2:] == ("solid", "tab:red")

        assert legend == [
            (r"$\tau^{net}_{CH_4-to-CO}/\tau^{LS,dom}_{adv}=1$", "dashed"),
            (r"Zahnle & Marley (2014) $t_{CO}/\tau^{LS,dom}_{adv}=1$", "dashed", "tab:red"),
            (r"Zahnle & Marley (2014) $t_{CO}/\tau^{QN}_{adv}=1$", "solid", "tab:red"),
        ]

    def test_zahnle_color_is_configurable(self):
        _contour_panels, legend = viz.quench_level_contour_panels(
            [0], ratio_zahnle_ls=self._ratio(), zahnle_color="tab:blue"
        )
        assert legend[0][2] == "tab:blue"

    def test_conversion_co_ch4_flips_the_default_label_not_the_zahnle_one(self):
        # Regression test: quench_level_contour_panels() used to hardcode
        # "CH_4-to-CO" regardless of conversion - a caller plotting a
        # CO_CH4 ratio without overriding every *_label kwarg got a
        # mislabeled legend. Zahnle & Marley's own label is unaffected -
        # their fit is always framed as t_CO, not conversion-dependent.
        _contour_panels, legend = viz.quench_level_contour_panels(
            [0], ratio_net_ls=self._ratio(), ratio_zahnle_ls=self._ratio(), conversion="co_ch4"
        )
        assert legend[0] == (r"$\tau^{net}_{CO-to-CH_4}/\tau^{LS,dom}_{adv}=1$", "dashed")
        assert legend[1][0] == r"Zahnle & Marley (2014) $t_{CO}/\tau^{LS,dom}_{adv}=1$"

    def test_conversion_accepts_a_speciesconversion_value_not_just_a_plain_string(self):
        # conversion is duck-typed (viz.py stays iris/config-free, see its
        # own docstring) - a real SpeciesConversion.CO_CH4 (a str subclass)
        # must compare equal to "co_ch4" the same way the plain string does.
        from quenchpath.config import SpeciesConversion

        _contour_panels, legend = viz.quench_level_contour_panels(
            [0], ratio_net_ls=self._ratio(), conversion=SpeciesConversion.CO_CH4
        )
        assert legend == [(r"$\tau^{net}_{CO-to-CH_4}/\tau^{LS,dom}_{adv}=1$", "dashed")]

    def test_ratio_label_override_wins_regardless_of_conversion(self):
        _contour_panels, legend = viz.quench_level_contour_panels(
            [0],
            ratio_net_ls=self._ratio(),
            ratio_net_ls_label="custom label",
            conversion="co_ch4",
        )
        assert legend == [("custom label", "dashed")]

    def test_region_bounds_draws_two_levels_per_entry(self):
        contour_panels, legend = viz.quench_level_contour_panels(
            [0],
            region_bounds=[(self._ratio(), 0.5, 2.0, "quench region", "dotted", "tab:purple")],
        )
        (entry,) = contour_panels[0]
        data, levels, ls, color = entry
        np.testing.assert_array_equal(data, self._ratio()[..., 0])
        assert levels == [0.5, 2.0]
        assert (ls, color) == ("dotted", "tab:purple")
        assert legend == [("quench region", "dotted", "tab:purple")]

    def test_region_bounds_alone_is_enough_to_avoid_the_none_none_shortcut(self):
        # No ratio_* given at all - region_bounds on its own must still
        # produce real panels/legend, not the "nothing to draw" (None,
        # None) test_no_ratios_given_returns_none() pins down.
        contour_panels, legend = viz.quench_level_contour_panels(
            [0], region_bounds=[(self._ratio(), 0.1, 10.0, "region", "dashdot", None)]
        )
        assert contour_panels is not None
        assert legend is not None

    def test_region_bounds_can_combine_with_a_fixed_ratio_line(self):
        contour_panels, legend = viz.quench_level_contour_panels(
            [0],
            ratio_net_qn=self._ratio(),
            region_bounds=[(self._ratio(), 0.5, 2.0, "quench region", "dotted", "tab:purple")],
        )
        net_qn_entry, region_entry = contour_panels[0]
        assert net_qn_entry[1] == [1]  # the fixed ratio=1 crossing, unaffected
        assert region_entry[1] == [0.5, 2.0]
        assert len(legend) == 2


class TestSlicePanelCoords:
    """slice_panel_coords() directly - the coordinate/shape-validation
    helper plot_field_slices() (and any caller building its own grid
    instead, e.g. a figure needing a per-row colorbar) shares."""

    def test_mean_mode(self):
        panel_data = _field_data()
        x_cols, y_cols, pres_min, pres_max = viz.slice_panel_coords(
            LATS, PRES_MEAN, "mean", panel_data
        )
        assert all(np.array_equal(x, LATS) for x in x_cols)
        assert all(np.array_equal(y, PRES_MEAN) for y in y_cols)
        assert pres_min == PRES_MEAN.min()
        assert pres_max == PRES_MEAN.max()

    def test_local_mode(self):
        pres_cols = _local_pres_cols()
        panel_data = _field_data()
        x_cols, y_cols, pres_min, pres_max = viz.slice_panel_coords(
            LATS, pres_cols, "local", panel_data
        )
        for col, (x, y) in enumerate(zip(x_cols, y_cols)):
            assert y.shape == pres_cols[col].shape
            assert x.shape == y.shape
        assert pres_min == min(p.min() for p in pres_cols)
        assert pres_max == max(p.max() for p in pres_cols)

    def test_invalid_pres_mode_raises(self):
        with pytest.raises(AssertionError, match="pres_mode"):
            viz.slice_panel_coords(LATS, PRES_MEAN, "bogus", _field_data())
