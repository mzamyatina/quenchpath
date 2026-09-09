"""What: regenerate docs/figures/example_hd189733b_slices.png - the
CH4-to-CO pathway-index figure shown in the README, built from the
bundled inputs/example_data/ and nothing else.

Why: the README should show what quenchpath actually produces, and that
image has to live in a tracked directory (outputs/ is gitignored). This
regenerates it whenever the plotting changes, rather than leaving a
stale PNG committed with no way to reproduce it.

Why a script rather than a notebook export: the figure is the same one
notebooks/plt_slices.ipynb produces from the same bundled data - this
script just reproduces those calls headlessly, so refreshing the README
image needs no Jupyter and no manual "run the notebook, copy the file"
step. Keep the two in step: if the notebook's overlays change, change
them here too.

Why no arguments, unlike scripts/generate_example_data.py: this figure
is definitionally the bundled example (hd189733b, kinetics, SAMPLE,
CH4-to-CO), so there is nothing to parameterise. Point
generate_example_data.py at a different run if the bundled example
itself should change.

Usage
-----
    python scripts/generate_example_figure.py

Needs only what already ships with the repo - no inputs/raw_runs/ and
no completed pipeline run of your own.
"""

import shutil
import warnings

import matplotlib
import numpy as np
import yaml

from quenchpath import io
from quenchpath.analysis.quench_level import (
    quench_level,
    quench_level_zahnle_ls,
    quench_level_zahnle_qn,
)
from quenchpath.analysis.zahnle_marley_2014 import zahnle_marley_2014_invalid_mask
from quenchpath.config import GridResolution, RunConfig, SpeciesConversion
from quenchpath.mypaths import (
    path_to_example_data,
    path_to_planets_yaml,
    path_to_plots,
    path_to_repo,
)
from quenchpath.viz import (
    pathway_index_contourf_kw,
    pathway_index_legend_handles,
    plot_field_slices,
    quench_level_contour_panels,
    representative_lon_indices,
)

# Headless: this script is run from a terminal, not a notebook, and
# never needs an interactive window.
matplotlib.use("Agg")

warnings.filterwarnings("ignore", module="iris")

# The bundled example - see this module's own docstring for why these
# are fixed rather than arguments.
CFG = RunConfig(planet="hd189733b", exp="kin")
GRID_RESOLUTION = GridResolution.SAMPLE
CONVERSION = SpeciesConversion.CH4_CO

FIGURE_DIR = path_to_repo / "docs" / "figures"
FIGURE_NAME = "example_hd189733b_slices"

_SPECIES_TEX = {"CH4": r"$CH_{4}$", "CO": "CO"}


def main():
    with open(path_to_planets_yaml) as f:
        planet_names = {k: v["tex"] for k, v in yaml.safe_load(f).items()}
    # Plain display name, no metallicity: row 0's own loc="left" row
    # title shares the first panel's axes with the loc="right"
    # "Antistellar" column title, and a longer string collides with it
    # (same reason notebooks/plt_slices.ipynb keeps this short).
    run_title = planet_names.get(CFG.planet, CFG.planet)

    (
        temp_cube,
        pres_cube,
        path_index_cube,
        _net_rate_inv_cube,
        _max_rate_inv_cube,
        path_index_map,
    ) = io.load_merged_paths(
        CFG, GRID_RESOLUTION, conversion=CONVERSION, base_dir=path_to_example_data
    )
    n_indices = len(path_index_map)
    print(
        f"{run_title} ([M/H]={CFG.mdh})/{CFG.exp} "
        f"({GRID_RESOLUTION.value} resolution, {CONVERSION.label}): "
        f"{path_index_cube.data.size} locations, {n_indices} known pathways"
    )

    lon_idx = representative_lon_indices(path_index_cube.coord("longitude").points)
    # hd189733b is solar ([M/H]=0 -> m=1), one of Zahnle & Marley
    # (2014)'s two calibrated values, so nothing here is extrapolated.
    zahnle_m = 10 ** float(CFG.mdh)

    # Spelled out rather than built in a loop: each name is one of
    # quench_level_contour_panels()' own keyword arguments, and keeping
    # them greppable matters more here than saving four lines.
    ratio_net_ls = quench_level(
        CFG, GRID_RESOLUTION, "net", "ls", conversion=CONVERSION, base_dir=path_to_example_data
    ).data
    ratio_net_qn = quench_level(
        CFG, GRID_RESOLUTION, "net", "qn", conversion=CONVERSION, base_dir=path_to_example_data
    ).data
    ratio_rls_ls = quench_level(
        CFG, GRID_RESOLUTION, "rls", "ls", conversion=CONVERSION, base_dir=path_to_example_data
    ).data
    ratio_rls_qn = quench_level(
        CFG, GRID_RESOLUTION, "rls", "qn", conversion=CONVERSION, base_dir=path_to_example_data
    ).data
    ratio_zahnle_ls = quench_level_zahnle_ls(
        CFG, pres_cube, temp_cube, m=zahnle_m, base_dir=path_to_example_data
    ).data
    ratio_zahnle_qn = quench_level_zahnle_qn(
        CFG,
        pres_cube,
        temp_cube,
        m=zahnle_m,
        conversion=CONVERSION,
        base_dir=path_to_example_data,
    ).data

    # A fixed 1 e-folding band around ratio_net_qn's own ratio=1 line -
    # low/high are exp(-bound)/exp(bound), drawn on the *linear* ratio
    # array so the contours land at the right value.
    region_bounds = [
        (
            ratio_net_qn,
            np.exp(-1.0),
            np.exp(1.0),
            "CH4 quench region (1 $e$-fold)",
            "dotted",
            "tab:purple",
        )
    ]
    contour_panels, legend = quench_level_contour_panels(
        lon_idx,
        ratio_net_ls=ratio_net_ls,
        ratio_net_qn=ratio_net_qn,
        ratio_rls_ls=ratio_rls_ls,
        ratio_rls_qn=ratio_rls_qn,
        ratio_zahnle_ls=ratio_zahnle_ls,
        ratio_zahnle_qn=ratio_zahnle_qn,
        conversion=CONVERSION,
        region_bounds=region_bounds,
    )
    # Where Zahnle & Marley's calibration envelope doesn't cover this
    # location (T >= 2200 K), on the field's own true local (level,
    # lat) grid - matching pathway_index_row()'s panel_data slicing.
    invalid_mask = zahnle_marley_2014_invalid_mask(temp_cube).data
    hatch_panels = [invalid_mask[..., lo] for lo in lon_idx]

    def pathway_index_row(_cfg):
        lats = path_index_cube.coord("latitude").points
        panel_data = [path_index_cube.data[..., lo] for lo in lon_idx]
        # pres_mode="local" - each column's own true local pressure
        # (level, lat), reused as the contour lines' own default
        # coordinate too.
        pres_cols = [pres_cube.data[..., lo] for lo in lon_idx]
        return lats, pres_cols, panel_data

    kw, cbar_ticks = pathway_index_contourf_kw(n_indices)
    plot_field_slices(
        [CFG],
        pathway_index_row,
        kw,
        f"{_SPECIES_TEX[CONVERSION.source]}-to-{_SPECIES_TEX[CONVERSION.target]} pathway index",
        FIGURE_NAME,
        get_contour_panels=lambda _c: contour_panels,
        get_hatch_panels=lambda _c: hatch_panels,
        contour_legend=legend,
        extra_legend_handles=pathway_index_legend_handles(path_index_map),
        legend_title="Pathway index, ranked by dominance (1 = most dominant, 2 = next, etc.)",
        cbar_ticks=cbar_ticks,
        ptype="contourf",
        planet_names={CFG: run_title},
        pres_mode="local",
    )

    # plot_field_slices() always saves through quenchpath.viz.save_figure(),
    # which writes into the gitignored outputs/ tree - copy it into the
    # tracked docs/ location the README points at. Same pattern as
    # scripts/generate_example_data.py's own _copy_into_example_data().
    src = path_to_plots / f"{FIGURE_NAME}.png"
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    dst = FIGURE_DIR / f"{FIGURE_NAME}.png"
    shutil.copy2(src, dst)
    print(f"Copied {src} -> {dst}")


if __name__ == "__main__":
    main()
