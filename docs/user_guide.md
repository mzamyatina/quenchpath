# User guide

Detail behind the quickstart in the top-level README: external data
layout, the full pipeline command reference, notebooks, the bundled
example dataset, plotting, and package layout.

## External data

- Not bundled: quenchpath is agnostic of where raw model input comes
  from.
- Stage 03 (`convert`) reads one merged raw run per planet/exp/mdh/tK
  from `quenchpath.mypaths.path_to_raw_runs` (`inputs/raw_runs`, files
  named `{planet}_{exp}_mdh{mdh}_tK{tK}.nc` - see
  `quenchpath.io.raw_run_filename()`).
- `inputs/raw_runs` is a symlink, gitignored - set it up once per
  machine (`ln -s /wherever/your/data/lives inputs/raw_runs`) before
  running `convert`/`run-pipeline-*` for the first time.
- Bundled and tracked: `inputs/run_metadata/` (per-planet/universal
  physical constants, `aeolus.const.init_const()`'s JSON format, plus
  `planets.yaml` for planet display names).
- Also bundled: the Venot et al. (2019) reduced chemical network's raw
  source files (`inputs/raw_networks/venot2019/`). Stages 01/02
  (`read-polynomials`/`read-network`) parse those once, network-wide,
  into `venot2019_network.json`/`nasa_polynomials_7_term.json` under
  `quenchpath.mypaths.path_to_networks_processed`
  (`networks_processed/venot2019/` - generated but small enough to
  track).
  Why kept out of both `inputs/` and the gitignored `outputs/`: it's
  neither raw input nor per-run output.

### Working outside the quenchpath checkout

By default every path resolves relative to the quenchpath checkout, so
working inside a clone needs no setup beyond the `inputs/raw_runs`
symlink above. To drive an *installed* quenchpath from a separate
project directory instead - keeping your own planets and outputs out of
the package's own checkout - set `QUENCHPATH_DATA_ROOT`:

```shell
export QUENCHPATH_DATA_ROOT=/path/to/my-project
```

That redirects the **user data** paths only: `inputs/raw_runs`,
`inputs/run_metadata`, and everything generated under `outputs/` are
then read and written under that root (so it needs its own
`inputs/run_metadata/<planet>_mdh<mdh>.json` and `inputs/raw_runs`
symlink). **Package data** - `inputs/raw_networks/`,
`inputs/example_data/`, and `networks_processed/` - always stays with
the package, since it's identical for every user. See
`quenchpath.mypaths`' own module docstring for the full split.

## Running the pipeline

One CLI subcommand per stage, `--planet`/`--exp`/`--mdh`/`--tK` select
a run:

```shell
quenchpath --help
quenchpath read-polynomials                                                        # stage 01, network-wide
quenchpath read-network                                                            # stage 02, network-wide
quenchpath convert --planet hd189733b --exp kin                                    # stage 03
quenchpath calc-rates --planet hd189733b --exp kin                                 # stage 04
quenchpath calc-paths-serial --planet hd189733b --exp kin --grid-resolution sample # stage 05
quenchpath calc-paths-parallel --planet hd189733b --exp kin --grid-resolution full --max-workers 8   # stage 05, multiprocessing
quenchpath merge-paths --planet hd189733b --exp kin --grid-resolution full         # stage 06, merges calc-paths-serial/-parallel's JSON into one .nc
quenchpath calc-taus --planet hd189733b --exp kin                                  # stage 07 (calc only)
quenchpath run-pipeline-serial --planet hd189733b --exp kin                        # 03->04->05->06->07 chained, calc-paths-serial - SAMPLE testing
quenchpath run-pipeline-parallel --planet hd189733b --exp kin --grid-resolution full --max-workers 8   # same, calc-paths-parallel instead - the FULL-resolution command
```

- `--exp`: `kin` (kinetics) or `equ` (equilibrium). `--mdh`/`--tK`
  default to `"0"`/`"1"`.
- `--grid-resolution`: `sample` (~4896 locations/planet, every 4th
  level - see `quenchpath.config.GridResolution`) or `full` (every
  native UM grid point, n_levels x 90 x 144).
- `--conversion`: `ch4_co` (default) or `co_ch4` - which species
  conversion's pathway to search for. See
  `quenchpath.config.SpeciesConversion` and
  `docs/uni_vs_bidirectional_pathway_search.md`.
- `calc-paths-serial` saves one JSON Lines file per level. `calc-paths-parallel`
  (`quenchpath.stages.stage_05_paths_parallel`) is a separate,
  multiprocessing implementation of the same stage - one vertical
  level per OS process, byte-for-byte identical output to `calc-paths-serial`;
  `--max-workers` defaults to every available core.
  Why prefer it at FULL (SAMPLE is already fast serially): can turn a
  ~20+ min/planet serial run into a couple of minutes.
- `merge-paths` reads a completed `calc-paths-serial`/`calc-paths-parallel`
  run's JSON Lines back in one pass and merges it into one
  self-describing `.nc` (temp, true per-location pressure, pathway
  index, bottleneck-segment timescale, gridded on
  level_height/latitude/longitude).
  Why level_height, not pressure: pressure isn't a shared axis across
  locations (see issue #11).

**Measured wall time**, one planet through the full stage 03-07 matrix
(both resolutions, both conversions, kin+equ), 16-core Apple M4 Max,
`--max-workers 10`:

| Step | Time |
|---|---|
| convert | ~2s |
| calc-rates | ~70-73s |
| calc-paths-parallel, SAMPLE (per conversion) | ~30s |
| merge-paths/calc-taus, SAMPLE (per conversion) | ~1-2s each |
| calc-paths-parallel, FULL, ch4_co | ~185s (~3 min) |
| calc-paths-parallel, FULL, co_ch4 | ~330-395s (~5.5-6.5 min) |
| merge-paths, FULL | ~10-24s |
| calc-taus, FULL/equ (per conversion) | ~1-2s each |
| **Total** | **~11.5-12.6 min/planet** |

`co_ch4`'s FULL search runs ~1.8-2x slower than `ch4_co`'s.
Why: `build_bidirect_graph()` admits both directions of every reaction
instead of one - a genuinely larger graph to search, not a regression.

**Files created per run**, by stage (SAMPLE unless noted; see
`quenchpath.io` for exact filenames):

| Stage | Writes | Files/run |
|---|---|---|
| 01 read-polynomials | NASA-7 polynomial JSON | 1 (not per-run) |
| 02 read-network | reaction network JSON | 1 (not per-run) |
| 03 convert | converted number-density cubes | 1 |
| 04 calc-rates | k_eq/rate_f/rate_r/gibbs cubes, 6 x 180 reactions | 9 (chunked, 120 cubes/file) |
| 05 calc-paths-serial/-parallel (sample) | per-location pathway JSON Lines | ~17 (1/level, every 4th) |
| 05 calc-paths-serial/-parallel (full) | per-location pathway JSON Lines | n_levels (1/level) |
| 06 merge-paths | merged pathway-index cubes | 1 |
| 07 calc-taus (kin) | tau_adv_ls + `tau_<conversion>_rls`/`_net` cubes | 3 (tau_adv_ls is shared/resolution-independent, harmlessly rewritten every call) |
| 07 calc-taus (equ) | tau_adv_ls + tau_adv_qn cubes | 2 (same tau_adv_ls sharing) |

Every subcommand is a thin wrapper around a pure/testable function in
`quenchpath.chemistry.*`/`quenchpath.stages.*` - see those modules'
docstrings for the numerics, `quenchpath.io`/`quenchpath.mypaths` for
filename/directory conventions.

### Notebooks

- `notebooks/NN_*.ipynb`: source of truth for interactive/exploratory
  work, one per pipeline stage - same stage functions the CLI calls.
  Use the CLI for batch/multi-planet runs.
- `notebooks/plt_graphs.ipynb`: not a pipeline stage - a
  step-by-step exercise of the core per-location graph functions
  (`quenchpath.stages.stage_05_paths_serial.load_eqns_with_rates`,
  `quenchpath.graph.shortest_path.build_unidirect_graph`/
  `build_bidirect_graph`/`prune_to_carbon_subgraph`/`find_pathway`) -
  plots one location's full reaction network, carbon-only subgraph,
  then its pathway (either conversion, via a `conversion` variable).
  Why no completed `calc-paths-serial`/`calc-paths-parallel` run is
  needed: works from stage 03's converted output and stage 04's saved
  rates directly.
- `notebooks/plt_slices.ipynb`: see "Plotting" below.

### Example data

- `inputs/example_data/`: tracked (not gitignored), just enough real
  output for `plt_slices.ipynb`/`plt_graphs.ipynb` to run
  out of the box with no completed pipeline run - both default to
  `use_bundled_example = True` (CH4_CO only; set to `False` for a real
  run, any planet/conversion, at full resolution).
- Why not just a subset of a real `outputs/` run: that's multiple GB
  per planet. Instead: `plt_graphs.ipynb`'s example is a purpose-built
  ~30KB JSON snapshot (one location's own rates/number densities);
  `plt_slices.ipynb`'s is a genuine SAMPLE-resolution `merge-paths`
  output plus its 4 tau files (~530KB total). `tau_adv_qn` - which has
  no genuine SAMPLE variant in the real pipeline (always computed at
  full resolution) - is explicitly subset down for this bundle (see
  `scripts/generate_example_data.py`'s own `_save_subsampled_tau_adv()`).
- `scripts/generate_example_data.py` builds this dataset (a real
  from-scratch SAMPLE-resolution pipeline run - except tau_adv_ls/
  tau_adv_qn, computed at the only resolution they actually have
  (full native grid) and then explicitly subset down to SAMPLE for
  the bundle - copied into `inputs/example_data/`) - a maintainer-only
  tool, not part of the `pytest` suite or a CLI subcommand. Re-run
  whenever the bundled example needs regenerating.
- `scripts/generate_example_figure.py` regenerates
  `docs/figures/example_hd189733b_slices.png`, the figure the README
  shows, from that same bundled dataset - byte-identical to what
  `notebooks/plt_slices.ipynb` produces, just rendered headlessly so
  refreshing the README image needs no Jupyter. Also maintainer-only;
  re-run whenever the plotting changes.

## Plotting

- `quenchpath.viz` (install the `viz` extra: `pip install -e .[viz]`
  for matplotlib/pyyaml): shared, data-agnostic plotting primitives
  (slice-plot grid, pathway-index color palette, quench-level contour
  overlay) for anyone building figures against saved output.
- `plot_field_slices()`'s `pres_mode` picks which pressure the color
  field plots against: `"local"` (default) - each column's own true 2D
  `(level, latitude)` pressure; `"mean"` - one 1D profile shared by all
  4 longitude columns in a row (needed only for the `tau_adv_ls_*`
  family, a day-side spatial mean with no per-location pressure of its
  own, up to ~9% off true local pressure). The quench-level contour
  overlay defaults to the same local pressure as the color field
  (`get_contour_coords` omitted) - every quench-level ratio
  `quenchpath.analysis.quench_level`/`quench_region` computes shares
  its color field's exact grid (see `quench_level_contour_panels()`'s
  own docstring), so this is the common case, not a per-quantity
  exception. Pass an explicit `get_contour_coords` only for a contour
  that genuinely needs a different pressure convention from the field
  it's drawn over. Shapes and pressure ranges are validated - see
  `slice_panel_coords()`/`check_contour_coords()`.
- `notebooks/plt_slices.ipynb`: edit its top-of-file
  `RUNS`/`grid_resolution`/`conversion`/`overlay_*` variables and
  re-run. Builds one pathway-index slice figure per `RunConfig` in
  `RUNS` (all 8 real planet/mdh/tK configs by default), with
  `tau_adv_ls`/`tau_adv_qn` quench-level contours and a Zahnle & Marley
  (2014) overlay independently toggleable. Figures save to
  `outputs/results/plots/slices/<sample|full>/slices_<planet>_
  mdh<mdh>_tK<tK>_<resolution>_<source>_<target>.png` (`save_figure()`
  creates the `slices/<sample|full>` subdirectory on demand).
  Why the Zahnle overlay is silently skipped (not excluded from
  `RUNS`) for a non-solar/non-3x-solar metallicity: Zahnle & Marley's
  own fit isn't calibrated for it.
- `notebooks/plt_graphs.ipynb` (see "Notebooks" above): saves 3 figures
  per run - `outputs/results/plots/graphs/graph_<planet>_<source>_
  <target>_<full|carbon|shortest>.png` (full reaction network,
  carbon-only subgraph, found shortest path).
  Why overwritten on re-run regardless of which location was plotted:
  the filename is keyed by (planet, conversion) only, not location -
  one canonical set per (planet, conversion).

## Package layout

- `quenchpath.config` - `RunConfig` (planet/exp/mdh/tK),
  `GridResolution`, `SpeciesConversion` (CH4-to-CO or CO-to-CH4),
  passed through every stage instead of module-level globals.
- `quenchpath.chemistry` - everything specific to the Venot et al.
  (2019) network:
  - `.network` - the `VENOT2019` species-list dict.
  - `.polynomials` / `.reactions` - parse/save the NASA polynomials
    and reaction network (once, network-wide).
  - `.formulas` - rate-coefficient/thermodynamics formulas (Kooij,
    Troe, Gibbs energy, equilibrium constant), used by
    `stages.stage_04_rates`.
- `quenchpath.graph.shortest_path` - pure, iris-free graph
  construction + shortest-path search for one location's pathway -
  stage 05's core algorithm.
- `quenchpath.stages` - `.stage_03_convert` / `.stage_04_rates` /
  `.stage_05_paths_serial` / `.stage_05_paths_parallel` /
  `.stage_06_merge_paths` / `.stage_07_taus`, each a pure numerical
  core plus one `RunConfig`-based I/O driver.
- `quenchpath.analysis` - science-output modules downstream of the
  pipeline stages:
  - `.path_index` - pathway identity/ranking (`eqn_sequence_of()`,
    `assign_path_indices_from_locations()`,
    `path_index_map_from_locations()`, plus
    `assign_path_indices_from_maps()` for pooling multiple
    already-merged datasets).
  - `.quench_level` - the "quench level" (tau_chem/tau_adv=1)
    contour's data acquisition (`quench_level()`), plus
    `subset_to_grid_of()`, the shared grid-alignment helper.
  - `.quench_region` - `quench_region()`, a continuous `ln(ratio)`
    alternative to `quench_level()`'s own sharp ratio=1 crossing.
  - `.k_test_over_k_eq` - the system-level net-flux-direction check
    motivating `docs/uni_vs_bidirectional_pathway_search.md`.
  - `.zahnle_marley_2014` - Zahnle & Marley (2014)'s CO<->CH4
    quench-timescale fit, for comparison against quenchpath's own
    pathway-derived timescales.
- `quenchpath.io` - central filename/load/save conventions, keyed by
  `RunConfig`.
- `quenchpath.mypaths` - path constants, split into package data
  (`inputs/raw_networks/`, `inputs/example_data/`,
  `networks_processed/` - identical for every user, always resolved
  against the package's own checkout) and user data
  (`inputs/raw_runs/`, `inputs/run_metadata/`, and the
  pipeline-generated, gitignored `outputs/` - resolved against
  `path_to_data_root`, overridable via `QUENCHPATH_DATA_ROOT`, see
  "Working outside the quenchpath checkout" above).
- `quenchpath.cli` - the `quenchpath` console script described above.
