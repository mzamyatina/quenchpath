"""Path constants used throughout the pipeline.

Two categories, deliberately kept apart:

- **Package data** - the same for every user, versioned alongside the
  code: the raw chemical-network source files, the parsed network, and
  the bundled example dataset. Always resolved relative to the package's
  own checkout.
- **User data** - this user's own runs: raw model input, per-run
  metadata, and everything the pipeline generates. Resolved relative to
  `path_to_data_root`, which defaults to the same checkout but can be
  pointed elsewhere with the `QUENCHPATH_DATA_ROOT` environment
  variable.

Why the split: working inside a clone (the default) behaves exactly as
it always has, but an *installed* quenchpath can be driven from a
separate project directory - so a user's own planets and outputs never
have to live inside the package's own checkout. See
`path_to_data_root`'s own comment below.
"""

import os
from pathlib import Path

# repo root
path_to_repo = Path(__file__).resolve().parents[2]


# %% Package data - ships with the code, identical for every user.

# What: genuine input - read in, never written by any quenchpath stage.
# Why grouped under inputs/: so the raw chemical-network source files
# and the bundled example sort as one entry rather than being scattered
# among src/, tests/, etc.
path_to_inputs = path_to_repo / "inputs"

# venot2019 chemical network's raw source files
path_to_raw_networks = path_to_inputs / "raw_networks" / "venot2019"

# What: small, tracked (not gitignored) example dataset - just enough
# real output for notebooks/plt_slices.ipynb and plt_graphs.ipynb to
# run out of the box without a completed pipeline run first. Mirrors
# path_to_results/path_to_intermediates' own sub-layout underneath
# (e.g. example_data/ch4_co_paths/sample/... matches
# path_to_results/ch4_co_paths/sample/...).
# Why mirror that layout: so quenchpath.io's loaders can point at
# either one via their own base_dir parameter.
path_to_example_data = path_to_inputs / "example_data"

# What: venot2019 network parsed from path_to_raw_networks
# (read-polynomials/read-network) - deterministic and small enough to
# track. Kept apart from both inputs/ and outputs/, one level up from
# inputs/'s own gitignore boundary.
# Why kept apart from both: neither genuine input (it's generated) nor
# a stage's per-run output (it's network-wide, done once - unlike
# everything under path_to_results below, keyed by planet/exp/mdh/tK).
# Why one level up from inputs/: so it doesn't need fighting outputs/'s
# blanket ignore to stay tracked.
# Why package data despite being generated: it's a property of the
# network, not of any user's runs - every user's copy is identical.
path_to_networks_processed = path_to_repo / "networks_processed" / "venot2019"


# %% User data - this user's own runs, redirectable.

# What: the root everything below is resolved against. Defaults to the
# package's own checkout, so working inside a clone is unchanged; set
# QUENCHPATH_DATA_ROOT to a different directory to drive an installed
# quenchpath from a separate project (its own inputs/ and outputs/ are
# then read and written there, not inside the package's checkout).
# Why an environment variable rather than a CLI flag: every stage and
# both notebooks need it, and it's a property of the working setup
# rather than of any single command - the same reason inputs/raw_runs
# is a per-machine symlink rather than an argument.
DATA_ROOT_ENV_VAR = "QUENCHPATH_DATA_ROOT"
path_to_data_root = Path(os.environ.get(DATA_ROOT_ENV_VAR, path_to_repo)).expanduser().resolve()

# symlink to raw model input (gitignored), "{planet}_{exp}_mdh{mdh}_tK{tK}.nc" per run
path_to_raw_runs = path_to_data_root / "inputs" / "raw_runs"

# Per-planet/universal physical constants (aeolus.const.init_const()'s
# own JSON format) plus planets.yaml (planet display names, e.g.
# hatp11b -> "HAT-P-11b" - a user-defined shorthand, quenchpath only
# consumes it, see quenchpath.viz's own planet_names parameter).
# Why user data, not package data: which planets exist, and the
# constants describing them, are properties of a user's own runs.
path_to_run_metadata = path_to_data_root / "inputs" / "run_metadata"
path_to_planets_yaml = path_to_run_metadata / "planets.yaml"

# regenerable per-stage intermediates (stages 03-05) vs small final results
# (merge-paths, stage 07) - see quenchpath.io for what lives where
path_to_intermediates = path_to_data_root / "outputs" / "intermediates"
path_to_results = path_to_data_root / "outputs" / "results"
path_to_plots = path_to_results / "plots"
