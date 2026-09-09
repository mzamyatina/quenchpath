"""Command-line interface for the quenchpath pipeline.

Wraps the stage functions (quenchpath.chemistry.*, quenchpath.stages.*)
in `click` subcommands - an additional invocation path for batch/multi-
planet runs, alongside the notebooks/pipeline/ scripts.
"""

import warnings

import click

from quenchpath import io
from quenchpath.chemistry.network import VENOT2019
from quenchpath.chemistry.polynomials import load_and_save_nasa_polynomials
from quenchpath.chemistry.reactions import load_and_save_venot2019_network
from quenchpath.config import GridResolution, RunConfig, SpeciesConversion
from quenchpath.mypaths import path_to_networks_processed, path_to_raw_networks
from quenchpath.stages.stage_03_convert import convert_mole_fraction_to_number_density
from quenchpath.stages.stage_04_rates import calc_rates
from quenchpath.stages.stage_05_paths_parallel import calc_paths_parallel
from quenchpath.stages.stage_05_paths_serial import calc_paths_serial
from quenchpath.stages.stage_06_merge_paths import merge_and_save_paths
from quenchpath.stages.stage_07_taus import (
    calc_and_save_adv_ls,
    calc_and_save_adv_qn,
    calc_and_save_chem_timescale,
)

_EXP_CHOICE = click.Choice(["kin", "equ"])
_GRID_RESOLUTION_CHOICE = click.Choice([g.value for g in GridResolution])
_SPECIES_CONVERSION_CHOICE = click.Choice([d.value for d in SpeciesConversion])


def _run_config_options(f):
    """Shared --planet/--exp/--mdh/--tK options for every per-run command."""
    f = click.option("--tK", "tK", default="1", show_default=True, help="Temperature scaling.")(f)
    f = click.option("--mdh", default="0", show_default=True, help="Metallicity ([M/H]).")(f)
    f = click.option(
        "--exp", type=_EXP_CHOICE, default="kin", show_default=True, help="Experiment."
    )(f)
    f = click.option(
        "--planet", required=True, help="Planet name (matches the raw-run filenames' <planet>)."
    )(f)
    return f


def _grid_resolution_option(help_text):
    """--grid-resolution option shared by calc-paths/calc-paths-parallel/
    merge-paths/run-pipeline-serial/run-pipeline-parallel (only their
    help text differs)."""
    return click.option(
        "--grid-resolution",
        type=_GRID_RESOLUTION_CHOICE,
        default=GridResolution.SAMPLE.value,
        show_default=True,
        help=help_text,
    )


def _max_workers_option(f):
    """--max-workers option shared by calc-paths-parallel/
    run-pipeline-parallel."""
    return click.option(
        "--max-workers",
        type=int,
        default=None,
        help="OS processes to use - defaults to every available core. Pass a "
        "smaller number to leave headroom for other work on the same machine.",
    )(f)


def _species_chem_option(f):
    """What: --conversion option shared by every command that
    runs/reads the per-location pathway search (calc-paths-serial/
    calc-paths-parallel/merge-paths/calc-taus/run-pipeline-serial/
    run-pipeline-parallel) - see quenchpath.config.SpeciesConversion.
    Each conversion writes to its own
    `<conversion>_paths/`/`tau_<conversion>_*/` trees (see
    quenchpath.io.paths_dir()/tau_chem_dir()).

    Why separate trees: so the two never collide and either can be
    (re)run independently of the other.

    What: flag text matches the Python parameter name (`conversion`)
    directly - no click.option() dest override needed, unlike some
    other flags in this file."""
    return click.option(
        "--conversion",
        type=_SPECIES_CONVERSION_CHOICE,
        default=SpeciesConversion.CH4_CO.value,
        show_default=True,
        help="Species conversion - CH4-to-CO (the original conversion) or "
        "CO-to-CH4 (added for a like-for-like comparison against literature quench "
        "timescales fit in that direction - see quenchpath.analysis.zahnle_marley_2014). "
        "Saved to a separate directory tree per conversion, never colliding.",
    )(f)


@click.group()
def main():
    """quenchpath: CH4-to-CO (and, via --conversion, CO-to-CH4) chemical
    pathway analysis for hot Jupiter atmospheres."""


@main.command("read-polynomials")
def read_polynomials():
    """Stage 01: parse the 7-term NASA polynomial coefficients and save
    them as JSON. Not per-run - covers the whole chemical network."""
    plns = load_and_save_nasa_polynomials(
        path_to_raw_networks, path_to_networks_processed, VENOT2019["all_gases"]
    )
    click.echo(f"Parsed and saved {len(plns)} species' NASA polynomials.")


@main.command("read-network")
def read_network():
    """Stage 02: parse the Venot et al. (2019) reduced chemical network's
    reaction rate-law files and save them as JSON. Not per-run."""
    eqns = load_and_save_venot2019_network(path_to_raw_networks, path_to_networks_processed)
    click.echo(f"Parsed and saved {len(eqns)} reactions.")


@main.command("convert")
@_run_config_options
def convert(planet, exp, mdh, tK):
    """Stage 03: convert a raw run's mole fractions to number densities."""
    cfg = RunConfig(planet=planet, exp=exp, mdh=mdh, tK=tK)
    convert_mole_fraction_to_number_density(cfg)
    click.echo(f"Converted and saved cubes for {cfg}.")


@main.command("calc-rates")
@_run_config_options
def calc_rates_cmd(planet, exp, mdh, tK):
    """Stage 04: compute equilibrium constants and forward/reverse rates
    for every reaction in the network."""
    cfg = RunConfig(planet=planet, exp=exp, mdh=mdh, tK=tK)
    calc_rates(cfg)
    click.echo(f"Computed and saved rate cubes for {cfg}.")


@main.command("calc-paths-serial")
@_run_config_options
@_grid_resolution_option(
    "SAMPLE (~4896 locations/planet) or FULL (every native grid point - "
    "n_levels x 90 x 144, n_levels is the run's own native level count)."
)
@_species_chem_option
def calc_paths_serial_cmd(planet, exp, mdh, tK, grid_resolution, conversion):
    """Stage 05: find the requested species conversion's pathway (CH4-to-CO
    by default) at every grid location. Saves one JSON Lines file per level
    as it goes (see quenchpath.io)."""
    cfg = RunConfig(planet=planet, exp=exp, mdh=mdh, tK=tK)
    conversion = SpeciesConversion(conversion)
    n = 0
    for _ in calc_paths_serial(cfg, GridResolution(grid_resolution), conversion=conversion):
        n += 1
    click.echo(f"Found and saved {conversion.label} pathways at {n} locations for {cfg}.")


@main.command("calc-paths-parallel")
@_run_config_options
@_grid_resolution_option(
    "SAMPLE (~4896 locations/planet) or FULL (every native grid point - "
    "n_levels x 90 x 144, n_levels is the run's own native level count)."
)
@_max_workers_option
@_species_chem_option
def calc_paths_parallel_cmd(planet, exp, mdh, tK, grid_resolution, max_workers, conversion):
    """Stage 05 (parallel): same as calc-paths-serial, multiprocessing across
    vertical levels instead of one serial loop - see
    quenchpath.stages.stage_05_paths_parallel's module docstring."""
    cfg = RunConfig(planet=planet, exp=exp, mdh=mdh, tK=tK)
    conversion = SpeciesConversion(conversion)
    n = 0
    for _ in calc_paths_parallel(
        cfg, GridResolution(grid_resolution), max_workers, conversion=conversion
    ):
        n += 1
    click.echo(f"Found and saved {conversion.label} pathways at {n} locations for {cfg}.")


@main.command("merge-paths")
@_run_config_options
@_grid_resolution_option(
    "Which grid resolution's already-saved calc-paths-serial/calc-paths-parallel "
    "JSON to merge - see either command's own --help."
)
@_species_chem_option
def merge_paths(planet, exp, mdh, tK, grid_resolution, conversion):
    """Stage 06: merge a completed calc-paths-serial/calc-paths-parallel run's
    per-level JSON Lines (one file per level) into one .nc file of gridded cubes (temp, pres,
    path_index, net_rate_inv, max_rate_inv) - a single, frozen, shareable
    artifact with the path-index assignment pinned, for reproducible
    cross-run comparisons and figure provenance."""
    cfg = RunConfig(planet=planet, exp=exp, mdh=mdh, tK=tK)
    grid_resolution = GridResolution(grid_resolution)
    conversion = SpeciesConversion(conversion)
    temp_cube, _, _, _, _, seq_to_index, _ = merge_and_save_paths(
        cfg, grid_resolution, conversion=conversion
    )
    fname = io.merged_paths_filename(cfg, grid_resolution, conversion)
    click.echo(
        f"Merged {temp_cube.data.size} locations for {cfg} ({grid_resolution.value}) into "
        f"{len(seq_to_index)} pathway indices. Saved to {fname}."
    )


def _calc_and_save_taus(cfg, grid_resolution, conversion=SpeciesConversion.CH4_CO):
    """calc-taus's actual work - shared with run-pipeline-serial/
    run-pipeline-parallel's own final step. Computes and saves whichever
    timescales are actually valid for `cfg.exp` - see
    quenchpath.stages.stage_07_taus's calc_and_save_adv_qn()/
    calc_and_save_chem_timescale() docstrings for why tau_adv_qn/
    tau_ch4_co_* aren't computable from just any exp:

    - tau_adv_ls: always (works for either "kin" or "equ") - `conversion`
      is not passed.
      Why: it's meaningless for this one (a wind/temperature-only
      quantity, not pathway-derived) - recomputing the same tau_adv_ls
      file under a different `--conversion` is harmless, just
      redundant.
    - tau_adv_qn: only if `cfg.exp == "equ"` - `conversion.source` is
      passed rather than `conversion` itself (that function itself
      takes a plain `species` string, not a conversion).
      Why: unlike tau_adv_ls, this one does depend on `conversion` -
      it's computed from `conversion.source`'s own mole fraction (see
      calc_and_save_adv_qn()'s own docstring).
    - tau_ch4_co_rls/tau_ch4_co_net: only if `cfg.exp == "kin"` (needs a
      completed merge-paths run at `grid_resolution`/`conversion`).
    """
    conversion = SpeciesConversion(conversion)
    calc_and_save_adv_ls(cfg)
    if cfg.exp == "equ":
        calc_and_save_adv_qn(cfg, species=conversion.source)
    if cfg.exp == "kin":
        calc_and_save_chem_timescale(cfg, grid_resolution, "rls", conversion=conversion)
        calc_and_save_chem_timescale(cfg, grid_resolution, "net", conversion=conversion)


@main.command("calc-taus")
@_run_config_options
@_grid_resolution_option(
    "Which grid resolution's already-saved merge-paths .nc to use for the "
    "CH4-to-CO pathway timescales - see `merge-paths --help`. Ignored unless "
    "--exp kin (see this command's own help for which timescales get computed "
    "for which exp)."
)
@_species_chem_option
def calc_taus(planet, exp, mdh, tK, grid_resolution, conversion):
    """Stage 07 (calculation only): compute and save whichever advection/
    pathway timescales are valid for --exp, each to its own .nc file -
    tau_adv_ls always; tau_adv_qn only for --exp equ; tau_<conversion>_rls/
    tau_<conversion>_net only for --exp kin (need a completed merge-paths
    run at --conversion).
    """
    warnings.filterwarnings("ignore", module="iris")
    cfg = RunConfig(planet=planet, exp=exp, mdh=mdh, tK=tK)
    _calc_and_save_taus(
        cfg, GridResolution(grid_resolution), conversion=SpeciesConversion(conversion)
    )
    click.echo(f"Computed and saved timescales for {cfg} ({conversion}).")


def _run_pipeline(cfg, grid_resolution, conversion, calc_paths_fn, calc_paths_label):
    """Shared convert -> calc-rates -> calc-paths-serial/-parallel ->
    merge-paths -> calc-taus chain - run-pipeline-serial/-parallel are
    thin wrappers around this, differing only in which calc-paths-serial/
    calc-paths-parallel implementation they use (see either command's own
    docstring for the --exp equ skip's rationale, identical for both).

    `calc_paths_fn(cfg, grid_resolution, conversion=conversion)` must be a
    generator yielding one entry per completed location (only its count
    is used - run-pipeline-parallel's own command passes a lambda
    closing over its own --max-workers). `calc_paths_label` names that
    step in the echoed progress ("calc-paths-serial"/"calc-paths-parallel").
    """
    if cfg.exp == "equ":
        _calc_and_save_taus(cfg, grid_resolution, conversion=conversion)
        click.echo(f"calc-taus: done for {cfg} (--exp equ needs no pathway search).")
        return
    convert_mole_fraction_to_number_density(cfg)
    click.echo(f"[1/5] convert: done for {cfg}.")
    calc_rates(cfg)
    click.echo(f"[2/5] calc-rates: done for {cfg}.")
    n = 0
    for _ in calc_paths_fn(cfg, grid_resolution, conversion=conversion):
        n += 1
    click.echo(f"[3/5] {calc_paths_label}: done for {cfg}, {n} locations ({conversion.label}).")
    temp_cube, _, _, _, _, seq_to_index, _ = merge_and_save_paths(
        cfg, grid_resolution, conversion=conversion
    )
    click.echo(
        f"[4/5] merge-paths: done for {cfg}, {temp_cube.data.size} locations, "
        f"{len(seq_to_index)} pathway indices."
    )
    _calc_and_save_taus(cfg, grid_resolution, conversion=conversion)
    click.echo(f"[5/5] calc-taus: done for {cfg}.")


@main.command("run-pipeline-serial")
@_run_config_options
@_grid_resolution_option(
    "Grid resolution for the calc-paths-serial step - see `calc-paths-serial "
    "--help`. SAMPLE (the default) is fast enough serially that this is the "
    "recommended command for quick testing; for a FULL-resolution run, "
    "use run-pipeline-parallel instead."
)
@_species_chem_option
def run_pipeline_serial(planet, exp, mdh, tK, grid_resolution, conversion):
    """Convenience command chaining convert -> calc-rates -> calc-paths-serial
    -> merge-paths -> calc-taus (stages 03-07) for one planet/exp/mdh/tK,
    using calc-paths-serial's single-process implementation for stage 05.
    Stages 01/02 (network parsing) are separate commands since they're
    network-wide, not per-run.

    What: `--exp equ` skips straight to calc-taus.

    Why: pathway search has no equilibrium-chemistry equivalent (see
    calc-paths-serial --help), and calc-taus's own equ timescales
    (tau_adv_ls/tau_adv_qn) read straight from the raw run, never from
    convert/calc-rates/calc-paths-serial/merge-paths' output - running
    those first would be genuine wasted compute, not just conceptually
    dubious.
    """
    warnings.filterwarnings("ignore", module="iris")
    cfg = RunConfig(planet=planet, exp=exp, mdh=mdh, tK=tK)
    _run_pipeline(
        cfg,
        GridResolution(grid_resolution),
        SpeciesConversion(conversion),
        calc_paths_serial,
        "calc-paths-serial",
    )


@main.command("run-pipeline-parallel")
@_run_config_options
@_grid_resolution_option(
    "Grid resolution for the calc-paths-parallel step - see `calc-paths-"
    "parallel --help`. Pass --grid-resolution full for a FULL-resolution "
    "run (this command's main reason to exist over run-pipeline-serial - "
    "see calc-paths-parallel's own module docstring for the ~20+ minute "
    "serial vs. couple-of-minutes parallel comparison at FULL)."
)
@_max_workers_option
@_species_chem_option
def run_pipeline_parallel(planet, exp, mdh, tK, grid_resolution, max_workers, conversion):
    """Convenience command chaining convert -> calc-rates ->
    calc-paths-parallel -> merge-paths -> calc-taus (stages 03-07) for one
    planet/exp/mdh/tK, using calc-paths-parallel's multiprocessing
    implementation for stage 05. Stages 01/02 (network parsing) are
    separate commands since they're network-wide, not per-run.

    --exp equ skips straight to calc-taus - see run-pipeline-serial's own
    docstring for why (identical reasoning, same skip)."""
    warnings.filterwarnings("ignore", module="iris")
    cfg = RunConfig(planet=planet, exp=exp, mdh=mdh, tK=tK)
    _run_pipeline(
        cfg,
        GridResolution(grid_resolution),
        SpeciesConversion(conversion),
        lambda cfg, gr, conversion: calc_paths_parallel(
            cfg, gr, max_workers, conversion=conversion
        ),
        "calc-paths-parallel",
    )


if __name__ == "__main__":
    main()
