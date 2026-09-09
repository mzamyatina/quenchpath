"""What: regenerate inputs/example_data/ from scratch - a full, real
pipeline run for one planet at SAMPLE resolution (stages 03-07, kin;
plus calc-taus equ for tau_adv_qn), copied into inputs/example_data/'s
own layout, plus a fresh one-location rates/number-density snapshot for
notebooks/plt_graphs.ipynb.

Why: not part of the fast pytest suite (tests/) - this needs real
raw-run data under inputs/raw_runs/ and takes real compute time (a few
minutes, not milliseconds), closer to a heavier integration check than
a unit test. Not a `quenchpath` CLI subcommand either - this is a
maintainer-only tool for refreshing a bundled fixture, not something a
quenchpath user would ever run. A plain script here, reusable whenever
inputs/example_data/ needs regenerating (a stage's save format
changes, a different planet is wanted, etc.), is the plainest fit for
something that's neither.

Usage
-----
    python scripts/generate_example_data.py [--planet PLANET] [--mdh MDH]
        [--tK TK] [--lat LAT] [--lon LON] [--pres PRES]

Defaults match what's currently bundled: hd189733b, mdh=0, tK=1, the
same (lat, lon, pres) target notebooks/plt_graphs.ipynb's
own non-bundled branch defaults to. Needs inputs/raw_runs/
{planet}_kin_mdh{mdh}_tK{tK}.nc and the equ counterpart (for
tau_adv_qn) already present - see docs/user_guide.md's "External data".
"""

import argparse
import json
import shutil

import iris.cube
import numpy as np

from quenchpath import io
from quenchpath.analysis.quench_level import subset_to_grid_of
from quenchpath.chemistry.network import VENOT2019
from quenchpath.config import GridResolution, RunConfig
from quenchpath.mypaths import path_to_example_data
from quenchpath.stages.stage_03_convert import convert_mole_fraction_to_number_density
from quenchpath.stages.stage_04_rates import calc_rates, split_gases_and_bath_gases
from quenchpath.stages.stage_05_paths_parallel import calc_paths_parallel
from quenchpath.stages.stage_05_paths_serial import load_eqns_with_rates
from quenchpath.stages.stage_06_merge_paths import merge_and_save_paths
from quenchpath.stages.stage_07_taus import (
    calc_and_save_adv_ls,
    calc_and_save_adv_qn,
    calc_and_save_chem_timescale,
)


def _copy_into_example_data(src, relative_to_example_data):
    dst = path_to_example_data / relative_to_example_data
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, dst)
    print(f"  {dst.relative_to(path_to_example_data)}")


def _save_subsampled_tau_adv(result, keys, has_pres_mean, reference, out_path, identity):
    """What: subset each cube in `result` (keyed by `keys` -
    io._TAU_ADV_LS_KEYS or io._TAU_ADV_QN_KEYS) onto `reference`'s
    SAMPLE-resolution grid and save them together to `out_path`.

    Why: tau_adv_ls_dom/tau_adv_qn have no genuine SAMPLE variant of
    their own (see quenchpath.analysis.quench_level's own
    _aligned_tau_adv() docstring for why that matters) - subsetting
    once here, before bundling, means that module doesn't have to redo
    it on every load, and the bundled file is small instead of carrying
    the full native grid (tau_adv_qn: 66x90x144 -> 17x18x16, ~26MB ->
    ~150KB).

    What: mirrors quenchpath.io.save_tau_adv()'s own shared-coordinate
    save pattern (one shared level_height, and pres_mean if
    `has_pres_mean`, object across all cubes - some lose their coord
    bounds along the way, which otherwise makes iris.save() split
    level_height across level_height_0/_1/... instead of aligning
    everything under one shared name) - but written directly under
    inputs/example_data/, not path_to_results (quenchpath.io's own save
    functions always target the real pipeline location, by design).
    Also mirrors save_tau_adv()'s own identity-stamping (`identity` -
    planet/exp/mdh/tK/grid_resolution/adv_kind/species - into
    `attributes.globals`). Why: this function doesn't go through
    io.save_tau_adv() itself (that always tags grid_resolution=FULL,
    the real pipeline's own only resolution; this bundle is
    deliberately SAMPLE-tagged instead, see this script's own
    tau_adv_ls_fname/tau_adv_qn_fname), so it has to redo the stamping
    itself.
    """
    reduced = {key: subset_to_grid_of(result[key], reference) for key in keys}
    level_coord = reduced[keys[0]].coord("level_height").copy()
    pres_mean_coord = reduced[keys[0]].coord("pres_mean").copy() if has_pres_mean else None

    cubes_to_save = iris.cube.CubeList()
    for key in keys:
        cube = reduced[key].copy()
        cube.var_name = key
        (dim,) = cube.coord_dims("level_height")
        cube.remove_coord("level_height")
        cube.add_dim_coord(level_coord, dim)
        if has_pres_mean:
            cube.remove_coord("pres_mean")
            cube.add_aux_coord(pres_mean_coord, dim)
        cube.attributes.globals.update(identity)
        cubes_to_save.append(cube)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    iris.save(cubes_to_save, out_path)
    print(f"  {out_path.relative_to(path_to_example_data)}")


def generate_location_snapshot(cfg, lat, lon, pres):
    """What: same construction notebooks/plt_graphs.ipynb's own
    non-bundled branch uses - one location's own reaction rates/species
    number densities.

    Why: small enough to check in directly rather than needing the
    multi-GB converted+rates grid it's drawn from.
    """
    eqns = load_eqns_with_rates(cfg)
    converted_cubes = io.load_converted(cfg)

    temp_cube = converted_cubes.extract_cube("air_temperature")
    pres_cube = converted_cubes.extract_cube("air_pressure")
    full_lats = temp_cube.coord("latitude").points
    full_lons = temp_cube.coord("longitude").points
    lai = int(np.argmin(np.abs(full_lats - lat)))
    loi = int(np.argmin(np.abs(full_lons - lon)))
    pres_profile = pres_cube.data[:, lai, loi]
    pi = int(np.argmin(np.abs(pres_profile - pres)))
    temp = float(temp_cube.data[pi, lai, loi])

    eqns_at_location = [
        {
            "eqn": eqn["eqn"],
            "reacs": eqn["reacs"],
            "prods": eqn["prods"],
            "k_eq": float(eqn["k_eq"].data[pi, lai, loi]),
            "rate_f": float(eqn["rate_f"].data[pi, lai, loi]),
            "rate_r": float(eqn["rate_r"].data[pi, lai, loi]),
        }
        for eqn in eqns
    ]
    all_gases_nd, _ = split_gases_and_bath_gases(converted_cubes, VENOT2019["bath_gases"])
    number_density = {gas.name(): float(gas.data[pi, lai, loi]) for gas in all_gases_nd}

    snapshot = {
        "cfg": {"planet": cfg.planet, "exp": cfg.exp, "mdh": cfg.mdh, "tK": cfg.tK},
        "lat": float(full_lats[lai]),
        "lon": float(full_lons[loi]),
        "pres": float(pres_profile[pi]),
        "temp": temp,
        "eqns_at_location": eqns_at_location,
        "number_density": number_density,
    }
    out_path = path_to_example_data / f"{io._cfg_prefix(cfg)}_location_snapshot.json"
    with open(out_path, "w") as f:
        json.dump(snapshot, f)
    print(f"  {out_path.relative_to(path_to_example_data)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planet", default="hd189733b")
    parser.add_argument("--mdh", default="0")
    parser.add_argument("--tK", default="1")
    parser.add_argument("--lat", type=float, default=-19.0)
    parser.add_argument("--lon", type=float, default=1.25)
    parser.add_argument("--pres", type=float, default=1e5)
    args = parser.parse_args()

    cfg = RunConfig(planet=args.planet, exp="kin", mdh=args.mdh, tK=args.tK)
    cfg_equ = RunConfig(planet=args.planet, exp="equ", mdh=args.mdh, tK=args.tK)
    grid_resolution = GridResolution.SAMPLE

    print(f"Running the kin pipeline for {cfg} at {grid_resolution.value} resolution...")
    convert_mole_fraction_to_number_density(cfg)
    calc_rates(cfg)
    n = sum(1 for _ in calc_paths_parallel(cfg, grid_resolution))
    print(f"  calc-paths-parallel: {n} locations")
    merge_and_save_paths(cfg, grid_resolution)
    tau_adv_ls_result = calc_and_save_adv_ls(cfg)
    # What: this cube also doubles as the SAMPLE-resolution reference grid
    # tau_adv_ls/tau_adv_qn get subset onto below. Why: it's already on
    # grid_resolution's own grid, so no separate reference needs building.
    sample_reference = calc_and_save_chem_timescale(cfg, grid_resolution, "net")
    calc_and_save_chem_timescale(cfg, grid_resolution, "rls")

    print(f"Running calc-taus (equ) for {cfg_equ} (tau_adv_qn only)...")
    tau_adv_qn_result = calc_and_save_adv_qn(cfg_equ)

    print(f"Copying into {path_to_example_data}:")
    ch4_co_paths_fname = io.merged_paths_filename(cfg, grid_resolution)
    _copy_into_example_data(
        io.merged_paths_dir(grid_resolution) / ch4_co_paths_fname,
        f"ch4_co_paths/{grid_resolution.value}/{ch4_co_paths_fname}",
    )
    _copy_into_example_data(
        io.tau_chem_dir("net", grid_resolution) / io.tau_chem_filename(cfg, grid_resolution, "net"),
        f"taus/{grid_resolution.value}/tau_ch4_co_net/"
        f"{io.tau_chem_filename(cfg, grid_resolution, 'net')}",
    )
    _copy_into_example_data(
        io.tau_chem_dir("rls", grid_resolution) / io.tau_chem_filename(cfg, grid_resolution, "rls"),
        f"taus/{grid_resolution.value}/tau_ch4_co_rls/"
        f"{io.tau_chem_filename(cfg, grid_resolution, 'rls')}",
    )

    print("Subsetting tau_adv_ls/tau_adv_qn to SAMPLE resolution before bundling:")
    # What: pass grid_resolution=GridResolution.SAMPLE explicitly here (not
    # io.tau_adv_filename()'s own default, FULL). Why: this bundled copy is
    # subsampled content, and the real pipeline only ever produces FULL - a
    # distinct, honest filename for this bundle needs SAMPLE stated
    # explicitly (see its own docstring).
    tau_adv_ls_fname = io.tau_adv_filename(cfg, "ls", grid_resolution=GridResolution.SAMPLE)
    tau_adv_ls_out = path_to_example_data / "taus" / "sample" / "tau_adv_ls" / tau_adv_ls_fname
    _save_subsampled_tau_adv(
        tau_adv_ls_result,
        io._TAU_ADV_LS_KEYS,
        True,
        sample_reference,
        tau_adv_ls_out,
        {
            "planet": cfg.planet,
            "exp": cfg.exp,
            "mdh": cfg.mdh,
            "tK": cfg.tK,
            "grid_resolution": GridResolution.SAMPLE.value,
            "adv_kind": "ls",
            "species": "CH4",
        },
    )

    tau_adv_qn_fname = io.tau_adv_filename(cfg_equ, "qn", grid_resolution=GridResolution.SAMPLE)
    tau_adv_qn_out = path_to_example_data / "taus" / "sample" / "tau_adv_qn" / tau_adv_qn_fname
    _save_subsampled_tau_adv(
        tau_adv_qn_result,
        io._TAU_ADV_QN_KEYS,
        False,
        sample_reference,
        tau_adv_qn_out,
        {
            "planet": cfg_equ.planet,
            "exp": cfg_equ.exp,
            "mdh": cfg_equ.mdh,
            "tK": cfg_equ.tK,
            "grid_resolution": GridResolution.SAMPLE.value,
            "adv_kind": "qn",
            "species": "CH4",
        },
    )

    print("Generating the location snapshot:")
    generate_location_snapshot(cfg, args.lat, args.lon, args.pres)

    print("Done. Remember to update the two notebooks' planet/exp/mdh/tK if they changed.")


if __name__ == "__main__":
    main()
