"""Filename conventions and iris load/save wrappers shared across
pipeline stages, keyed by RunConfig.
"""

import json

import iris
import iris.cube

from quenchpath.config import GridResolution, SpeciesConversion
from quenchpath.mypaths import path_to_intermediates, path_to_raw_runs, path_to_results

# What: opt in to iris's newer split-attribute netCDF saving (global
# attributes shared across every saved cube vs local, per-variable ones
# - see iris.cube.CubeAttrsDict) instead of the deprecated legacy mode.
# Every custom identity attribute this module sets (see
# save_merged_paths()/quenchpath.stages.stage_04_rates.calc_rates()) is
# placed into `cube.attributes.globals` explicitly now.
#
# Why: the legacy mode guessed the split by comparing cubes' attributes
# at save time, rather than relying on that legacy auto-detection.
iris.FUTURE.save_split_attrs = True


def _cfg_prefix(cfg):
    """What: the `{planet}_{exp}_mdh{mdh}_tK{tK}` identity prefix every
    saved filename below is keyed by.
    Why shared: so it can't silently drift between functions."""
    return f"{cfg.planet}_{cfg.exp}_mdh{cfg.mdh}_tK{cfg.tK}"


def raw_run_filename(cfg):
    """Filename of the raw (merged) model output for `cfg`, under
    `path_to_raw_runs`."""
    return f"{_cfg_prefix(cfg)}.nc"


_CONVERTED_DIR = path_to_intermediates / "converted"
_RATES_DIR = path_to_intermediates / "rates"


def converted_filename(cfg):
    """Filename of stage 03's converted (number density on native height
    levels) output for `cfg`, under `path_to_intermediates/converted`."""
    return f"{_cfg_prefix(cfg)}_converted.nc"


def load_raw_run(cfg):
    """Load the raw CubeList for `cfg`, from `path_to_raw_runs`."""
    return iris.load(path_to_raw_runs / raw_run_filename(cfg))


def save_converted(cubes, cfg):
    """What: save stage 03's output CubeList for `cfg`, under
    `path_to_intermediates/converted`. Stamps cfg's own identity
    (planet/exp/mdh/tK) into `attributes.globals`, matching every other
    save_*() in this module.
    Why stamp identity: self-describing on its own, not just via the
    filename."""
    _CONVERTED_DIR.mkdir(parents=True, exist_ok=True)
    identity = {"planet": cfg.planet, "exp": cfg.exp, "mdh": cfg.mdh, "tK": cfg.tK}
    for cube in cubes:
        cube.attributes.globals.update(identity)
    iris.save(cubes, _CONVERTED_DIR / converted_filename(cfg))


def load_converted(cfg):
    """Load stage 03's converted output CubeList for `cfg`, from
    `path_to_intermediates/converted`."""
    return iris.load(_CONVERTED_DIR / converted_filename(cfg))


# What: cubes per saved file - bounds peak memory per iris.save() call.
#
# Why chunk at all: iris.save() realizes a whole CubeList before
# writing (no true streaming/append support in the installed version),
# so saving all ~1080 cubes (180 reactions x 6) in one call held the
# entire ~7GB dataset in memory at once and was ~8x slower than saving
# incrementally.
#
# Why 120: keeps memory bounded (~120 cubes, roughly 800MB, per file)
# while still cutting the file count drastically (180 -> ~9) versus one
# file per reaction.
_RATE_CUBES_PER_FILE = 120


def rate_cubes_filename(cfg, chunk):
    """What: filename of stage 04's saved rate/energy cubes for `cfg`'s
    `chunk`-th batch (see `_RATE_CUBES_PER_FILE`), under
    `path_to_intermediates/rates`. `chunk="*"` builds load_rate_cubes()'s
    glob pattern instead of an int's filename.
    Why: single source of truth for both."""
    chunk_str = f"{chunk:03d}" if isinstance(chunk, int) else str(chunk)
    return f"{_cfg_prefix(cfg)}_rates_{chunk_str}.nc"


def save_rate_cubes(cubes, cfg):
    """Save every reaction's rate/energy cubes for `cfg` (6 cubes x
    n_reactions), batched into a handful of files (see
    `_RATE_CUBES_PER_FILE`) under `path_to_intermediates/rates`. Each
    cube's `eqn` attribute identifies which reaction it belongs to.
    """
    _RATES_DIR.mkdir(parents=True, exist_ok=True)
    for chunk, start in enumerate(range(0, len(cubes), _RATE_CUBES_PER_FILE)):
        batch = iris.cube.CubeList(cubes[start : start + _RATE_CUBES_PER_FILE])
        iris.save(batch, _RATES_DIR / rate_cubes_filename(cfg, chunk))


def load_rate_cubes(cfg):
    """Load every reaction's saved rate/energy cubes for `cfg`, across
    all its chunked files. Caller splits the result by cube
    `.name()`/`.attributes["eqn"]`."""
    return iris.load(_RATES_DIR / rate_cubes_filename(cfg, "*"))


def paths_dir(cfg, grid_resolution, conversion=SpeciesConversion.CH4_CO):
    """What: directory stage 05's per-location JSON files live under:
    `path_to_intermediates/<ch4_co|co_ch4>_paths/<sample|full>/<planet>/
    <exp>_mdh{mdh}_tK{tK}/` - fully keyed by cfg's identity. `exp` is
    folded into the leaf directory name rather than getting its own
    nesting level.

    Why fully keyed by identity: so different mdh/tK runs for the same
    planet/exp never collide. Why exp folded into the leaf name:
    kin/equ are never meaningfully compared at the same planet/mdh/tK
    the way different mdh/tK values might be.

    What: `conversion` (see quenchpath.config.SpeciesConversion) picks
    which of two sibling trees this lands under - `CH4_CO` (the
    default) reproduces this function's own pre-`conversion=` path
    byte-for-byte; `CO_CH4` is a clean sibling.

    Why CH4_CO reproduces the old path exactly: so every already-saved
    CH4-to-CO run keeps its exact location. Why CO_CH4 is a clean
    sibling: never colliding with it."""
    conversion = SpeciesConversion(conversion)
    gr = GridResolution(grid_resolution).value
    return (
        path_to_intermediates
        / f"{conversion.value}_paths"
        / gr
        / cfg.planet
        / f"{cfg.exp}_mdh{cfg.mdh}_tK{cfg.tK}"
    )


def paths_level_filename(cfg, grid_resolution, level_index, conversion=SpeciesConversion.CH4_CO):
    """What: filename of one vertical level's saved pathway JSON Lines
    (every lat/lon at that level, one JSON object per line). Keyed by
    integer level index, not the level's coordinate value.

    Why integer index, not value: two levels can round to the same
    formatted value (e.g. adjacent native level_height levels near the
    top of the atmosphere), which would silently collide/overwrite
    under a value-based filename.

    What: `grid_resolution` right after `cfg`'s own identity, matching
    every other filename in this module. `conversion` - see
    paths_dir().

    Why grid_resolution belongs in the filename: a SAMPLE and a FULL
    run of the same cfg/conversion only ever differ by directory (see
    paths_dir()) otherwise, and a filename should be unique on its own,
    not just relative to whatever directory happens to hold it."""
    conversion = SpeciesConversion(conversion)
    gr = GridResolution(grid_resolution).value
    return f"{_cfg_prefix(cfg)}_{gr}_{conversion.value}_paths_lev{level_index:02d}.jsonl"


def save_paths_level(
    location_dicts,
    cfg,
    grid_resolution,
    level_index,
    level_height,
    conversion=SpeciesConversion.CH4_CO,
):
    """Save one vertical level's worth of pathway dicts as JSON Lines
    under `paths_dir`. Cuts stage 05's file count from one per
    location (90*144 per level at FULL resolution) to one per level (as
    many levels as the run's native grid has at FULL, every 4th at
    SAMPLE).

    What: the first line is a header record - `cfg`'s identity,
    `grid_resolution`/`conversion` (the same identity the filename/
    directory already encode), plus this level's own
    `level_index`/`level_height`.

    Why keep identity in the header too: so the file is self-describing
    on its own, not just via wherever it happens to be found.

    Why level_index/level_height go in the header, not each line: so
    every following line can drop those, which would otherwise be
    repeated unchanged on every one of a level's ~90*144 locations."""
    conversion = SpeciesConversion(conversion)
    out_dir = paths_dir(cfg, grid_resolution, conversion)
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = paths_level_filename(cfg, grid_resolution, level_index, conversion)
    header = {
        "planet": cfg.planet,
        "exp": cfg.exp,
        "mdh": cfg.mdh,
        "tK": cfg.tK,
        "grid_resolution": GridResolution(grid_resolution).value,
        "conversion": conversion.value,
        "hgt_idx": level_index,
        "hgt": float(level_height),
    }
    with open(out_dir / fname, "w") as f:
        f.write(json.dumps(header))
        f.write("\n")
        for d in location_dicts:
            f.write(json.dumps(d))
            f.write("\n")


def load_paths_from_disk(cfg, grid_resolution, conversion=SpeciesConversion.CH4_CO):
    """What: load every saved vertical level's pathway JSON Lines for
    `cfg`/`grid_resolution`/`conversion` from disk into one flat list of
    location dicts, each with its file's header-line `hgt_idx`/`hgt`
    folded back in - `grid_resolution`/`conversion` themselves aren't.

    Why not fold in grid_resolution/conversion too: already known from
    this call's own arguments, and unlike hgt_idx/hgt they aren't
    needed per-location downstream, only checked once per file below."""
    conversion = SpeciesConversion(conversion)
    gr = GridResolution(grid_resolution).value
    directory = paths_dir(cfg, grid_resolution, conversion)
    prefix = f"{_cfg_prefix(cfg)}_{gr}_{conversion.value}_paths_lev"
    locations = []
    for entry in directory.iterdir():
        if not (entry.name.startswith(prefix) and entry.suffix == ".jsonl"):
            continue
        with open(entry) as f:
            header = json.loads(f.readline())
            level = {"hgt_idx": header["hgt_idx"], "hgt": header["hgt"]}
            locations.extend({**json.loads(line), **level} for line in f)
    if not locations:
        raise FileNotFoundError(f"No {conversion.label} path JSONL found under {directory}")
    return locations


def merged_paths_dir(grid_resolution, conversion=SpeciesConversion.CH4_CO, base_dir=None):
    """Directory quenchpath.stages.stage_06_merge_paths's merged pathway
    cubes live under: `path_to_results/<ch4_co|co_ch4>_paths/<sample|full>/`
    (see paths_dir() for `conversion`'s own doc). `base_dir`, if
    given, replaces `path_to_results` - e.g. for loading (not saving -
    real pipeline output always lives under `path_to_results`) a bundled
    example dataset from `example_data/` instead, same
    "<conversion>_paths/<sample|full>/" layout underneath."""
    conversion = SpeciesConversion(conversion)
    gr = GridResolution(grid_resolution).value
    return (base_dir or path_to_results) / f"{conversion.value}_paths" / gr


def merged_paths_filename(cfg, grid_resolution, conversion=SpeciesConversion.CH4_CO):
    """Filename of quenchpath.stages.stage_06_merge_paths's merged
    pathway cubes for `cfg`/`grid_resolution`/`conversion`, under
    `merged_paths_dir(grid_resolution, conversion)`. `grid_resolution`
    right after `cfg`'s own identity, matching every other filename in
    this module."""
    conversion = SpeciesConversion(conversion)
    gr = GridResolution(grid_resolution).value
    return f"{_cfg_prefix(cfg)}_{gr}_{conversion.value}_paths.nc"


def save_merged_paths(
    temp_cube,
    pres_cube,
    path_index_cube,
    net_rate_inv_cube,
    max_rate_inv_cube,
    identity_map,
    cfg,
    grid_resolution,
    conversion=SpeciesConversion.CH4_CO,
):
    """What: save merge_paths_to_cubes()'s 5 cubes to one .nc file under
    `merged_paths_dir(grid_resolution, conversion)`. `identity_map` (see
    quenchpath.analysis.path_index.path_index_map_from_locations() -
    {path_index: {"eqns", "directions", "nodes"}}) is embedded as a JSON
    string attribute on the path_index cube.

    Why embed identity_map: so the saved .nc is self-describing - the
    full pathway identity survives the merge, not just the bare
    path_index int.

    What: every cube also gets cfg/grid_resolution/conversion's identity
    attributes (planet/exp/mdh/tK/grid_resolution/conversion) - identical
    across all 5, placed directly into each cube's `attributes.globals`
    (see iris.cube.CubeAttrsDict/`iris.FUTURE.save_split_attrs`, set in
    this module's own header).

    Why attributes.globals, not per-variable: so iris.save() writes
    them once as the .nc's global attributes rather than repeating them
    per-variable.

    Note: load_merged_paths() never reads these attributes back (only
    `path_index_map`) - they're for external/human inspection of the
    saved .nc only.
    """
    conversion = SpeciesConversion(conversion)
    out_dir = merged_paths_dir(grid_resolution, conversion)
    out_dir.mkdir(parents=True, exist_ok=True)
    identity = {
        "planet": cfg.planet,
        "exp": cfg.exp,
        "mdh": cfg.mdh,
        "tK": cfg.tK,
        "grid_resolution": GridResolution(grid_resolution).value,
        "conversion": conversion.value,
    }
    temp_cube = temp_cube.copy()
    pres_cube = pres_cube.copy()
    path_index_cube = path_index_cube.copy()
    net_rate_inv_cube = net_rate_inv_cube.copy()
    max_rate_inv_cube = max_rate_inv_cube.copy()
    path_index_cube.attributes["path_index_map"] = json.dumps(identity_map)
    cubes = iris.cube.CubeList(
        [temp_cube, pres_cube, path_index_cube, net_rate_inv_cube, max_rate_inv_cube]
    )
    for cube in cubes:
        cube.attributes.globals.update(identity)
    fname = merged_paths_filename(cfg, grid_resolution, conversion)
    iris.save(cubes, out_dir / fname)


def _extract_cube_by_var_name(cubes, var_name):
    """The one cube in `cubes` whose `var_name` is `var_name` - raises if
    there isn't exactly one match."""
    [cube] = [c for c in cubes if c.var_name == var_name]
    return cube


def load_merged_paths(cfg, grid_resolution, conversion=SpeciesConversion.CH4_CO, base_dir=None):
    """Load save_merged_paths()'s output back as
    (temp_cube, pres_cube, path_index_cube, net_rate_inv_cube,
    max_rate_inv_cube, path_index_map), where `path_index_map` is the
    {index: {"eqns", "directions", "nodes"}} dict (see
    quenchpath.analysis.path_index.path_index_map_from_locations()) recovered from the
    path_index cube's embedded attribute. `conversion`/`base_dir` - see
    merged_paths_dir()."""
    conversion = SpeciesConversion(conversion)
    fname = merged_paths_filename(cfg, grid_resolution, conversion)
    merged_dir = merged_paths_dir(grid_resolution, conversion, base_dir=base_dir)
    cubes = iris.load(merged_dir / fname)
    temp_cube = _extract_cube_by_var_name(cubes, "temp")
    pres_cube = _extract_cube_by_var_name(cubes, "pres")
    path_index_cube = _extract_cube_by_var_name(cubes, "path_index")
    net_rate_inv_cube = _extract_cube_by_var_name(cubes, "net_rate_inv")
    max_rate_inv_cube = _extract_cube_by_var_name(cubes, "max_rate_inv")
    path_index_map = {
        int(index): entry
        for index, entry in json.loads(path_index_cube.attributes["path_index_map"]).items()
    }
    return (
        temp_cube,
        pres_cube,
        path_index_cube,
        net_rate_inv_cube,
        max_rate_inv_cube,
        path_index_map,
    )


# stage 07's advection/quenching timescale cubes, keyed by cfg like
# every other stage's I/O - see quenchpath.stages.stage_07_taus. No raw/
# regridded passthroughs (u/v/w/temp/ch4) - stage_07_taus.py doesn't
# return those, see its own calc_and_save_adv_ls()/calc_and_save_adv_qn()
# docstrings.
#
# What: tau_adv_ls and tau_adv_qn are saved to separate files (not one
# combined one).
#
# Why: they aren't always computable together - tau_adv_ls only needs
# wind/temperature (either "kin" or "equ" raw run has it), but
# tau_adv_qn needs a mole fraction (some species' - CH4 by default, see
# calc_and_save_adv_qn()'s own docstring) from specifically the "equ"
# run - a caller with only one experiment's data available couldn't
# produce a combined file at all. Splitting the
# files (plus the tau_ch4_co_* ones below) also means a plotting stage
# loading e.g. tau_adv_qn can never accidentally end up with a "kin"
# one - there's no such file to load.
_TAU_ADV_LS_KEYS = ["tau_adv_ls_u", "tau_adv_ls_v", "tau_adv_ls_w", "tau_adv_ls_dom"]
_TAU_ADV_QN_KEYS = ["tau_adv_qn_u", "tau_adv_qn_v", "tau_adv_qn_w", "tau_adv_qn"]

# What: both always the full native grid, nested under "full"
# (mirroring ch4_co_paths/{sample,full}/).
#
# Why always full: stage_07_taus's wind-based timescales have no genuine
# sample-resolution variant (see that module's own docstring).
#
# Why nest under "full" anyway (not skip the resolution level
# entirely): keeps every "taus/<resolution>/<chem_kind>/" path shaped
# the same way, tau_ch4_co_* (below) included.
_TAU_ADV_LS_DIR = path_to_results / "taus" / "full" / "tau_adv_ls"
_TAU_ADV_QN_DIR = path_to_results / "taus" / "full" / "tau_adv_qn"


def tau_adv_filename(cfg, adv_kind, species="CH4", grid_resolution=GridResolution.FULL):
    """Filename of stage 07's saved advection timescale cubes for `cfg`,
    under `path_to_results/taus/full/tau_adv_<ls|qn>`. `adv_kind`: "ls"
    (length-scale) or "qn" (quenching-limited) - raises ValueError on
    anything else.

    What: `species` only matters for `adv_kind="qn"` (ignored for "ls",
    which has no species dependence of its own - see
    calc_adv_ls()'s own docstring).

    Why it matters for "qn": tau_adv_qn is computed from `species`'s
    own mole fraction (see calc_and_save_adv_qn()'s own docstring), so
    a CH4-flavoured and a CO-flavoured run are different data, not the
    same quantity twice.

    What: deliberately a plain species string, not a
    quenchpath.config.SpeciesConversion.

    Why: tau_adv_qn only cares which single species it's about,
    independent of which pathway-search direction/conversion a caller
    (quenchpath.analysis.quench_level) might be comparing it against.

    What: lowercased in the filename itself (`species.lower()`) -
    `species` stays its real uppercase form everywhere else (e.g. the
    raw run's own "CH4 mole fraction" cube name).

    Why lowercased in the filename: to match every other on-disk name's
    own lowercase convention (tau_ch4_co_*, ch4_co_paths/, ...).

    What: `grid_resolution` defaults to FULL.

    Why: the only resolution the real pipeline ever saves tau_adv at
    (see this module's own header comment on
    _TAU_ADV_LS_DIR/_TAU_ADV_QN_DIR for why).

    What: the one real exception is `inputs/example_data/`'s own
    bundled, deliberately-subsampled copy (see
    scripts/generate_example_data.py), which passes
    `grid_resolution=GridResolution.SAMPLE` explicitly - see
    load_tau_adv()'s own `base_dir` handling.

    Why pass it explicitly there: to get a genuinely different, honest
    filename for that different content."""
    if adv_kind not in ("ls", "qn"):
        raise ValueError(f"adv_kind must be 'ls' or 'qn', got {adv_kind!r}")
    gr = GridResolution(grid_resolution).value
    if adv_kind == "ls":
        return f"{_cfg_prefix(cfg)}_{gr}_tau_adv_ls.nc"
    return f"{_cfg_prefix(cfg)}_{gr}_tau_adv_qn_{species.lower()}.nc"


def _cubes_with_shared_coords(result, keys, extra_aux_coord_name=None):
    """What: `[result[key].copy() for key in keys]`, each renamed to a
    stable var_name matching its dict key, with level_height (and
    `extra_aux_coord_name`, if given, e.g. "pres_mean") replaced by one
    shared coord object taken from `result[keys[0]]` - not each cube's
    own independently-derived copy. Shared by save_tau_adv()'s two
    `adv_kind`s (only whether an extra aux coord is carried along
    differs between them).

    Why replace with one shared coord object: some of these cubes lost
    their coord bounds along the way (derived from u/v, regridded onto
    temp's grid), which otherwise makes iris.save() treat them as
    different axes and split level_height across level_height_0/_1/...
    instead of aligning everything under one shared name.
    """
    level_coord = result[keys[0]].coord("level_height").copy()
    extra_coord = (
        result[keys[0]].coord(extra_aux_coord_name).copy() if extra_aux_coord_name else None
    )

    cubes = iris.cube.CubeList()
    for key in keys:
        cube = result[key].copy()
        cube.var_name = key
        (dim,) = cube.coord_dims("level_height")
        cube.remove_coord("level_height")
        cube.add_dim_coord(level_coord, dim)
        if extra_coord is not None:
            cube.remove_coord(extra_aux_coord_name)
            cube.add_aux_coord(extra_coord, dim)
        cubes.append(cube)
    return cubes


def save_tau_adv(result, cfg, adv_kind, species="CH4"):
    """Save calc_and_save_adv_ls()'s or calc_and_save_adv_qn()'s result
    dict (4 Cubes, keyed by _TAU_ADV_LS_KEYS/_TAU_ADV_QN_KEYS) to one
    .nc file under `path_to_results/taus/full/tau_adv_<ls|qn>` - each
    renamed to a stable var_name matching its dict key.

    What: `adv_kind`: "ls" (length-scale) attaches a shared pres_mean
    aux coordinate; "qn" (quenching-limited) attaches none. Raises
    ValueError on anything else.

    Why "ls" attaches pres_mean: a genuine derived day-side-mean
    pressure worth saving alongside it (see calc_adv_ls()'s own
    docstring).

    Why "qn" attaches none: its pressure would just be an unmodified
    passthrough of the raw run's own air_pressure cube, better reloaded
    directly when actually needed than carried along here (see
    calc_and_save_adv_qn()'s own docstring).

    `species` - see tau_adv_filename() (only matters for
    `adv_kind="qn"`, harmlessly ignored for "ls").

    What: every cube also gets cfg/adv_kind/species/grid_resolution's
    identity stamped into `attributes.globals` (same pattern as
    save_merged_paths()'s own `identity` dict).

    Why stamp identity: self-describing on its own, not just via
    whichever directory/filename it's found under."""
    if adv_kind == "ls":
        keys, out_dir = _TAU_ADV_LS_KEYS, _TAU_ADV_LS_DIR
        extra_aux_coord_name = "pres_mean"
    elif adv_kind == "qn":
        keys, out_dir = _TAU_ADV_QN_KEYS, _TAU_ADV_QN_DIR
        extra_aux_coord_name = None
    else:
        raise ValueError(f"adv_kind must be 'ls' or 'qn', got {adv_kind!r}")
    out_dir.mkdir(parents=True, exist_ok=True)
    cubes_to_save = _cubes_with_shared_coords(result, keys, extra_aux_coord_name)
    identity = {
        "planet": cfg.planet,
        "exp": cfg.exp,
        "mdh": cfg.mdh,
        "tK": cfg.tK,
        "grid_resolution": GridResolution.FULL.value,
        "adv_kind": adv_kind,
        "species": species,
    }
    for cube in cubes_to_save:
        cube.attributes.globals.update(identity)
    iris.save(cubes_to_save, out_dir / tau_adv_filename(cfg, adv_kind, species))


def load_tau_adv(cfg, adv_kind, species="CH4", base_dir=None):
    """Load save_tau_adv()'s output back into a dict keyed by
    _TAU_ADV_LS_KEYS/_TAU_ADV_QN_KEYS (4 Cubes each; level_height is
    each cube's own dim coordinate - plus pres_mean as an aux
    coordinate for `adv_kind="ls"` only, see save_tau_adv()'s own
    docstring for why qn has none).

    `adv_kind`: "ls" (length-scale - works from either a "kin" or "equ"
    run) or "qn" (quenching-limited - "equ"-only, see
    calc_and_save_adv_qn()'s own docstring for
    why) - raises ValueError on anything else.

    `species` - see tau_adv_filename() (only matters for
    `adv_kind="qn"`, harmlessly ignored for "ls").

    What: `base_dir`, if given, replaces `path_to_results` - see
    merged_paths_dir() for the same pattern. The only real caller today
    is `inputs/example_data/` (a bundled SAMPLE-resolution dataset, see
    scripts/generate_example_data.py), and a `base_dir` tree is
    expected to hold whatever's actually been bundled there, under
    "taus/sample/" - pass a `base_dir` some other shape isn't bundled
    under, and this raises the same as a missing file would.

    Why base_dir implies SAMPLE here: unlike the real pipeline location
    (always native/"full"), tau_adv has no genuine SAMPLE variant of
    its own."""
    if adv_kind == "ls":
        keys, subdir, default_dir = _TAU_ADV_LS_KEYS, "tau_adv_ls", _TAU_ADV_LS_DIR
    elif adv_kind == "qn":
        keys, subdir, default_dir = _TAU_ADV_QN_KEYS, "tau_adv_qn", _TAU_ADV_QN_DIR
    else:
        raise ValueError(f"adv_kind must be 'ls' or 'qn', got {adv_kind!r}")
    # What: base_dir given <=> the bundled SAMPLE example (see this
    # function's own docstring).
    # Why: the real pipeline location is always FULL.
    tau_gr = GridResolution.SAMPLE if base_dir else GridResolution.FULL
    tau_dir = (base_dir / "taus" / "sample" / subdir) if base_dir else default_dir
    cubes = iris.load(tau_dir / tau_adv_filename(cfg, adv_kind, species, grid_resolution=tau_gr))
    return {key: _extract_cube_by_var_name(cubes, key) for key in keys}


_TAU_CHEM_KINDS = ("rls", "net")


def tau_chem_dir(chem_kind, grid_resolution, conversion=SpeciesConversion.CH4_CO, base_dir=None):
    """What: directory a pathway timescale variant (`chem_kind`: "rls"
    or "net" - see quenchpath.stages.stage_07_taus.calc_chem_timescale())
    is saved under:
    `path_to_results/taus/<sample|full>/tau_<ch4_co|co_ch4>_<chem_kind>/`
    - keyed by grid_resolution and by `conversion` (see paths_dir()).
    `base_dir`, if given, replaces `path_to_results` - see
    merged_paths_dir() for the same pattern.

    Why keyed by grid_resolution: these come from quenchpath
    merge-paths' resolution-specific merged pathway data.

    Why keyed by conversion: rls/net are computed per-conversion (a
    CH4-to-CO and a CO-to-CH4 run of the same planet/exp/mdh/tK are
    different pathway searches, not the same chem_kind twice)."""
    assert chem_kind in _TAU_CHEM_KINDS, f"chem_kind must be one of {_TAU_CHEM_KINDS}"
    conversion = SpeciesConversion(conversion)
    gr = GridResolution(grid_resolution).value
    return (base_dir or path_to_results) / "taus" / gr / f"tau_{conversion.value}_{chem_kind}"


def tau_chem_filename(cfg, grid_resolution, chem_kind, conversion=SpeciesConversion.CH4_CO):
    """Filename of a pathway timescale variant (see tau_chem_dir())
    for `cfg`/`grid_resolution`/`conversion` - `grid_resolution` right
    after `cfg`'s own identity, matching every other filename in this
    module."""
    conversion = SpeciesConversion(conversion)
    gr = GridResolution(grid_resolution).value
    return f"{_cfg_prefix(cfg)}_{gr}_tau_{conversion.value}_{chem_kind}.nc"


def save_tau_chem(cube, cfg, grid_resolution, chem_kind, conversion=SpeciesConversion.CH4_CO):
    """What: save one pathway timescale Cube (`chem_kind`: "rls" or
    "net" - see tau_chem_dir()) for `cfg`/`grid_resolution`/`conversion`.
    Stamps cfg/grid_resolution/chem_kind/conversion's identity into
    `attributes.globals` (same pattern as save_merged_paths()'s own
    `identity` dict).
    Why stamp identity: self-describing on its own, not just via
    whichever directory/filename it's found under."""
    conversion = SpeciesConversion(conversion)
    out_dir = tau_chem_dir(chem_kind, grid_resolution, conversion)
    out_dir.mkdir(parents=True, exist_ok=True)
    cube = cube.copy()
    cube.attributes.globals.update(
        {
            "planet": cfg.planet,
            "exp": cfg.exp,
            "mdh": cfg.mdh,
            "tK": cfg.tK,
            "grid_resolution": GridResolution(grid_resolution).value,
            "chem_kind": chem_kind,
            "conversion": conversion.value,
        }
    )
    iris.save(cube, out_dir / tau_chem_filename(cfg, grid_resolution, chem_kind, conversion))


def load_tau_chem(
    cfg, grid_resolution, chem_kind, conversion=SpeciesConversion.CH4_CO, base_dir=None
):
    """Load save_tau_chem()'s output back as a single Cube.
    `conversion`/`base_dir` - see tau_chem_dir()."""
    conversion = SpeciesConversion(conversion)
    fname = tau_chem_filename(cfg, grid_resolution, chem_kind, conversion)
    tau_dir = tau_chem_dir(chem_kind, grid_resolution, conversion, base_dir=base_dir)
    return iris.load_cube(tau_dir / fname)
