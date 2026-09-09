"""Quench-level data acquisition: tau_<conversion>_<rls|net> compared against
an advection timescale, returning the ratio of the two as one Cube -
the level itself is wherever that ratio crosses 1, which
quenchpath.viz.quench_level_contour_panels() draws as a contour.
quench_level() covers all 4 combinations of:

- `chem_kind`: "rls" (rate-limiting/bottleneck-segment pathway
  timescale) or "net" (whole-path-summed pathway timescale).
- `adv_kind`: "ls" (length-scale - tau_adv_ls_dom, a dayside-mean profile
  from either a "kin" or "equ" run, no conversion of its own) or "qn"
  (quenching-limited - tau_adv_qn, a full local field, "equ"-only,
  computed from `conversion.source`'s own mole fraction - see
  quenchpath.stages.stage_07_taus.calc_and_save_adv_qn()'s own
  docstring).

What: plus quench_level_zahnle_ls()/quench_level_zahnle_qn(), pairing
the same two tau_adv quantities against a literature timescale instead
of a quenchpath-derived one - see quenchpath.analysis.zahnle_marley_2014's
own module docstring for why. Kept separate from quench_level() itself
(not a 3rd `chem_kind` option) - they take already-loaded `pres`/`temp`
Cubes instead of `grid_resolution`/`conversion`.

Why kept separate: no on-disk quantity of their own to load, so
folding them in would mean a pile of arguments that are unused
depending on mode instead of a clean shared signature.

Pure data acquisition, no matplotlib - see quenchpath.viz for the actual
contour/legend construction these ratio cubes feed into. See also
quenchpath.analysis.quench_region, a continuous `ln(ratio)` alternative
to quench_level()'s own sharp ratio=1 crossing, built on quench_level()
but kept in its own module rather than folded into this one.
"""

import iris
import numpy as np

from quenchpath.analysis.zahnle_marley_2014 import zahnle_marley_2014_tco
from quenchpath.config import RunConfig, SpeciesConversion
from quenchpath.io import load_tau_adv, load_tau_chem

_QUENCH_LEVEL_ADV_KINDS = ("ls", "qn")


def subset_to_grid_of(cube, reference):
    """What: `cube`, subset onto `reference`'s own level_height/latitude/
    longitude points, along whichever of those `cube` actually has as a
    real (not collapsed-to-scalar) dimension - a tau_adv_ls_dom cube:
    level_height only; a tau_adv_qn cube: all three. A no-op wherever
    `cube` already matches `reference` along a given coordinate (e.g. at
    GridResolution.FULL, or on an already pre-shaped bundled example -
    see scripts/generate_example_data.py).

    Why a no-op in that case: so callers don't need to know or guard on
    which case they're in.

    Why public: also used by scripts/generate_example_data.py to
    pre-shape tau_adv_ls/tau_adv_qn before bundling them, not just
    internally by quench_level() below."""
    constraint_kw = {}
    for coord in cube.coords(dim_coords=True):
        name = coord.name()
        if not reference.coords(name):
            continue
        target_points = reference.coord(name).points
        if np.array_equal(coord.points, target_points):
            continue
        constraint_kw[name] = lambda c, target_points=target_points: c.point in target_points
    if not constraint_kw:
        return cube
    subset = cube.extract(iris.Constraint(**constraint_kw))
    assert subset is not None, (
        f"{cube.name()!r} couldn't be subset onto {reference.name()!r}'s grid along "
        f"{sorted(constraint_kw)} - no overlapping points found."
    )
    for name in constraint_kw:
        np.testing.assert_array_equal(subset.coord(name).points, reference.coord(name).points)
    return subset


def _aligned_tau_adv(cfg, adv_kind, reference, conversion=SpeciesConversion.CH4_CO, base_dir=None):
    """tau_adv_<adv_kind> for `cfg`, subset onto `reference`'s own grid
    via subset_to_grid_of() - the load+align step quench_level() shares
    with quench_level_zahnle_ls()/quench_level_zahnle_qn().

    What: `adv_kind="qn"` derives `cfg_equ` (same planet/mdh/tK,
    `exp="equ"`) internally instead of using `cfg` directly.

    Why: tau_adv_qn only ever exists for an "equ" run (see
    quench_level()'s own docstring for why mixing a "kin" cfg's own
    tau_<conversion> with its "equ" sibling's tau_adv_qn is
    deliberate).

    What: `conversion.source` picks which saved tau_adv_qn file to load
    - `conversion` itself is ignored for `adv_kind="ls"`, which has no
    conversion of its own. tau_adv_qn's own on-disk storage
    (quenchpath.io.save_tau_adv()/load_tau_adv()) only knows about a
    plain species string, not SpeciesConversion - this is the one place
    that bridges the two.

    Why bridge it here: CH4's and CO's tau_adv_qn are genuinely
    different data, not the same quantity twice (see
    quenchpath.stages.stage_07_taus.calc_and_save_adv_qn()'s own
    docstring), and this module is the one that
    genuinely operates in terms of `conversion`.

    What: subsetting happens here (a no-op wherever `reference` already
    matches, e.g. at GridResolution.FULL, or an already pre-shaped
    bundled example - see scripts/generate_example_data.py). `adv_kind`
    other than "ls"/"qn" raises ValueError.

    Why: neither adv_kind has a genuine SAMPLE variant of its own
    (always saved on the full native grid)."""
    if adv_kind == "ls":
        tau_adv = load_tau_adv(cfg, "ls", base_dir=base_dir)["tau_adv_ls_dom"]
    elif adv_kind == "qn":
        conversion = SpeciesConversion(conversion)
        cfg_equ = RunConfig(planet=cfg.planet, exp="equ", mdh=cfg.mdh, tK=cfg.tK)
        tau_adv = load_tau_adv(cfg_equ, "qn", species=conversion.source, base_dir=base_dir)[
            "tau_adv_qn"
        ]
        # What: stamp conversion.source into the cube's own name (it
        # loads back as the species-agnostic "tau_adv_qn" - see
        # io.save_tau_adv()'s own docstring for why the on-disk var_name
        # doesn't carry species, unlike the filename/directory it's in).
        # Why: gives every caller here (quench_level(),
        # quench_level_zahnle_qn()) something concrete to assert against
        # - a species mismatch fails loudly there instead of silently
        # returning a wrong ratio.
        tau_adv.rename(f"tau_adv_qn_{conversion.source.lower()}")
    else:
        raise ValueError(f"adv_kind must be 'ls' or 'qn', got {adv_kind!r}")
    return subset_to_grid_of(tau_adv, reference)


def quench_level(
    cfg, grid_resolution, chem_kind, adv_kind, conversion=SpeciesConversion.CH4_CO, base_dir=None
):
    """tau_<conversion>_<chem_kind> / tau_adv_<adv_kind> - the "quench
    level" ratio Cube: wherever it crosses 1,
    quenchpath.viz.quench_level_contour_panels() draws it as a contour.

    Parameters
    ----------
    cfg: quenchpath.config.RunConfig
    grid_resolution: quenchpath.config.GridResolution
        Which already-merged pathway-search grid to use - quenchpath
        merge-paths must already have been run for this
        cfg/grid_resolution/conversion.
    chem_kind: str
        "rls" (rate-limiting/bottleneck-segment pathway timescale) or
        "net" (whole-path-summed pathway timescale). Anything else
        raises AssertionError, via load_tau_chem()/tau_chem_dir()'s own
        check - not duplicated here.
    adv_kind: str
        "ls" (length-scale - tau_adv_ls_dom, works from either a "kin"
        or "equ" run, no conversion of its own) or "qn" (quenching-
        limited - tau_adv_qn, "equ"-only, computed from `conversion.
        source`'s own mole fraction - see _aligned_tau_adv()'s own
        docstring for why mixing a "kin" cfg's tau_<conversion> with an
        "equ" tau_adv_qn is deliberate). Anything else raises ValueError.
    conversion: quenchpath.config.SpeciesConversion, optional
        Also picks which conversion's tau_adv_qn to load when
        `adv_kind="qn"` (see _aligned_tau_adv()'s own docstring) - both
        conversions are genuinely supported; there's no restriction to
        check here.
    base_dir: optional
        See quenchpath.io.merged_paths_dir().

    Returns
    -------
    iris.cube.Cube: tau_<conversion>_<chem_kind> divided by
    tau_adv_<adv_kind>, both already loaded and aligned onto the same
    grid.
    """
    if adv_kind not in _QUENCH_LEVEL_ADV_KINDS:
        raise ValueError(f"adv_kind must be one of {_QUENCH_LEVEL_ADV_KINDS}, got {adv_kind!r}")
    conversion = SpeciesConversion(conversion)

    tau_chem = load_tau_chem(
        cfg, grid_resolution, chem_kind, conversion=conversion, base_dir=base_dir
    )
    tau_adv = _aligned_tau_adv(cfg, adv_kind, tau_chem, conversion=conversion, base_dir=base_dir)
    # What: for adv_kind="qn", assert tau_adv is genuinely the species
    # this call asked for (see _aligned_tau_adv()'s own species-stamped
    # rename).
    # Why: `conversion` is threaded through two separate calls above
    # (load_tau_chem(), _aligned_tau_adv()) - this catches a future
    # refactor that breaks that threading (the exact bug GitHub issue
    # #22 found in quench_level_zahnle_qn(), fixed there too now)
    # before it silently returns a species-mismatched ratio.
    if adv_kind == "qn":
        expected_name = f"tau_adv_qn_{conversion.source.lower()}"
        assert tau_adv.name() == expected_name, (
            f"expected {expected_name!r}, got {tau_adv.name()!r} - tau_adv_qn doesn't match "
            f"conversion={conversion.value!r}'s own source species."
        )
    return tau_chem / tau_adv


def quench_level_zahnle_ls(cfg, pres, temp, m=1.0, base_dir=None):
    """zahnle_marley_2014_tco(pres, temp, m) / tau_adv_ls_dom - Zahnle &
    Marley (2014)'s literature quench timescale compared against the
    length-scale advection timescale, in place of tau_<conversion>_<rls|net>
    (see quenchpath.analysis.zahnle_marley_2014's own module
    docstring for why).

    What: `pres`/`temp` - already-loaded air_pressure/air_temperature
    Cubes, whatever grid they're on (e.g.
    quenchpath.io.load_merged_paths()'s own `pres_cube`/`_temp_cube`).
    No separate I/O happens here, and no separate `grid_resolution`
    argument either - `pres`/`temp`'s own grid is all that's aligned
    onto via _aligned_tau_adv() here.

    Why no separate I/O/grid_resolution: unlike tau_<conversion>_rls/net,
    there's no saved-to-disk quantity to load these from, and they're
    typically already in scope wherever this is called from.

    What: `m` - Zahnle & Marley's own metallicity-relative-to-solar,
    not `cfg.mdh`'s [M/H] dex (see zahnle_marley_2014_tco()'s own
    docstring). `cfg`/`base_dir` - only used to load tau_adv_ls_dom."""
    tau_zahnle = zahnle_marley_2014_tco(pres, temp, m=m)
    tau_adv = _aligned_tau_adv(cfg, "ls", tau_zahnle, base_dir=base_dir)
    return tau_zahnle / tau_adv


def quench_level_zahnle_qn(
    cfg, pres, temp, m=1.0, conversion=SpeciesConversion.CO_CH4, base_dir=None
):
    """zahnle_marley_2014_tco(pres, temp, m) / tau_adv_qn - Zahnle &
    Marley (2014)'s literature quench timescale compared against the
    quenching-limited advection timescale (an "equ" quantity - see
    _aligned_tau_adv()'s own docstring for why), in place of
    tau_<conversion>_<rls|net>. `pres`/`temp`/`m` - see
    quench_level_zahnle_ls()'s own docstring. `cfg`/`base_dir` - only
    used to load tau_adv_qn.

    What: `conversion` - unlike quench_level_zahnle_ls() (whose
    "ls" tau_adv has no conversion of its own - see
    _aligned_tau_adv()'s own docstring), tau_adv_qn genuinely is
    computed per-species, so this needs its own `conversion` to pick
    the right one (`conversion.source`'s own mole fraction - same
    bridge quench_level() itself uses). Defaults to CO_CH4, not
    CH4_CO/quench_level()'s own default.

    Why CO_CH4: Zahnle & Marley's own t_CO fit is calibrated to
    CO->CH4 quench points specifically (see
    quenchpath.config.SpeciesConversion's own docstring - GitHub issue
    #15) - CO_CH4 is the scientifically matched comparison for this
    function, not just an option alongside CH4_CO."""
    conversion = SpeciesConversion(conversion)
    tau_zahnle = zahnle_marley_2014_tco(pres, temp, m=m)
    tau_adv = _aligned_tau_adv(cfg, "qn", tau_zahnle, conversion=conversion, base_dir=base_dir)
    expected_name = f"tau_adv_qn_{conversion.source.lower()}"
    assert tau_adv.name() == expected_name, (
        f"expected {expected_name!r}, got {tau_adv.name()!r} - tau_adv_qn doesn't match "
        f"conversion={conversion.value!r}'s own source species."
    )
    return tau_zahnle / tau_adv
