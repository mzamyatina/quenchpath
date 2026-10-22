"""Stage 07 (calculation half only - see quenchpath.io.save_tau_adv()):
advection and quenching timescales for a planet/experiment.

What: u/v are on the model's staggered wind (Arakawa C) grid - offset
from w/temp/pres/mf's shared native grid both vertically and
horizontally (longitude for u, latitude for v; `mf` is whichever
species' mole fraction - CH4 by default, CO for a CO_CH4 conversion,
see _load_raw_run_fields()). That's the one relevelling this stage
can't avoid (quenchpath-apps issue #11): u/v (and mf's own ln() derivatives, regridded
back onto the shared grid to undo float drift picked up from
d_dlambda()/d_dphi()'s internal unit-conversion side effects) get
regridded onto temp's grid; w/temp/pres/mf themselves stay untouched.

Why: so everything ends up on stage 03/05's own native grid - no
second relevel needed to combine tau_adv_qn with tau_ch4.

What: `tau_adv_ls_u`/`tau_adv_ls_v`/`tau_adv_ls_w`/`tau_adv_ls_dom` are
day-side *spatial means* - genuinely uniform across lat/lon, reported
on level_height.

Why level_height: the same shared axis as everything else here, so
combining them with e.g. tau_ch4 needs no further interpolation.

What: no pressure anywhere in this module - level_height is the only
vertical axis, native and unambiguous. A pressure *label* for that
axis (e.g. for a plot's y-axis) is a plotting-time concern: whichever
domain/mean is right for one particular figure - day-side to match
tau_adv_ls_*, unaveraged to match tau_adv_qn's own per-location
values, or something else entirely.

Why left to plotting time: this module has no way to make that choice
once and for all, so it doesn't try.
"""

import cf_units
import iris
import iris.analysis
import iris.coords
import numpy as np
from aeolus.calc import spatial_mean
from aeolus.const import init_const
from aeolus.model import um
from iris.analysis import Linear
from iris.analysis.calculus import _coord_cos, differentiate

from quenchpath import io
from quenchpath.analysis.quench_level import subset_to_grid_of
from quenchpath.config import GridResolution, SpeciesConversion
from quenchpath.mypaths import path_to_run_metadata
from quenchpath.stages import assert_mole_fraction, assert_number_density

_MOLAR_GAS_CONST = init_const().molar_gas_constant


def deriv(cube, coord):
    """Calculate a derivative w.r.t. the given coordinate.

    Uses `iris.analysis.calculus.differentiate` and then interpolates the
    result to the grid points of the original cube.

    Parameters
    ----------
    cube: iris.cube.Cube
        Input cube containing the given coordinate.
    coord: str or iris.coords.Coord
        Coordinate for differentiation.

    Returns
    -------
    iris.cube.Cube
        d(cube)/d(coord).
    """
    pnts = cube.coord(coord).points
    diff = differentiate(cube, coord)
    return diff.interpolate([(coord, pnts)], Linear())


def d_dphi(cube, const, model=um):
    r"""Calculate a derivative w.r.t. latitude.

    .. math::
        \frac{1}{r}\frac{\partial A}{\partial \phi}

    Parameters
    ----------
    cube: iris.cube.Cube
        Field A to differentiate; must have `model.y` (latitude) as a
        coordinate.
    const: aeolus.const.const.ConstContainer
        Planet constants (radius r [m]).
    model: aeolus.model.Model, optional
        Coordinate-name mapping (x/y/z). Defaults to `aeolus.model.um`.

    Returns
    -------
    iris.cube.Cube
        dA/dphi [A's units per radian, divided by planet radius],
        interpolated back onto `cube`'s own latitude points.
    """
    cube.coord(model.y).convert_units("radians")
    out = deriv(cube, model.y) / const.planet_radius
    # What: differentiate() differentiates w.r.t. the coordinate's own
    # units (radians here), leaving a spurious "per radian" factor in
    # out.units - this multiplies it back out.
    #
    # Why: radian is dimensionless (1 rad = 1, exactly - see UDUNITS),
    # so leaving that factor in would make out's units unable to
    # add/subtract cleanly against a cube that never went through a
    # radians coordinate (d_dz()'s own output, e.g.) even though
    # they're physically the same dimension.
    #
    # Why convert_units(), not a bare `out.units = ...` reassignment:
    # so iris rescales the data by whatever the (here always exactly
    # 1.0) conversion factor is, instead of silently mismatching data
    # and metadata.
    out.convert_units(out.units * cf_units.Unit("radian"))
    cube.coord(model.y).convert_units("degrees")
    out.coord(model.y).convert_units("degrees")
    return out


def d_dlambda(cube, const, model=um):
    r"""Calculate a derivative w.r.t. longitude.

    .. math::
        \frac{1}{r\cos\phi}\frac{\partial A}{\partial \lambda}

    Parameters
    ----------
    cube: iris.cube.Cube
        Field A to differentiate; must have `model.x`/`model.y`
        (longitude/latitude) as coordinates.
    const: aeolus.const.const.ConstContainer
        Planet constants (radius r [m]).
    model: aeolus.model.Model, optional
        Coordinate-name mapping (x/y/z). Defaults to `aeolus.model.um`.

    Returns
    -------
    iris.cube.Cube
        dA/dlambda [A's units per radian, divided by r*cos(phi)],
        interpolated back onto `cube`'s own longitude points.
    """
    xcoord = cube.coord(model.x).copy()
    ycoord = cube.coord(model.y).copy()
    cube.coord(model.x).convert_units("radians")
    cube.coord(model.y).convert_units("radians")
    out = (
        iris.analysis.maths.divide(
            deriv(cube, model.x), _coord_cos(cube.coord(model.y)), dim=cube.coord_dims(model.y)[0]
        )
        / const.planet_radius
    )
    # Same spurious "per radian" leftover (from differentiating w.r.t.
    # model.x, in radians) as d_dphi()'s own - see its comment on this
    # same convert_units() call for why.
    out.convert_units(out.units * cf_units.Unit("radian"))
    cube.coord(model.x).convert_units("degrees")
    cube.coord(model.y).convert_units("degrees")
    cube.replace_coord(xcoord)
    cube.replace_coord(ycoord)
    out.coord(model.x).convert_units("degrees")
    out.coord(model.y).convert_units("degrees")
    out.replace_coord(xcoord)
    out.replace_coord(ycoord)
    return out


def d_dz(cube, const, model=um):
    r"""Calculate a derivative w.r.t. height.

    .. math::
        \frac{\partial A}{\partial z}

    Parameters
    ----------
    cube: iris.cube.Cube
        Field A to differentiate; must have `model.z` (height) as a
        coordinate.
    const: aeolus.const.const.ConstContainer
        Unused - present only so d_dz() has the same signature as
        d_dphi()/d_dlambda() (all three are called uniformly in
        calc_adv_qn()).
    model: aeolus.model.Model, optional
        Coordinate-name mapping (x/y/z). Defaults to `aeolus.model.um`.

    Returns
    -------
    iris.cube.Cube
        dA/dz [A's units per m].
    """
    return deriv(cube, model.z)


def regrid_to_shared_grid(cube, *, target):
    """Regrid/interpolate `cube` onto `target`'s own grid - the exact
    shared native grid `_load_raw_run_fields()` puts u/v/tau_adv_ls_*/
    tau_adv_qn_* on (see this module's own docstring) -
    `cube.regrid(target, Linear())`, then interpolated onto `target`'s
    own vertical coordinate's points.

    What: `target` is purely a grid reference here, not a physical
    input - named generically (not e.g. "temp").

    Why generic naming: regridding onto its grid is the only thing this
    function does with it; whatever cube's grid happens to be used, its
    own data never enters the computation.

    What: `_load_raw_run_fields()` happens to always pass its own
    `temp` cube for this (`raw = io.load_raw_run(cfg)`, `temp =
    raw.extract_cube("air_temperature")`).

    Why temp specifically: that's this stage's own shared-grid
    convention, not because temperature matters here.

    What: public - a no-op (still goes through the same
    regrid+interpolate, just with no actual change) for any field
    already natively on `target`'s own grid (w/pres/temp/ch4
    themselves, when `target` is `temp`) - only u/v (the staggered wind
    grid) genuinely need it.

    Why public: so a caller other than this module - a plotting stage
    wanting some other raw field (CH4, pres, ...) positioned exactly
    like the saved tau_* cubes, not just already-matching by
    coincidence - can relevel it the identical way, without
    reimplementing this regrid+interpolate pairing itself.

    Parameters
    ----------
    cube: iris.cube.Cube
        Field to relevel.
    target: iris.cube.Cube
        Regrid reference, both horizontal and vertical - only its grid
        is used, never its own data.

    Returns
    -------
    iris.cube.Cube
        `cube`, regridded/interpolated onto `target`'s own grid.
    """
    z_coord_common = target.coord(um.z)
    return cube.regrid(target, Linear()).interpolate(
        [(z_coord_common.name(), z_coord_common.points)], Linear()
    )


def calc_adv_ls(u, v, w, temp, pres, planet_const):
    r"""What: day-side, spatial-mean length-scale advection timescales:
    tau_adv_ls_u (planet circumference / |u|), tau_adv_ls_v
    (equator-to-pole quarter-circumference / |v|), tau_adv_ls_w (scale
    height / |w|), and tau_adv_ls_dom (the fastest/dominant of the
    three, per level) - plus pres_mean, the same day-side spatial mean
    applied to pressure instead of a wind/scale-height quantity.

    Why pres_mean uses the exact same day-side selection: it's not a
    plotting-time label choice but the pressure genuinely matching
    tau_adv_ls_*'s own domain - computed the same way rather than
    risking a second, independently-written version silently drifting
    onto a different domain (e.g. a plain global mean - checked
    empirically against real hatp11b data, up to ~9% relative
    difference from the day-side mean at some levels, not a rounding-
    level discrepancy).

    .. math::
        \tau_{ls}^{u} = \frac{2\pi R}{|\bar u|}, \quad
        \tau_{ls}^{v} = \frac{\pi R}{2|\bar v|}, \quad
        \tau_{ls}^{w} = \frac{H}{|\bar w|}, \quad
        H = \frac{R_{gas} T}{g M}

    Parameters
    ----------
    u, v, w: iris.cube.Cube
        Wind components [m/s], already regridded onto one shared grid
        (see _load_raw_run_fields()).
    temp: iris.cube.Cube
        Air temperature [K], on that same shared grid.
    pres: iris.cube.Cube
        Air pressure [Pa], on that same shared grid - not a physical
        input to any of the 4 timescales themselves, only reduced
        (day-side spatial mean) alongside them for pres_mean.
    planet_const: aeolus.const.const.ConstContainer
        This run's own planet+metallicity constants (radius [m], surface
        gravity [m/s2], mean molar mass of air [kg/mol]) - see
        calc_and_save_adv_ls(), which loads the
        file matching both `cfg.planet` and `cfg.mdh`.

    Returns
    -------
    (tau_adv_ls_u, tau_adv_ls_v, tau_adv_ls_w, tau_adv_ls_dom):
        All four Cubes, in seconds, on the level_height axis (a day-side
        spatial mean already collapsed latitude/longitude), each also
        carrying pres_mean (the matching day-side spatial mean pressure)
        as its aux coordinate - there's no native per-location pressure
        left to pair an already-collapsed profile with (see the module
        docstring).
    """
    # What: scale height H = R_gas*T/(g*M).
    # Why: the standard e-folding height of an isothermal, hydrostatic
    # atmosphere - R_gas is molar, M is the mean molar mass of air, so
    # this is length, not a per-mole quantity.
    scale_height = (
        _MOLAR_GAS_CONST
        * temp
        / (planet_const.planet_surface_gravity * planet_const.model_mean_molar_mass_of_air)
    )
    scale_height.convert_units("m")
    # What: value-based (not index-position-) selection. Degrees match
    # this codebase's other substellar=180deg convention (see
    # quenchpath.viz.representative_lon_indices()/LON_TITLES).
    # Why value-based: so this doesn't silently break on a
    # differently-sized/-ordered grid.
    lons_all = u.coord("longitude").points
    lats_all = u.coord("latitude").points
    day_lons = lons_all[(lons_all >= 90) & (lons_all <= 270)]  # dayside
    # What: northern (positive) latitudes.
    # Why: an arbitrary but fixed choice - tau_adv_ls_v doesn't depend
    # on which hemisphere, by symmetry.
    northern_lats = lats_all[lats_all > 0]
    assert len(day_lons) > 0 and len(northern_lats) > 0, (
        "day_lons/northern_lats selection matched nothing - check "
        "the longitude/latitude coordinate convention (0-360 degrees, "
        "substellar at 180) still holds for this raw run."
    )
    constr_day = iris.Constraint(longitude=lambda lo: lo.point in day_lons)
    constr_day_nhe = iris.Constraint(
        longitude=lambda lo: lo.point in day_lons,
        latitude=lambda la: la.point in northern_lats,
    )
    u_mean = spatial_mean(u.extract(constr_day))
    v_mean = spatial_mean(v.extract(constr_day_nhe))
    w_mean = spatial_mean(w.extract(constr_day))
    scale_height_mean = spatial_mean(scale_height.extract(constr_day))
    # What: same day-side domain as u/w/scale_height, not v's narrower
    # day+northern-hemisphere one.
    # Why: pres_mean isn't tied to any single one of the 3 length-scale
    # timescales specifically.
    pres_mean = spatial_mean(pres.extract(constr_day))
    # What: use iris.analysis.maths.abs(), not apply_ufunc(np.abs, ...).
    # Why: the latter can't tell abs() leaves units unchanged and drops
    # them to "unknown"; abs() is iris's own explicit wrapper and keeps
    # u/v/w's real units (m/s) throughout.
    u_mean_abs = iris.analysis.maths.abs(u_mean)
    v_mean_abs = iris.analysis.maths.abs(v_mean)
    w_mean_abs = iris.analysis.maths.abs(w_mean)
    # time to circle the whole planet zonally at speed |u_mean|
    tau_adv_ls_u = 2 * np.pi * planet_const.planet_radius / u_mean_abs
    # What: time to advect equator-to-pole meridionally at speed
    # |v_mean|.
    # Why pi*R/2 is correct (not pi*R): it's a quarter of the planet's
    # circumference (2*pi*R), not a half, despite np.pi alone appearing
    # in the numerator.
    tau_adv_ls_v = np.pi * planet_const.planet_radius / (2 * v_mean_abs)
    # time to advect one scale height vertically at speed |w_mean|
    tau_adv_ls_w = scale_height_mean / w_mean_abs

    # What: dominant (shortest/fastest) of the 3 length-scale
    # timescales, per level - a Cube (not a plain array) like its 3
    # siblings, inheriting tau_adv_ls_u's own coords/metadata via
    # .copy(data=...).
    #
    # Why no np.abs(): already positive.
    #
    # Why nested np.minimum(): it only takes 2 operands.
    #
    # Why .copy(data=...): only replaces the data, not the level_height
    # coord/attributes riding along with it.
    tau_adv_ls_dom = tau_adv_ls_u.copy(
        data=np.minimum(np.minimum(tau_adv_ls_u.data, tau_adv_ls_v.data), tau_adv_ls_w.data)
    )

    # What: .copy() per cube (not one shared AuxCoord instance).
    # Why: iris expects each cube to own its own coord objects.
    pres_mean_coord = iris.coords.AuxCoord(pres_mean.data, long_name="pres_mean", units="Pa")
    for cube in (tau_adv_ls_u, tau_adv_ls_v, tau_adv_ls_w, tau_adv_ls_dom):
        (dim,) = cube.coord_dims("level_height")
        cube.add_aux_coord(pres_mean_coord.copy(), dim)

    return tau_adv_ls_u, tau_adv_ls_v, tau_adv_ls_w, tau_adv_ls_dom


def calc_adv_qn(u, v, w, mf, planet_const, *, target):
    r"""Quenching-limited advection timescales: tau_adv_qn (from the full 3D
    advection of ln(X_eq) along u/v/w) and tau_adv_qn_u/tau_adv_qn_v/tau_adv_qn_w
    (each wind component's own contribution alone).

    .. math::
        \tau_{qn} = \frac{1}{|\vec{u} \cdot \nabla \ln X_{eq}|}

    i.e. the e-folding timescale of species X's own local fractional
    rate of change under advection alone, evaluated on X_eq - X's
    *equilibrium* mole fraction (not a kinetics run's already-quenched
    field - see calc_and_save_adv_qn()'s own docstring for why) - same
    X_eq/(dX_eq/dt) pattern as a chemical lifetime, applied to the
    advective term instead. Unlike tau_adv_ls (a single geometry-based
    estimate per level), this varies point to point with the real,
    unquenched-by-construction X_eq field, so it can read much faster
    *or* much slower than tau_adv_ls locally - including blowing up
    wherever X_eq happens to be locally well mixed (flat gradient),
    which is expected, not a bug.

    The formula itself is species-agnostic - X_eq is whichever
    equilibrium mole fraction `mf` holds, not necessarily CH4 (see
    quenchpath.config.SpeciesConversion.source for which species a given
    conversion actually needs this for - CH4 for CH4_CO, CO for CO_CH4 -
    and _load_raw_run_fields() for where that selection happens; this
    function itself has no CH4-specific logic anywhere in it).

    What: regridding onto `target`'s own grid explicitly, rather than
    relying on the derivatives already matching.

    Why: d_dlambda()/d_dphi() mutate their input cube's x/y coord units
    in place (restored before returning, but real - not just index
    bookkeeping), which can leave the *output* cube's coordinate
    objects subtly different from `mf`'s own (e.g. float ULP
    differences after the radians/degrees round-trip).

    Parameters
    ----------
    u, v, w: iris.cube.Cube
        Wind components [m/s], already regridded onto one shared grid
        (see _load_raw_run_fields()).
    mf: iris.cube.Cube
        *Equilibrium* mole fraction X_eq [dimensionless] of whichever
        species this timescale is being computed for (CH4 or CO - see
        _load_raw_run_fields()), native grid - not regridded itself; its
        ln() derivatives are regridded onto `target`'s own grid inside
        this function instead. Checked (via assert_mole_fraction()), not
        just assumed from the name - that check confirms it's *a* mole
        fraction, not that it's specifically the equilibrium one; the
        latter is the caller's responsibility (calc_and_save_adv_qn()
        enforces cfg.exp="equ" for exactly this reason).

        Why check it: using a number density here by mistake would
        still run (both are plain positive arrays log() accepts), but
        the physical rate this timescale represents - X_eq's own
        fractional rate of change under advection - depends
        specifically on the mole fraction gradient, not the number
        density one (number density also carries the atmosphere's own
        hydrostatic density fall-off, unrelated to quenching).
    planet_const: aeolus.const.const.ConstContainer
        Planet constants (radius [m]).
    target: iris.cube.Cube
        Keyword-only. Regrid reference for `mf`'s derivatives (the
        shared grid u/v/w are already on) - purely a grid reference,
        not a physical input. Whatever cube is passed here, only its
        grid matters, never its data.

        Why not a physical input (unlike calc_adv_ls()'s own `temp`,
        which genuinely feeds into scale height): tau_adv_qn doesn't
        depend on temperature, only on wind and X_eq.

        Why _load_raw_run_fields() passes its own `temp` cube here:
        that's the stage's shared-grid convention, not because
        temperature enters this calculation.

    Returns
    -------
    (tau_adv_qn_u, tau_adv_qn_v, tau_adv_qn_w, tau_adv_qn): all Cubes, in seconds.
    """
    assert_mole_fraction(mf)
    # What: log, not the raw mole fraction.
    # Why: makes the gradient a *fractional* rate of change
    # (d(ln X_eq)/dl = (dX_eq/dl)/X_eq), which is what the
    # X_eq/(dX_eq/dt) timescale definition below needs - a plain
    # dX_eq/dl would instead be dominated by X_eq's steep hydrostatic
    # fall-off with height, unrelated to quenching.
    #
    # What: new_unit="1" (not iris.analysis.maths.log()'s own default
    # "ln(re 1)" logarithmic unit).
    # Why: ln(X_eq) of a dimensionless X_eq is itself plain dimensionless
    # here, used only as an intermediate for the fractional-gradient
    # math below; cf_units's logarithmic "ln(re 1)" unit can't even be
    # divided/multiplied against d_dlambda()'s own metric factors (a
    # hard ValueError), so it's actively the wrong unit to carry
    # forward, not just unhelpfully verbose.
    ln_mf = iris.analysis.maths.apply_ufunc(np.log, mf, new_unit="1")
    d_ln_mf_dx = regrid_to_shared_grid(d_dlambda(ln_mf, const=planet_const), target=target)
    d_ln_mf_dy = regrid_to_shared_grid(d_dphi(ln_mf, const=planet_const), target=target)
    d_ln_mf_dz = regrid_to_shared_grid(d_dz(ln_mf, const=planet_const), target=target)
    # What: u.grad(ln X_eq) - the full advective derivative in
    # spherical+height coordinates, a plain dot product. Each wind
    # component's own contribution (contrib_u/v/w) computed once and
    # reused below for tau_adv_qn_u/v/w, rather than redone.
    # Why a plain dot product, not a raw sum of partials: d_dlambda/
    # d_dphi already carry the 1/(r cos(phi)) and 1/r metric factors.
    contrib_u = u * d_ln_mf_dx
    contrib_v = v * d_ln_mf_dy
    contrib_w = w * d_ln_mf_dz
    den = contrib_u + contrib_v + contrib_w
    # What: iris.analysis.maths.abs(), not apply_ufunc(np.abs, ...) -
    # see calc_adv_ls()'s own comment on the same substitution.
    # Why: keeps den/contrib_u/v/w's real units (1/s, from ln_mf's
    # now-dimensionless gradient times u/v/w's m/s) instead of dropping
    # to "unknown".
    tau_adv_qn = 1 / iris.analysis.maths.abs(den)
    # What: each wind component's own contribution alone, not summands
    # of tau_adv_qn (1/sum(|x_i|) != sum(1/|x_i|)).
    # Why: useful for seeing which component dominates locally, not for
    # arithmetic recombination.
    tau_adv_qn_u = 1 / iris.analysis.maths.abs(contrib_u)
    tau_adv_qn_v = 1 / iris.analysis.maths.abs(contrib_v)
    tau_adv_qn_w = 1 / iris.analysis.maths.abs(contrib_w)

    return tau_adv_qn_u, tau_adv_qn_v, tau_adv_qn_w, tau_adv_qn


def _load_raw_run_fields(cfg, species="CH4"):
    """I/O + shared-grid setup common to calc_and_save_adv_ls()/
    calc_and_save_adv_qn(): load `cfg`'s raw run and planet constants,
    and regrid u/v onto the shared native grid (w/pres/temp/mf's own
    grid already - see the module docstring) - the one relevel this
    stage can't avoid. target=temp here is a grid reference only, not
    a physical input - see regrid_to_shared_grid()/calc_adv_qn()'s own
    docstrings.

    What: `species` picks which species' mole fraction `mf` is - only
    meaningful to calc_and_save_adv_qn()'s own caller. Deliberately a
    plain species string, not a
    quenchpath.config.SpeciesConversion - see tau_adv_qn's own on-disk
    naming (quenchpath.io.tau_adv_filename()) for why tau_adv_qn is
    independent of that concept.

    Why harmless for the other caller: calc_and_save_adv_ls() discards
    `mf` entirely, so its own default-CH4 load is harmless dead
    weight, not a real dependency.

    Returns
    -------
    (u, v, w, temp, pres, mf, planet_const): mf is `species`'s *mole
    fraction* (checked via assert_mole_fraction(), not
    quenchpath.stages.stage_03_convert's number density output) - the
    raw run's own native field.
    """
    planet_const = init_const(f"{cfg.planet}_mdh{cfg.mdh}", directory=path_to_run_metadata)
    raw = io.load_raw_run(cfg)
    u = raw.extract_cube("x_wind")
    v = raw.extract_cube("y_wind")
    w = raw.extract_cube("upward_air_velocity")
    pres = raw.extract_cube("air_pressure")
    temp = raw.extract_cube("air_temperature")
    mf = raw.extract_cube(f"{species} mole fraction")
    assert_mole_fraction(mf)
    u = regrid_to_shared_grid(u, target=temp)
    v = regrid_to_shared_grid(v, target=temp)
    return u, v, w, temp, pres, mf, planet_const


def calc_and_save_adv_ls(cfg):
    """Compute `cfg`'s length-scale advection timescales and save them
    to a single .nc file (see quenchpath.io.save_tau_adv()) - see
    calc_adv_ls() for the timescale itself; this is the I/O +
    shared-grid-setup driver around it.

    What: unlike calc_and_save_adv_qn(), `cfg.exp` can be either "kin"
    or "equ".
    Why: tau_adv_ls only needs wind/temperature, present in both.

    Returns a dict of 4 Cubes [s] (tau_adv_ls_u/tau_adv_ls_v/tau_adv_ls_w/
    tau_adv_ls_dom) - see calc_adv_ls() for what each one carries
    (level_height as its own dim coordinate, pres_mean as its aux
    coordinate).
    """
    u, v, w, temp, pres, _mf, planet_const = _load_raw_run_fields(cfg)

    tau_adv_ls_u, tau_adv_ls_v, tau_adv_ls_w, tau_adv_ls_dom = calc_adv_ls(
        u, v, w, temp, pres, planet_const
    )
    # What: explicit convert_units("s") anyway, even though already
    # convertible to "s" coming out of calc_adv_ls() (a real
    # conversion, not a no-op fix-up - see d_dphi()/d_dlambda()'s own
    # comments on the "per radian" unit bookkeeping this depends on).
    # Why: matches tau_ch4_co_rls/net's own pattern below, so every
    # saved tau_* cube's units are the literal same "s", not just
    # something convertible to it.
    for cube in (tau_adv_ls_u, tau_adv_ls_v, tau_adv_ls_w, tau_adv_ls_dom):
        cube.convert_units("s")
    result = {
        "tau_adv_ls_u": tau_adv_ls_u,
        "tau_adv_ls_v": tau_adv_ls_v,
        "tau_adv_ls_w": tau_adv_ls_w,
        "tau_adv_ls_dom": tau_adv_ls_dom,
    }

    io.save_tau_adv(result, cfg, "ls")
    return result


def calc_and_save_adv_qn(cfg, species="CH4"):
    """Compute `cfg`'s quenching-limited advection timescales and save
    them to a single .nc file (see quenchpath.io.save_tau_adv()) - see
    calc_adv_qn() for the timescale itself; this is the I/O +
    shared-grid-setup driver around it.

    What: `cfg.exp` must be "equ".
    Why: tau_adv_qn asks how fast advection would quench a location's
    *equilibrium* field, so it needs that field's own
    (unquenched-by-construction) gradient, not a kinetics run's
    (already shaped by the reaction network this timescale is meant to
    be compared against - using it here would be circular).

    What: `species` (default "CH4", reproducing every already-saved
    tau_adv_qn file's own data) picks which species' mole fraction (see
    _load_raw_run_fields()) tau_adv_qn is computed from. Deliberately a
    plain species string, not a quenchpath.config.SpeciesConversion -
    see quenchpath.io.tau_adv_filename()'s own docstring for why
    tau_adv_qn is independent of that concept; a caller that has a
    `conversion` instead (e.g. quenchpath.analysis.quench_level) passes
    `conversion.source` here.

    Why species matters: the underlying formula (calc_adv_qn()) has no
    species-specific logic of its own - a caller wanting this for CO's
    own pathway timescale genuinely needs CO's own gradient here, not
    CH4's, for the same physical reason calc_chem_timescale()
    multiplies by `conversion.source`'s own number density rather than
    always CH4's.

    Returns
    -------
    dict of 4 Cubes [s] (tau_adv_qn_u/tau_adv_qn_v/tau_adv_qn_w/
    tau_adv_qn, on the shared native grid - see the module docstring),
    each carrying only level_height as its own dim coordinate - no pres
    aux coordinate.

    Why no pres aux coordinate (unlike tau_adv_ls's pres_mean, a genuine
    reduction computed here alongside tau_adv_ls's own spatial
    averaging - there's no native per-location pressure left to fall
    back on once you've collapsed the horizontal dimensions): pres here
    would just be an unmodified passthrough of the raw run's own
    air_pressure cube - trivially reloadable byte-identically at
    plotting time (`io.load_raw_run(cfg).extract_cube("air_pressure")`)
    rather than carried along as dead weight with no current reader.
    """
    assert cfg.exp == "equ", (
        f"calc_and_save_adv_qn() needs an equilibrium-chemistry run "
        f"(cfg.exp='equ'), not exp={cfg.exp!r} - see this function's own docstring for why."
    )
    u, v, w, temp, _pres, mf, planet_const = _load_raw_run_fields(cfg, species)

    tau_adv_qn_u, tau_adv_qn_v, tau_adv_qn_w, tau_adv_qn = calc_adv_qn(
        u, v, w, mf, planet_const, target=temp
    )
    # See calc_and_save_adv_ls()'s own comment on this same
    # convert_units() call.
    for cube in (tau_adv_qn_u, tau_adv_qn_v, tau_adv_qn_w, tau_adv_qn):
        cube.convert_units("s")
    result = {
        "tau_adv_qn_u": tau_adv_qn_u,
        "tau_adv_qn_v": tau_adv_qn_v,
        "tau_adv_qn_w": tau_adv_qn_w,
        "tau_adv_qn": tau_adv_qn,
    }

    io.save_tau_adv(result, cfg, "qn", species=species)
    return result


def _load_chem_rate_cubes(cfg, grid_resolution, conversion=SpeciesConversion.CH4_CO):
    """I/O + subsetting shared by calc_chem_timescale()'s two
    `chem_kind`s: quenchpath merge-paths' saved
    net_rate_inv/max_rate_inv cubes (quenchpath.graph.shortest_path.
    find_pathway()'s own inverse-rate, [cm3 s]), plus the pathway's
    *source* species' number density (quenchpath.stages.stage_03_convert's
    saved output, [cm-3] - CH4 for CH4-to-CO, CO for CO-to-CH4, see
    quenchpath.config.SpeciesConversion.source) already aligned onto
    net_rate_inv_cube's own grid via
    quenchpath.analysis.quench_level.subset_to_grid_of() - a no-op at
    GridResolution.FULL, a genuine subset at SAMPLE.

    Parameters
    ----------
    cfg: quenchpath.config.RunConfig
        `cfg.exp` must be "kin" - the pathway/rate-network search this
        reads from has no equilibrium-chemistry equivalent.
    grid_resolution: quenchpath.config.GridResolution
        Which already-merged pathway-search grid to use - quenchpath
        merge-paths must already have been run for this cfg/resolution.
    conversion: quenchpath.config.SpeciesConversion, optional
        Which already-merged pathway-search *conversion* to use (must
        also already have been run for this cfg/resolution/conversion),
        and which species' number density to load - see above. tau =
        [source species]/R is the actual physical requirement (not
        specifically CH4) - see calc_chem_timescale()'s own
        docstring for why the two directions' timescales genuinely
        differ by more than a relabeling.

    Returns
    -------
    (net_rate_inv_cube, max_rate_inv_cube, source_nd): all Cubes on the
    merged pathway grid (level_height, latitude, longitude). source_nd
    is `conversion.source`'s number density (checked via
    assert_number_density(species=conversion.source)) - the right
    quantity to multiply a rate_inv cube by, unlike the mole fraction
    _load_raw_run_fields() loads for tau_adv_qn.
    """
    conversion = SpeciesConversion(conversion)
    assert cfg.exp == "kin", (
        f"{conversion.label} pathway timescales need a kinetics run (cfg.exp='kin'), not "
        f"exp={cfg.exp!r} - the pathway/rate-network search has no equilibrium-chemistry "
        f"equivalent."
    )
    grid_resolution = GridResolution(grid_resolution)
    _, _, _, net_rate_inv_cube, max_rate_inv_cube, _ = io.load_merged_paths(
        cfg, grid_resolution, conversion=conversion
    )
    source_nd = io.load_converted(cfg).extract_cube(f"{conversion.source} number density")
    assert_number_density(source_nd, species=conversion.source)
    source_nd = subset_to_grid_of(source_nd, net_rate_inv_cube)
    return net_rate_inv_cube, max_rate_inv_cube, source_nd


def calc_chem_timescale(cfg, grid_resolution, chem_kind, conversion=SpeciesConversion.CH4_CO):
    r"""Pathway timescale - "rls" (rate-limiting-step: the pathway's
    single slowest/bottleneck segment's inverse rate, quenchpath
    merge-paths' saved max_rate_inv) or "net" (whole-path summed,
    quenchpath merge-paths' saved net_rate_inv - dijkstra's own reported
    total across every segment, what path-finding itself minimizes) -
    converted to a genuine timescale [s] by multiplying by
    `conversion.source`'s number density - the two units ([cm3 s] and
    [cm-3]) cancel exactly, see
    quenchpath.graph.shortest_path.find_pathway()'s own docstring.

    .. math::
        \tau_{X-to-Y}^{RLS} = \tau_{max\_rate\_inv} \cdot n_{X}

        \tau_{X-to-Y}^{net} = \tau_{net\_rate\_inv} \cdot n_{X}

    (X, Y = CH4, CO for the default CH4-to-CO conversion; CO, CH4 for
    CO-to-CH4.)

    Why not just a relabeling: tau = [source species]/R is the genuine
    physical definition, so the CO-to-CH4 timescale is scaled by the
    CO/CH4 number-density ratio relative to the CH4-to-CO one, on top of
    using CO-to-CH4's own (generally different) pathway/rate (see
    quenchpath-apps issue #15).

    What: computed as genuine Cube arithmetic
    (`rate_inv_cube * source_nd`), not plain numpy arrays.

    Why: so iris's own unit algebra checks the cm3 s * cm-3 -> s
    cancellation instead of trusting it by hand. This only works
    because quenchpath.stages.stage_06_merge_paths gives its cubes real
    coordinates (inherited from a native-grid reference cube, e.g. this
    same number density cube) rather than bare ones rebuilt from
    scratch - see that module's own merge_paths_to_cubes() docstring
    for why that distinction matters to iris's cube-arithmetic Resolve
    step.

    Parameters
    ----------
    cfg: quenchpath.config.RunConfig
    grid_resolution: quenchpath.config.GridResolution
        Which already-merged pathway-search grid to use - quenchpath
        merge-paths must already have been run for this cfg/resolution.
    chem_kind: str
        "rls" or "net" - anything else raises ValueError.
    conversion: quenchpath.config.SpeciesConversion, optional
        See _load_chem_rate_cubes()'s own docstring.

    Returns
    -------
    iris.cube.Cube [s], on the merged pathway grid (level_height,
    latitude, longitude).
    """
    if chem_kind not in ("rls", "net"):
        raise ValueError(f"chem_kind must be 'rls' or 'net', got {chem_kind!r}")
    conversion = SpeciesConversion(conversion)
    net_rate_inv_cube, max_rate_inv_cube, source_nd = _load_chem_rate_cubes(
        cfg, grid_resolution, conversion
    )
    rate_inv_cube = max_rate_inv_cube if chem_kind == "rls" else net_rate_inv_cube
    tau = rate_inv_cube * source_nd
    tau.convert_units("s")
    tau.rename(f"tau_{conversion.value}_{chem_kind}")
    return tau


def calc_and_save_chem_timescale(
    cfg, grid_resolution, chem_kind, conversion=SpeciesConversion.CH4_CO
):
    """calc_chem_timescale(), saved to a single .nc file (see
    quenchpath.io.save_tau_chem())."""
    conversion = SpeciesConversion(conversion)
    tau = calc_chem_timescale(cfg, grid_resolution, chem_kind, conversion)
    io.save_tau_chem(tau, cfg, grid_resolution, chem_kind, conversion=conversion)
    return tau
