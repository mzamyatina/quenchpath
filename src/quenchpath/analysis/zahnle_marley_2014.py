"""What: literature quench-timescale schemes, for comparison against
quenchpath's own pathway-derived tau_co_ch4_<rls|net> directly; pairing
against tau_ch4_co_<rls|net> instead is only valid assuming CH4<->CO
symmetry, not a like-for-like comparison on its own terms.

Why tau_co_ch4_<rls|net> is the direct comparison: Zahnle & Marley's
own fit is to CO->CH4 quench points specifically (see
quench_level_zahnle_qn()'s own docstring/quenchpath-apps issue #15).

Currently just Zahnle & Marley (2014, ApJ 797, 41; arXiv:1408.6283)
Eqs 12-14, the CO<->CH4 system timescale fit widely used as a
single-formula quench scheme in 1D RCE/retrieval modelling. Pure
functions, no I/O - callers load pres/temp themselves (e.g. from
quenchpath.io.load_merged_paths()'s own already-loaded
_temp_cube/pres_cube) and feed the result into
quenchpath.analysis.quench_level's quench_level_zahnle_ls()/
quench_level_zahnle_qn(), the same way tau_<conversion>_rls/net feed
the existing 4 quench_level_*() functions.

Why no I/O here: callers typically already have pres/temp in scope
(e.g. from load_merged_paths()), so no separate raw-run reload is
needed.
"""

import warnings

import numpy as np
from cf_units import Unit

# What: Zahnle & Marley's own stated bound - quenching in the CO-CH4
# system always took place below this temperature across their whole
# ensemble of self-luminous models (their Eq 9's surrounding text).
# Why it matters: the fit was never exercised above it, so a grid
# location this hot has no genuine quench-point analogue in their
# calibration data, whatever tCO happens to evaluate to there.
_VALID_TEMP_MAX_K = 2200.0

# What: the only two metallicities Zahnle & Marley actually fit
# t_q1/t_q2 against (their Eq 12's surrounding text).
# Why it matters: m**-0.7 happily evaluates at any m, but there's no
# calibration data backing it away from these.
_VALID_METALLICITIES = (1.0, 3.0)


def zahnle_marley_2014_tco(pres, temp, m=1.0):
    r"""CO<->CH4 system timescale, Zahnle & Marley (2014) Eqs 12-14 - an
    emergent quench timescale fit to their whole 1D-kinetics ensemble
    (self-luminous brown dwarfs/EGPs, 500-1100 K effective temperature,
    CO->CH4 direction), not a single rate-limiting reaction's own
    timescale - see this module's own docstring for how it's meant to
    be used here.

    .. math::
        t_{q1} = 1.5\times10^{-6}\,p^{-1}m^{-0.7}\exp(42000/T)\;\mathrm{s}

        t_{q2} = 40\,p^{-2}\exp(25000/T)\;\mathrm{s}

        t_{CO} = \left(t_{q1}^{-1} + t_{q2}^{-1}\right)^{-1}

    Parameters
    ----------
    pres: iris.cube.Cube
        Air pressure. Any convertible-to-bar unit (converted here, not
        assumed already bar).
    temp: iris.cube.Cube
        Air temperature [K].
    m: float
        Metallicity relative to solar (Zahnle & Marley's own `m`, not
        quenchpath's own [M/H]-dex `cfg.mdh` - convert via
        `m = 10 ** float(cfg.mdh)` if comparing against a non-solar
        quenchpath run). Their fit was calibrated at m=1 and m=3 only;
        m=1 (solar, the default here) needs no metallicity correction
        at all (`m ** -0.7 == 1`).

    Returns
    -------
    iris.cube.Cube
        `t_CO` [s], on `pres`'s own grid/coordinates.
    """
    if not any(np.isclose(m, valid) for valid in _VALID_METALLICITIES):
        warnings.warn(
            f"zahnle_marley_2014_tco() called with m={m}, but Zahnle & "
            f"Marley (2014) only fit t_q1/t_q2 at m in {_VALID_METALLICITIES} - "
            "treat this as an extrapolation, not a calibrated result.",
            stacklevel=2,
        )

    p_bar = pres.copy()
    p_bar.convert_units("bar")
    p = p_bar.core_data()
    T = temp.core_data()

    t_q1 = 1.5e-6 * p**-1 * m**-0.7 * np.exp(42000.0 / T)  # Eq 12
    t_q2 = 40.0 * p**-2 * np.exp(25000.0 / T)  # Eq 13
    tco = 1.0 / (1.0 / t_q1 + 1.0 / t_q2)  # Eq 14

    # What: re-wrap as a Cube, inheriting pres's coords/metadata (tco
    # is plain numpy above, from p/T) - same pattern as
    # quenchpath.chemistry.formulas.k_kooij()'s own.
    # Why: so .rename()/downstream cube ops work.
    tco = pres.copy(data=tco)
    tco.rename("zahnle_marley_2014_tco")
    tco.units = Unit("s")
    return tco


def zahnle_marley_2014_invalid_mask(temp):
    """What: boolean Cube, True wherever `temp` falls outside Zahnle &
    Marley (2014)'s own calibration envelope (local T >= 2200 K - see
    this module's own `_VALID_TEMP_MAX_K` for why) - i.e. where
    `zahnle_marley_2014_tco()` is being extrapolated beyond any quench
    point their ensemble ever produced. Meant for a hatched/transparent
    overlay on the same panel `zahnle_marley_2014_tco()`-derived
    contours are drawn on, not for masking `zahnle_marley_2014_tco()`'s
    own return value.

    Why not used for masking: `zahnle_marley_2014_tco()` keeps
    returning a real number everywhere, flagged or not.

    What: `m` isn't checked here (only T is a per-gridpoint quantity -
    m is one scalar per call, fixed for the whole field).

    Why: `zahnle_marley_2014_tco()` itself warns, once per call, when
    `m` is far from the two values Zahnle & Marley actually fit (1 and
    3), rather than through this mask.
    """
    invalid = temp.copy(data=temp.core_data() >= _VALID_TEMP_MAX_K)
    invalid.rename("zahnle_marley_2014_invalid_mask")
    invalid.units = Unit("1")
    return invalid
