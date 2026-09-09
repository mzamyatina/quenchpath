"""Reaction-rate-coefficient and thermodynamics formulas: Kooij/Troe rate
coefficients, standard chemical potential from NASA polynomial
coefficients, and the equilibrium constant/reverse rate coefficient
derived from them - consumed by quenchpath.stages.stage_04_rates.calc_rates().
Pure functions, no I/O.
"""

import iris
import iris.cube
import numpy as np
from cf_units import Unit

from quenchpath.chemistry.network import VENOT2019


def k_kooij(temp, alpha, beta, gamma):
    r"""
    Rate coefficient of a bimolecular/decomposition reaction, modified
    Arrhenius (Kooij) equation. Adapted from
    http://kida.astrophy.u-bordeaux.fr/help.html#heading23.

    .. math::
        k(T) = \alpha \left(\frac{T}{300}\right)^{\beta}
        \exp\left(-\frac{\gamma}{T}\right)

    Parameters
    ----------
    temp: iris.cube.Cube
        Air temperature [K].
    alpha: float
        Pre-exponential factor. Units match the returned rate coefficient's
        (cm3 s-1 for a bimolecular reaction, s-1 for unimolecular
        decomposition) - not fixed, depends on which reaction this is for.
    beta: float
        Temperature dependence of the pre-exponential factor [unitless].
    gamma: float
        E/R, activation energy over the universal gas constant [K].

    Returns
    -------
        iris.cube.Cube
            Rate coefficient k(T), same units as `alpha`.
    """
    # What: use float64 here.
    # Why: float32 underflows k_inf for CH4(+M)↔CH3+H(+M).
    T = temp.core_data().astype(np.float64)

    k = (alpha * (T / 300) ** beta) * np.exp(-gamma / T)

    # What: re-wrap as a Cube, inheriting temp's coords/metadata (k is
    # plain numpy above, from T).
    # Why: so .rename()/downstream cube ops work.
    k = temp.copy(data=k)
    k.rename("k_kooij")

    return k


# Cube name -> index into the raw network's 22-entry `effcs` (enhanced
# third-body efficiency) list, for VENOT2019["bath_gases"] - the other 13
# entries are for third bodies not present here. The 22 entries and
# their fixed order (O2, CO, CO2, H2O, CH4, H2, C2H6, Ar, N2, He, C2H4,
# benzene, toluene, H2O2, N2O, O3P, NH3, N2H4, N2O4, NO2, NO, N4S) come
# from networks/venot2019/raw/README.rtf's reactions_N.dat format
# description, not from VENOT2019 itself. Order must match
# VENOT2019["bath_gases"]'s own.
_BATH_GAS_EFFC_INDICES = [1, 2, 3, 4, 5, 8, 9, 15, 16]
assert len(_BATH_GAS_EFFC_INDICES) == len(VENOT2019["bath_gases"])
_BATH_GAS_EFFC_INDEX = dict(zip(VENOT2019["bath_gases"], _BATH_GAS_EFFC_INDICES))


def _bath_gas_number_density(effcs, bath_gas_mixture):
    r"""
    Weighted third-body number density M, for a pressure-dependent
    (low-pressure-limit/Troe) reaction's bath-gas mixture: each bath
    gas's own number density times its enhanced third-body efficiency,
    summed over all 9 bath gases (see _BATH_GAS_EFFC_INDEX).

    .. math::
        M = \sum_i \mathrm{effc}_i \, [X_i]

    Parameters
    ----------
    effcs: list
        Enhanced third body efficiencies [unitless].
    bath_gas_mixture: iris.cube.CubeList
        Number densities [cm-3] of the third-body species (CO, CO2, H2O,
        CH4, H2, N2, He, O3P, NH3 for the Venot2019 network).

    Returns
    -------
        numpy.ndarray
            M [cm-3] - a plain array, not a Cube.

            Why not a Cube: this function only ever feeds into further
            plain-numpy math (e.g. k_troe()'s P_r), so there's nothing
            to re-wrap it for.
    """
    M = None
    for name, idx in _BATH_GAS_EFFC_INDEX.items():
        term = effcs[idx] * bath_gas_mixture.extract_cube(name).core_data()
        M = term if M is None else M + term
    return M


def k_kooij_low_pressure_limit(temp, alpha, beta, gamma, effcs, bath_gas_mixture):
    r"""
    Low pressure limit rate coefficient of a combination/decomposition/
    desexcitation (third-body assisted) reaction, modified Arrhenius
    (Kooij) equation. Adapted from
    http://kida.astrophy.u-bordeaux.fr/help.html#heading23.

    .. math::
        k_0(T) = \alpha \left(\frac{T}{300}\right)^{\beta}
        \exp\left(-\frac{\gamma}{T}\right) M

    (M, the weighted third-body number density, from
    _bath_gas_number_density().)

    Parameters
    ----------
    temp: iris.cube.Cube
        Air temperature [K].
    alpha: float
        Pre-exponential factor at the low pressure limit. Units are one
        concentration factor higher than the reaction's high-pressure/
        bimolecular-equivalent coefficient would need (e.g. cm6 s-1 for a
        2-reactant association, cm3 s-1 for unimolecular decomposition) -
        the extra concentration factor is supplied by `M` below.
    beta: float
        Temperature dependence of the pre-exponential factor [unitless].
    gamma: float
        E/R at the low pressure limit [K].
    effcs: list
        Enhanced third body efficiencies [unitless].
    bath_gas_mixture: iris.cube.CubeList
        Number densities [cm-3] of the third-body species (CO, CO2, H2O,
        CH4, H2, N2, He, O3P, NH3 for the Venot2019 network).

    Returns
    -------
        iris.cube.Cube
            Rate coefficient k(T) with the third-body number density M
            already folded in - one concentration factor lower than
            `alpha` (cm3 s-1 for a 2-reactant association, s-1 for
            unimolecular decomposition).
    """
    # What: use float64 here.
    # Why: float32 underflows k_inf for CH4(+M)↔CH3+H(+M).
    T = temp.core_data().astype(np.float64)

    M = _bath_gas_number_density(effcs, bath_gas_mixture)
    k = (alpha * (T / 300) ** beta) * np.exp(-gamma / T) * M

    # What: re-wrap as a Cube, inheriting temp's coords/metadata (k is
    # plain numpy above, from T).
    # Why: so .rename()/downstream cube ops work.
    k = temp.copy(data=k)
    k.rename("k_kooij_low_pressure_limit")

    return k


def k_troe(temp, k_0, k_inf, a_troe, b_troe, c_troe, d_troe, effcs, bath_gas_mixture):
    r"""
    Rate coefficient of a combination/decomposition reaction, Lindemann
    (UNTESTED) or Troe formalism. Adapted from
    http://kida.astrophy.u-bordeaux.fr/help.html#heading23.

    .. math::
        P_r = \frac{k_0 M}{k_\infty}, \qquad
        k = k_\infty \, \frac{P_r}{1+P_r} \, F

    F = 1 (Lindemann-Hinshelwood, when a_troe = b_troe = c_troe = d_troe
    = 0) or, in the Troe formalism (a/b/c/d below are a_troe/b_troe/
    c_troe/d_troe):

    .. math::
        F_{cent} = (1-a)\,e^{-T/b} + a\,e^{-T/c} + e^{-d/T}

        \log_{10} F = \left[1 + \left(\frac{\log_{10}P_r + c'}
        {N - d'(\log_{10}P_r + c')}\right)^{2}\right]^{-1} \log_{10} F_{cent}

    with :math:`c' = -0.4 - 0.67\log_{10}F_{cent}`,
    :math:`N = 0.75 - 1.27\log_{10}F_{cent}`, :math:`d' = 0.14`.

    Parameters
    ----------
    temp: iris.cube.Cube
        Air temperature [K].
    k_0: float or iris.cube.Cube
        Rate coefficient at the low pressure limit (k_kooij()), one
        concentration factor higher than `k_inf` (e.g. cm6 s-1 vs `k_inf`'s
        cm3 s-1 for a 2-reactant association) so that P_r = k_0*M/k_inf
        below comes out dimensionless.
    k_inf: float or iris.cube.Cube
        Rate coefficient at the high pressure limit (k_kooij()) - the
        bimolecular-or-unimolecular-equivalent coefficient (e.g. cm3 s-1
        for a 2-reactant association, s-1 for unimolecular decomposition).
    a_troe, b_troe, c_troe, d_troe: float
        Troe falloff curve parameters (a_troe unitless, rest in K).
    effcs: list
        Enhanced third body efficiencies [unitless].
    bath_gas_mixture: iris.cube.CubeList
        Number densities [cm-3] of the third-body species (CO, CO2, H2O,
        CH4, H2, N2, He, O3P, NH3 for the Venot2019 network).

    Returns
    -------
        iris.cube.Cube
            Rate coefficient k(T), same units as `k_inf`.
    """
    # What: use float64 here.
    # Why: float32 underflows k_inf for CH4(+M)↔CH3+H(+M).
    T = temp.core_data().astype(np.float64)

    M = _bath_gas_number_density(effcs, bath_gas_mixture)
    P_r = k_0 * M / k_inf

    if a_troe == b_troe == c_troe == d_troe == 0.0:
        # Use Lindemann-Hinshelwood expression
        # UNTESTED as there are no such reactions in the Venot2019 network
        F = iris.cube.Cube(1.0, units="1")
    else:
        # Use Troe formalism to calculate broadening factor
        F_c = (
            (1 - a_troe) * np.exp(-T / b_troe) + a_troe * np.exp(-T / c_troe) + np.exp(-d_troe / T)
        )
        c = -0.4 - 0.67 * np.log10(F_c)
        N = 0.75 - 1.27 * np.log10(F_c)
        d = 0.14

        log10_P_r = iris.analysis.maths.apply_ufunc(np.log10, P_r).core_data()
        x = log10_P_r + c
        frac = 1 / (1 + (x / (N - d * x)) ** 2)
        F = F_c**frac

    # Lindemann-Hinshelwood expression multiplied by a (Troe) broadening factor
    k = k_inf * (P_r / (1 + P_r)) * F

    k.rename("k_troe")

    return k


def chemical_potential(temp, pln):
    r"""
    Standard chemical potential mu^0/(RT) of a selected chemical species.

    .. math::
        \frac{\mu^0}{RT} = a_1(1-\ln T) - \frac{a_2 T}{2} - \frac{a_3 T^2}{6}
        - \frac{a_4 T^3}{12} - \frac{a_5 T^4}{20} + \frac{a_6}{T} - a_7

    using whichever of `pln`'s low/high coefficient sets `temp` falls in
    (temp_low <= T < temp_mid, or temp_mid <= T <= temp_high).

    Parameters
    ----------
    temp: iris.cube.Cube
        Air temperature [K].
    pln: dict
        7 term NASA polynomial coefficients for a selected chemical
        species: "a1_low".."a7_low"/"a1_high".."a7_high" [unitless
        polynomial coefficients], "temp_low"/"temp_mid"/"temp_high" [K]
        (low/mid/high range boundaries), "species" (name, for error
        messages) - see quenchpath.chemistry.polynomials.

    Returns
    -------
        iris.cube.Cube
            Standard chemical potential mu^0/(RT) [dimensionless] of the
            species at each `temp` point.
    """
    # What: use float64 here.
    # Why: float32 underflows k_inf for CH4(+M)↔CH3+H(+M).
    T = temp.core_data().astype(np.float64)

    ln_T = np.log(T)

    # Calculate the chemical potential at low and high temperatures
    low_temp_condition = (T >= pln["temp_low"]) & (T < pln["temp_mid"])
    high_temp_condition = (T >= pln["temp_mid"]) & (T <= pln["temp_high"])

    mu_at_low_temps = np.where(
        low_temp_condition,
        pln["a1_low"] * (1 - ln_T)
        - (pln["a2_low"] * T) / 2
        - (pln["a3_low"] * T**2) / 6
        - (pln["a4_low"] * T**3) / 12
        - (pln["a5_low"] * T**4) / 20
        + pln["a6_low"] * T ** (-1)
        - pln["a7_low"],
        np.nan,
    )
    mu_at_high_temps = np.where(
        high_temp_condition,
        pln["a1_high"] * (1 - ln_T)
        - (pln["a2_high"] * T) / 2
        - (pln["a3_high"] * T**2) / 6
        - (pln["a4_high"] * T**3) / 12
        - (pln["a5_high"] * T**4) / 20
        + pln["a6_high"] * T ** (-1)
        - pln["a7_high"],
        np.nan,
    )

    # Merge into a single cube - mu is plain numpy above (from T), so
    # this also re-wraps it as a Cube inheriting temp's coords/metadata.
    isnan_condition = np.isnan(mu_at_low_temps)
    mu = temp.copy(data=np.where(isnan_condition, mu_at_high_temps, mu_at_low_temps))

    if np.isnan(mu.data).any():
        raise ValueError("NaNs in the chemical potential of " + pln["species"] + ".")

    mu.attributes = {}
    mu.rename("chemical_potential_of_a_species_over_RT")
    # mu^0/(RT) per this function's docstring/formula (standard
    # NASA-polynomial G/(RT) form) - genuinely dimensionless.
    mu.units = Unit("1")

    return mu


def equilibrium_constant(gibbs_energy_f):
    r"""
    Equilibrium constant of a reaction.

    .. math::
        K_{eq} = \exp\left(-\frac{\Delta G^0}{RT}\right)

    Parameters
    ----------
    gibbs_energy_f: iris.cube.Cube
        Gibbs energy of the forward reaction, ΔG^0/(RT) - dimensionless
        (sum of product chemical potentials minus reactant ones).

    Returns
    -------
        iris.cube.Cube
            K_eq [dimensionless] - the thermodynamic (K_p) equilibrium
            constant, not a concentration-based K_c.
    """
    k_eq = iris.analysis.maths.exp((-1) * gibbs_energy_f)

    k_eq.rename("equilibrium_constant")

    return k_eq


# standard pressure / Boltzmann constant (SI) - converts K_p to/from rate
# coefficients via (kB*T/P_0)**(nprods-nreacs) below
P_0 = iris.cube.Cube(1.01325 * 1e5, long_name="standard_pressure", units="kg m-1 s-2")  # Pa
kB = iris.cube.Cube(
    1.380649 * 1e-23, long_name="boltzmann_constant", units="kg m2 s-2 K-1"
)  # J K-1


def reverse_rate_coeff(temp, k_f, k_eq, nprods, nreacs):
    r"""
    Reverse reaction rate coefficient, from the forward rate coefficient
    and the equilibrium constant.

    .. math::
        k_r = \frac{k_f}{K_{eq}} \left(\frac{k_B T}{P_0}\right)^{n_{prods}-n_{reacs}}

    Parameters
    ----------
    temp: iris.cube.Cube
        Air temperature [K].
    k_f: iris.cube.Cube
        Rate coefficient of the forward reaction (cgs units - cm3 s-1
        for bimolecular, cm6 s-1 for termolecular, s-1 for unimolecular).
    k_eq: iris.cube.Cube
        The equilibrium constant K_eq [dimensionless] of the reaction.
    nprods: int
        Number of products in the forward reaction - sets the returned
        rate coefficient's units (3 -> cm6 s-1, 2 -> cm3 s-1, 1 -> s-1).
    nreacs: int
        Number of reactants in the forward reaction.

    Returns
    -------
        iris.cube.Cube
            Reverse reaction rate coefficient k_r, in the units matching
            `nprods` (see above) - the reverse reaction's own molecularity,
            since its reactants are the forward reaction's products.
    """
    k_r = k_f / k_eq * (kB * temp / P_0) ** (nprods - nreacs)
    k_r.rename("reverse_reaction_rate_coefficient")

    # What: convert_units() rescales (not just relabels).
    # Why: kB/P_0 are SI, k_f is cgs.
    if nprods == 3:
        k_r.convert_units("cm6 s-1")
    elif nprods == 2:
        k_r.convert_units("cm3 s-1")
    elif nprods == 1:
        k_r.convert_units("s-1")
    else:
        # What: raise ValueError here - not exercised by the Venot2019
        # network (every reaction has 1-3 products - checked directly).
        # Why still raise: k_r would otherwise keep silently mixed
        # cgs/SI units instead of erroring.
        raise ValueError(f"unexpected product count {nprods}, expected 1, 2, or 3.")

    return k_r
