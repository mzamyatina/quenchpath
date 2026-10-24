"""Stage 04: compute Gibbs energies, equilibrium constants, and forward/
reverse rate coefficients/rates for every reaction in the chemical
network, at every grid point of a planet/experiment's processed (stage 03)
output.

What: each pure function below operates on one reaction's `eqn` dict at
a time and returns the new cube(s) it computed (it does not mutate
`eqn`) - the driver, calc_rates(), does the `eqn.update(...)`
accumulation.

Why the driver accumulates: later passes need earlier passes' results
already present on `eqn`.
"""

import json
import re
from functools import reduce
from operator import mul

import iris

from quenchpath import io
from quenchpath.chemistry.formulas import (
    P_0,
    chemical_potential,
    equilibrium_constant,
    k_kooij,
    k_kooij_low_pressure_limit,
    k_troe,
    kB,
    reverse_rate_coeff,
)
from quenchpath.chemistry.network import VENOT2019
from quenchpath.mypaths import path_to_networks_processed

# Anchored + unbounded - see quenchpath.stages.stage_03_convert's identical
# _GAS_NAME_RE.
_GAS_ND_RE = re.compile(r"^([\w\-]+) number density$")


def split_gases_and_bath_gases(converted_cubes, bath_gas_names):
    """Split a stage-03-converted CubeList into (all_gases_nd,
    bath_gases_nd): every "<gas> number density" cube, renamed to just
    "<gas>", plus the subset whose name is in `bath_gas_names`. Cubes
    not matching the "<gas> number density" naming pattern are ignored.

    Note: mutates (renames) the cubes in `converted_cubes` in place.
    """
    gases_nd = iris.cube.CubeList()
    bath_gases_nd = iris.cube.CubeList()
    for gas in converted_cubes:
        m = _GAS_ND_RE.search(gas.name())
        if m is None:
            continue
        gas_name = m.group(1)
        gas.rename(gas_name)
        gases_nd.append(gas)
        if gas_name in bath_gas_names:
            bath_gases_nd.append(gas)
    return gases_nd, bath_gases_nd


def calc_gibbs_and_equilibrium_constant(eqn, temp, plns):
    r"""Gibbs energy of the forward reaction and the resulting
    equilibrium constant, for one reaction.

    .. math::
        \frac{\Delta G_f^0}{RT} = \sum_{prods} \frac{\mu^0}{RT}
        - \sum_{reacs} \frac{\mu^0}{RT}

        K_{eq} = \exp\left(-\frac{\Delta G_f^0}{RT}\right)

    (:math:`\mu^0/RT` from
    quenchpath.chemistry.formulas.chemical_potential(), :math:`K_{eq}`
    from equilibrium_constant() - both dimensionless.)

    Parameters
    ----------
    eqn: dict
        One reaction: "reacs"/"prods" (list[str] of species names).
    temp: iris.cube.Cube
        Air temperature [K].
    plns: dict
        {species: 7-term NASA polynomial coefficients} for every species
        in `eqn["reacs"]`/`eqn["prods"]` - see
        quenchpath.chemistry.polynomials.

    Returns
    -------
    (gibbs_energy_f, k_eq): (iris.cube.Cube, iris.cube.Cube)
        Both dimensionless.
    """
    gibbs_energy_f = 0
    for prod in eqn["prods"]:
        gibbs_energy_f += chemical_potential(temp, plns[prod])
    for reac in eqn["reacs"]:
        gibbs_energy_f -= chemical_potential(temp, plns[reac])
    gibbs_energy_f.rename("gibbs_energy_of_forward_reaction")

    k_eq = equilibrium_constant(gibbs_energy_f)
    return gibbs_energy_f, k_eq


def calc_forward_rate(eqn, temp, all_gases_nd, bath_gases_nd):
    r"""Forward rate coefficient (k_f) and forward rate (rate_f) for one
    reaction, dispatched on `eqn["ktype"]` (kooij / kooij low pressure
    limit / troe - see quenchpath.chemistry.formulas.k_kooij()/
    k_kooij_low_pressure_limit()/k_troe() for k_f's own formula in each
    case).

    .. math::
        \mathrm{rate}_f = k_f \prod_{reacs} [\mathrm{reactant}]

    (one concentration factor per reactant, so squared for a reactant
    appearing twice.)

    Parameters
    ----------
    eqn: dict
        One reaction: "reacs" (list[str]), "ktype", and that ktype's own
        rate-law parameters (see k_kooij()/k_kooij_low_pressure_limit()/
        k_troe()).
    temp: iris.cube.Cube
        Air temperature [K].
    all_gases_nd: iris.cube.CubeList
        Number density [cm-3] of every gas, keyed by name (see
        split_gases_and_bath_gases()).
    bath_gases_nd: iris.cube.CubeList
        Number density [cm-3] of the third-body species - only used for
        "kooij low pressure limit"/"troe" ktypes.

    Returns
    -------
    (k_f, rate_f): (iris.cube.Cube, iris.cube.Cube)
        k_f in cm3 s-1 (2 reactants) or s-1 (1 reactant); rate_f in
        cm-3 s-1.
    """
    reacs = all_gases_nd.extract(eqn["reacs"])
    # What: defensive assert.
    # Why: CubeList.extract() with duplicate names (e.g.
    # "OH+OH↔H2O+O-3P"'s reacs=["OH", "OH"] - a real reaction, not
    # hypothetical) isn't a documented/guaranteed 1-cube-per-constraint
    # behavior - if it ever silently returned one OH cube instead of
    # two, reduce(mul, reacs) below would compute [OH] instead of
    # [OH]^2.
    assert len(reacs) == len(eqn["reacs"]), (
        f"{eqn['eqn']}: expected {len(eqn['reacs'])} reactant cube(s), got {len(reacs)}"
    )

    if eqn["ktype"] == "kooij":
        k = k_kooij(temp, eqn["alpha"], eqn["beta"], eqn["gamma"])
    elif eqn["ktype"] == "kooij low pressure limit":
        k = k_kooij_low_pressure_limit(
            temp, eqn["alpha_0"], eqn["beta_0"], eqn["gamma_0"], eqn["effcs"], bath_gases_nd
        )
    elif eqn["ktype"] == "troe":
        k_0 = k_kooij(temp, eqn["alpha_0"], eqn["beta_0"], eqn["gamma_0"])
        k_inf = k_kooij(temp, eqn["alpha_inf"], eqn["beta_inf"], eqn["gamma_inf"])
        k = k_troe(
            temp,
            k_0,
            k_inf,
            eqn["a_troe"],
            eqn["b_troe"],
            eqn["c_troe"],
            eqn["d_troe"],
            eqn["effcs"],
            bath_gases_nd,
        )
    else:
        raise ValueError(f"No such ktype: {eqn['ktype']}.")

    if len(reacs) == 2:
        k.units = "cm3 s-1"
    elif len(reacs) == 1:
        k.units = "s-1"
    else:
        # What: raise ValueError here - not exercised by the Venot2019
        # network (every reaction has 1 or 2 reactants - checked
        # directly).
        # Why still raise: k would otherwise keep temp's inherited K
        # units instead of erroring.
        raise ValueError(f"{eqn['eqn']}: unexpected reactant count {len(reacs)}, expected 1 or 2.")

    k_f = k
    k_f.rename("forward_reaction_rate_coefficient")
    rate_f = reduce(mul, reacs) * k_f
    rate_f.rename("forward_reaction_rate")
    rate_f.convert_units("cm-3 s-1")
    return k_f, rate_f


def calc_reverse_rate(eqn, temp, all_gases_nd):
    r"""Reverse rate coefficient (k_r) and reverse rate (rate_r) for one
    reaction. Requires `eqn["k_f"]` and `eqn["k_eq"]` to already be set
    (see calc_forward_rate()/calc_gibbs_and_equilibrium_constant()).

    .. math::
        k_r = \frac{k_f}{K_{eq}} \left(\frac{k_B T}{P_0}\right)^{n_{prods}-n_{reacs}}

        \mathrm{rate}_r = k_r \prod_{prods} [\mathrm{product}]

    (k_r from quenchpath.chemistry.formulas.reverse_rate_coeff(); the
    :math:`(k_B T/P_0)^{n_{prods}-n_{reacs}}` factor converts
    :math:`K_{eq}`'s dimensionless K_p convention back to k_f/k_r's
    concentration-based K_c.)

    Parameters
    ----------
    eqn: dict
        One reaction: "reacs"/"prods" (list[str]), "k_f"/"k_eq" (already
        computed).
    temp: iris.cube.Cube
        Air temperature [K].
    all_gases_nd: iris.cube.CubeList
        Number density [cm-3] of every gas, keyed by name.

    Returns
    -------
    (k_r, rate_r): (iris.cube.Cube, iris.cube.Cube)
        Units matching the reverse reaction's own molecularity (see
        reverse_rate_coeff()); rate_r in cm-3 s-1.
    """
    nprods, nreacs = len(eqn["prods"]), len(eqn["reacs"])
    k_r = reverse_rate_coeff(temp, eqn["k_f"], eqn["k_eq"], nprods, nreacs)
    prods = iris.cube.CubeList(all_gases_nd.extract_cube(gas) for gas in eqn["prods"])
    rate_r = reduce(mul, prods) * k_r
    rate_r.rename("reverse_reaction_rate")
    rate_r.convert_units("cm-3 s-1")
    return k_r, rate_r


def merge_h_ch3_ch4_special_case(eqns, temp, all_gases_nd):
    r"""H+CH3(+M)<->CH4(+M) and CH4(+M)<->CH3+H(+M) are the reverse of
    each other, and CH4(+M)<->CH3+H(+M) is slower - add
    CH4(+M)<->CH3+H(+M)'s rates to H+CH3(+M)<->CH4(+M)'s, then drop the
    CH4(+M)<->CH3+H(+M) entry. Every `eqn` in `eqns` must already have
    k_f/k_r/gibbs_energy_f computed (runs after calc_forward_rate()/
    calc_reverse_rate()).

    What: whether summing rate coefficients is the right way to
    reconcile these two independently-fit duplicate entries (vs. e.g.
    picking one) is an open question - see issue #4.

    Why trust this merge anyway: it does preserve detailed balance
    exactly (checked empirically, not just algebraically), and the two
    entries' differing third-body treatment is a genuine feature of the
    raw Venot2019 data, not a parsing artifact.

    Using "hcm"/"c4m" as shorthand for the two entries:

    .. math::
        k_f^{merged} = k_f^{hcm} + k_r^{c4m}, \qquad
        k_r^{merged} = k_r^{hcm} + k_f^{c4m}

        K_{eq}^{merged} = \frac{k_f^{merged}}{k_r^{merged}}
        \left(\frac{k_B T}{P_0}\right)^{\Delta\nu}

    (:math:`\Delta\nu` = nprods - nreacs of the merged reaction;
    :math:`K_{eq}^{merged}` recomputed from the merged k_f/k_r rather
    than reused from either original entry - see the detailed-balance
    check above.)

    Parameters
    ----------
    eqns: list[dict]
        Every reaction, each with k_f/k_r/gibbs_energy_f already set.
        Must include both "H+CH3(+M)↔CH4(+M)" and "CH4(+M)↔CH3+H(+M)".
    temp: iris.cube.Cube
        Air temperature [K].
    all_gases_nd: iris.cube.CubeList
        Number density [cm-3] of every gas, keyed by name.

    Returns
    -------
    list[dict]
        A NEW list (does not mutate `eqns`) with the merge applied - one
        entry shorter than `eqns` ("CH4(+M)↔CH3+H(+M)" dropped).
    """
    h_ch3_m = next(eqn for eqn in eqns if eqn["eqn"] == "H+CH3(+M)↔CH4(+M)")
    ch4_m = next(eqn for eqn in eqns if eqn["eqn"] == "CH4(+M)↔CH3+H(+M)")

    # copy so reacs/prods/rtype/ktype/rate-law params/gibbs_energy_f carry
    # over; k_f/k_r/rate_f/rate_r/k_eq get overridden with merged values below
    h_ch3_m_ch4_m = dict(h_ch3_m)

    h_nd = all_gases_nd.extract_cube("H")
    ch3_nd = all_gases_nd.extract_cube("CH3")
    ch4_nd = all_gases_nd.extract_cube("CH4")

    h_ch3_m_ch4_m["k_f"] = h_ch3_m["k_f"] + ch4_m["k_r"]
    h_ch3_m_ch4_m["k_r"] = h_ch3_m["k_r"] + ch4_m["k_f"]
    h_ch3_m_ch4_m["k_f"].rename("forward_reaction_rate_coefficient")
    h_ch3_m_ch4_m["k_r"].rename("reverse_reaction_rate_coefficient")

    h_ch3_m_ch4_m["rate_f"] = h_ch3_m_ch4_m["k_f"] * h_nd * ch3_nd
    h_ch3_m_ch4_m["rate_r"] = h_ch3_m_ch4_m["k_r"] * ch4_nd
    h_ch3_m_ch4_m["rate_f"].rename("forward_reaction_rate")
    h_ch3_m_ch4_m["rate_r"].rename("reverse_reaction_rate")
    h_ch3_m_ch4_m["rate_f"].convert_units("cm-3 s-1")
    h_ch3_m_ch4_m["rate_r"].convert_units("cm-3 s-1")

    # k_eq from k_f/k_r (coefficients, not rate_f/rate_r - those bake in
    # local concentrations). k_f/k_r alone is K_c (cm3); convert to the
    # dimensionless K_p every other reaction's k_eq uses.
    delta_nu = len(h_ch3_m_ch4_m["prods"]) - len(h_ch3_m_ch4_m["reacs"])
    k_ratio = h_ch3_m_ch4_m["k_f"] / h_ch3_m_ch4_m["k_r"]
    h_ch3_m_ch4_m["k_eq"] = k_ratio * (kB * temp / P_0) ** delta_nu
    h_ch3_m_ch4_m["k_eq"].convert_units("1")
    h_ch3_m_ch4_m["k_eq"].rename("equilibrium_constant")

    new_eqns = []
    for eqn in eqns:
        if eqn["eqn"] == "H+CH3(+M)↔CH4(+M)":
            new_eqns.append(h_ch3_m_ch4_m)
        elif eqn["eqn"] == "CH4(+M)↔CH3+H(+M)":
            continue
        else:
            new_eqns.append(eqn)
    return new_eqns


def build_reaction_metadata(eqn, cfg):
    """What: dict of a reaction's identifying/rate-law-parameter fields,
    in the form saved as each output cube's `.attributes` - dispatched
    on `eqn["ktype"]`.
    Why dispatch on ktype: the field set differs per rate-law
    formalism.
    """
    metadata = {
        "planet": cfg.planet,
        "exp": cfg.exp,
        "mdh": cfg.mdh,
        "tK": cfg.tK,
        "eqn": eqn["eqn"],
        "reacs": eqn["reacs"],
        "prods": eqn["prods"],
        "rtype": eqn["rtype"],
        "ktype": eqn["ktype"],
    }
    if eqn["ktype"] == "kooij":
        metadata.update(
            {
                "kpars_alpha": eqn["alpha"],
                "kpars_beta": eqn["beta"],
                "kpars_gamma": eqn["gamma"],
                "kpars_F": eqn["F"],
                "kpars_g": eqn["g"],
            }
        )
    elif eqn["ktype"] in ("kooij low pressure limit", "troe"):
        # What: troe extends this same low-pressure-limit parameter
        # set with its own high-pressure-limit/falloff-curve params
        # below.
        # Why: keeps the two ktypes' shared fields listed once instead
        # of twice.
        metadata.update(
            {
                "kpars_alpha_0": eqn["alpha_0"],
                "kpars_beta_0": eqn["beta_0"],
                "kpars_gamma_0": eqn["gamma_0"],
                "kpars_F_0": eqn["F_0"],
                "kpars_g_0": eqn["g_0"],
            }
        )
        if eqn["ktype"] == "troe":
            metadata.update(
                {
                    "kpars_alpha_inf": eqn["alpha_inf"],
                    "kpars_beta_inf": eqn["beta_inf"],
                    "kpars_gamma_inf": eqn["gamma_inf"],
                    "kpars_F_inf": eqn["F_inf"],
                    "kpars_g_inf": eqn["g_inf"],
                    "kpars_a_troe": eqn["a_troe"],
                    "kpars_b_troe": eqn["b_troe"],
                    "kpars_c_troe": eqn["c_troe"],
                    "kpars_d_troe": eqn["d_troe"],
                }
            )
        metadata["kpars_effcs"] = eqn["effcs"]
    else:
        raise ValueError(f"No such ktype: {eqn['ktype']}.")
    return metadata


def calc_rates(cfg):
    """I/O driver: load `cfg`'s stage-03 output plus the chemical network
    and NASA polynomials, compute gibbs_energy_f/k_eq/k_f/rate_f/k_r/
    rate_r for every reaction (applying the H+CH3(+M)/CH4(+M) special
    case), save all reactions' cubes in a handful of chunked files under
    path_to_intermediates/rates/ (see io.save_rate_cubes), and return the
    final eqns list.
    """
    converted_cubes = io.load_converted(cfg)
    all_gases_nd, bath_gases_nd = split_gases_and_bath_gases(
        converted_cubes, VENOT2019["bath_gases"]
    )
    temp = converted_cubes.extract_cube("air_temperature")
    temp.attributes = {}

    with open(path_to_networks_processed / "venot2019_network.json") as f:
        eqns = json.load(f)
    with open(path_to_networks_processed / "nasa_polynomials_7_term.json") as f:
        plns = json.load(f)

    for eqn in eqns:
        gibbs_energy_f, k_eq = calc_gibbs_and_equilibrium_constant(eqn, temp, plns)
        eqn.update({"gibbs_energy_f": gibbs_energy_f, "k_eq": k_eq})

    for eqn in eqns:
        k_f, rate_f = calc_forward_rate(eqn, temp, all_gases_nd, bath_gases_nd)
        eqn.update({"k_f": k_f, "rate_f": rate_f})

    for eqn in eqns:
        k_r, rate_r = calc_reverse_rate(eqn, temp, all_gases_nd)
        eqn.update({"k_r": k_r, "rate_r": rate_r})

    eqns = merge_h_ch3_ch4_special_case(eqns, temp, all_gases_nd)

    cubes_to_save = iris.cube.CubeList()
    for eqn in eqns:
        metadata = build_reaction_metadata(eqn, cfg)
        for var in ["gibbs_energy_f", "k_eq", "k_f", "k_r", "rate_f", "rate_r"]:
            cube = eqn[var]
            cube.attributes = metadata
            # What: moved planet/exp/mdh/tK into attributes.globals
            # (rather than the default per-variable placement) - see
            # iris.cube.CubeAttrsDict/iris.FUTURE.save_split_attrs, set
            # in quenchpath.io's header.
            # Why: planet/exp/mdh/tK are identical across every cube
            # saved for this cfg (unlike eqn/reacs/prods/rtype/ktype/
            # kpars_*, which genuinely differ per reaction), so
            # io.save_rate_cubes() writes them once per file rather
            # than repeating them on every cube.
            for key in ("planet", "exp", "mdh", "tK"):
                cube.attributes.globals[key] = cube.attributes.locals.pop(key)
            cubes_to_save.append(cube)
    io.save_rate_cubes(cubes_to_save, cfg)

    return eqns
