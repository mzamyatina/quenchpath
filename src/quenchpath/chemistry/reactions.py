"""Parsing for the Venot et al. (2019) reduced chemical network's reaction
rate-law parameter files (reactions_N.dat).

- reactions_4,5,6,9,11,12.dat are empty, not read.
- all reactions are reversible.
- 181 reactions in total: 153 bimolecular (reactions_1.dat), 4
  combination low pressure limit (reactions_2.dat), 2 combination Troe
  (reactions_3.dat), 2 decomposition (reactions_7.dat), 18 decomposition
  low pressure limit (reactions_8.dat), 1 decomposition Troe
  (reactions_10.dat), 1 desexcitation (reactions_13.dat).
"""

import json
import re

_VENOT2019_NETWORK_PROCESSED_FILENAME = "venot2019_network.json"

_KOOIJ_RE = re.compile(r"\s([\w\-\s]{,10})" * 10 + r"\s([0-9Ee\.\+\-\s]{,10})" * 5)
_KOOIJ_LOW_PRESSURE_LIMIT_RE = re.compile(
    r"\s([\w\-\s]{,10})" * 10 + r"\s([0-9Ee\.\+\-\s]{,10})" * 27
)
_TROE_RE = re.compile(r"\s([\w\-\s]{,10})" * 10 + r"\s([0-9Ee\.\+\-\s]{,10})" * 36)


def parse_kooij_reactions(raw_lines, rtype):
    """Reversible bimolecular/decomposition reactions using the Kooij
    formalism (reactions_1.dat: bimolecular, reactions_7.dat: decomposition).
    """
    eqns = []
    for line in raw_lines:
        m = _KOOIJ_RE.search(line)
        reacs = [i.strip() for i in m.group(*range(1, 6)) if i.strip()]
        prods = [i.strip() for i in m.group(*range(6, 11)) if i.strip()]
        alpha, beta, gamma, F, g = [float(i) for i in m.group(*range(11, 16))]
        eqns.append(
            dict(
                eqn="+".join(reacs) + "↔" + "+".join(prods),
                reacs=reacs,
                prods=prods,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                F=F,
                g=g,
                rtype=rtype,
                ktype="kooij",
            )
        )
    return eqns


def parse_kooij_low_pressure_limit_reactions(raw_lines, rtype):
    """Reversible combination/decomposition reactions at the low pressure
    limit, and reversible third-body-assisted desexcitation reactions
    (reactions_2.dat: combination, reactions_8.dat: decomposition,
    reactions_13.dat: desexcitation).
    """
    eqns = []
    for line in raw_lines:
        m = _KOOIJ_LOW_PRESSURE_LIMIT_RE.search(line)
        reacs = [i.strip() for i in m.group(*range(1, 6)) if i.strip()]
        prods = [i.strip() for i in m.group(*range(6, 11)) if i.strip()]
        alpha_0, beta_0, gamma_0, F_0, g_0 = [float(i) for i in m.group(*range(11, 16))]
        effcs = [float(i) for i in m.group(*range(16, 38))]
        eqns.append(
            dict(
                eqn="+".join(reacs) + "+M↔" + "+".join(prods) + "+M",
                reacs=reacs,
                prods=prods,
                alpha_0=alpha_0,
                beta_0=beta_0,
                gamma_0=gamma_0,
                F_0=F_0,
                g_0=g_0,
                effcs=effcs,
                rtype=rtype,
                ktype="kooij low pressure limit",
            )
        )
    return eqns


def parse_troe_reactions(raw_lines, rtype):
    """Reversible combination reactions using the Troe formalism, and
    reversible decomposition reactions with a specific third body using
    the Troe formalism (reactions_3.dat: combination, reactions_10.dat:
    decomposition).
    """
    eqns = []
    for line in raw_lines:
        m = _TROE_RE.search(line)
        reacs = [i.strip() for i in m.group(*range(1, 6)) if i.strip()]
        prods = [i.strip() for i in m.group(*range(6, 11)) if i.strip()]
        alpha_0, beta_0, gamma_0, F_0, g_0 = [float(i) for i in m.group(*range(11, 16))]
        alpha_inf, beta_inf, gamma_inf, F_inf, g_inf = [float(i) for i in m.group(*range(16, 21))]
        a_troe, b_troe, c_troe, d_troe = [float(i) for i in m.group(*range(21, 25))]
        effcs = [float(i) for i in m.group(*range(25, 47))]
        eqns.append(
            dict(
                eqn="+".join(reacs) + "(+M)↔" + "+".join(prods) + "(+M)",
                reacs=reacs,
                prods=prods,
                alpha_0=alpha_0,
                beta_0=beta_0,
                gamma_0=gamma_0,
                F_0=F_0,
                g_0=g_0,
                alpha_inf=alpha_inf,
                beta_inf=beta_inf,
                gamma_inf=gamma_inf,
                F_inf=F_inf,
                g_inf=g_inf,
                a_troe=a_troe,
                b_troe=b_troe,
                c_troe=c_troe,
                d_troe=d_troe,
                effcs=effcs,
                rtype=rtype,
                ktype="troe",
            )
        )
    return eqns


# What: (reactions_N.dat number, reaction type, parser).
# Why order matters: some downstream code relies on eqns list position.
_REACTION_FILE_GROUPS = [
    (1, "bimolecular", parse_kooij_reactions),
    (2, "combination", parse_kooij_low_pressure_limit_reactions),
    (3, "combination", parse_troe_reactions),
    (7, "decomposition", parse_kooij_reactions),
    (8, "decomposition", parse_kooij_low_pressure_limit_reactions),
    (10, "decomposition", parse_troe_reactions),
    (13, "desexcitation", parse_kooij_low_pressure_limit_reactions),
]


def load_venot2019_network(path_to_raw_networks):
    """Read and parse all 7 non-empty reactions_N.dat files under
    `path_to_raw_networks` into one list of reaction dicts.

    Parameters
    ----------
    path_to_raw_networks: pathlib.Path

    Returns
    -------
    list[dict]
        One dict per reaction (see parse_kooij_reactions()/
        parse_kooij_low_pressure_limit_reactions()/parse_troe_reactions()
        for their fields), across all 7 files, in _REACTION_FILE_GROUPS'
        order.
    """
    eqns = []
    for n, rtype, parser in _REACTION_FILE_GROUPS:
        with open(path_to_raw_networks / f"reactions_{n}.dat") as f:
            raw_lines = f.read().splitlines()
        eqns.extend(parser(raw_lines, rtype))
    return eqns


def load_and_save_venot2019_network(path_to_raw_networks, path_to_networks_processed):
    """Stage 02 driver: read+parse all 7 non-empty reactions_N.dat files
    under `path_to_raw_networks`, then save the result as
    `venot2019_network.json` under `path_to_networks_processed`.

    Parameters
    ----------
    path_to_raw_networks: pathlib.Path
    path_to_networks_processed: pathlib.Path

    Returns
    -------
    list[dict]
        Same as load_venot2019_network() - the parsed reaction list.
    """
    eqns = load_venot2019_network(path_to_raw_networks)
    with open(path_to_networks_processed / _VENOT2019_NETWORK_PROCESSED_FILENAME, "w") as f:
        json.dump(eqns, f)
    return eqns
