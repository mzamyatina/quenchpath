"""What: CH4<->CO SYSTEM-level net-flux direction, given by the Zahnle
& Marley (2014)-style K_test/K_eq disequilibrium quotient for the net
reaction CH4 + H2O <-> CO + 3H2 - see ch4_co_k_test_over_k_eq()'s own
docstring for the formula/convention.

Why a separate module: this is a different, coarser quantity than any
single reaction's own k_eq-favoured orientation (see
quenchpath.graph.shortest_path.build_unidirect_graph()/
build_bidirect_graph(), which orient per-edge).

Why ch4_co_k_test_over_k_eq() exists (GitHub issue #16): quenchpath's
own CH4-to-CO pathway search assumes CH4->CO is the net-favoured
direction at/below the CH4 quench level - this function checks that
assumption directly instead of just asserting it.

Finding: holds on 7 of 8 real planet/mdh/tK configs except in a narrow
band right at the CH4 quench region (see
quenchpath.analysis.quench_region.quench_region()), where the net
direction can genuinely flip to CO->CH4. Investigation history: GitHub
issue #16's own comment thread (closed).
"""

import json

from quenchpath import io
from quenchpath.chemistry.formulas import P_0, kB
from quenchpath.mypaths import path_to_networks_processed
from quenchpath.stages.stage_04_rates import calc_gibbs_and_equilibrium_constant


def ch4_co_k_test_over_k_eq(cfg):
    r"""K_test/K_eq for the net CH4 + H2O <-> CO + 3H2 reaction, as a
    per-location Cube on `cfg`'s stage-03 converted (native) grid.

    .. math::
        K_{test} = \frac{p_{CH_4} \, p_{H_2O}}{p_{CO} \, p_{H_2}^3}
        \quad \text{(from the real, disequilibrium state)}

        K_{eq} = \text{the same quotient's value AT equilibrium, same } (p, T)

    K_test/K_eq > 1: CH4/H2O supersaturated relative to equilibrium ->
    net flux runs CH4->CO. K_test/K_eq < 1: net flux runs CO->CH4.
    """
    converted_cubes = io.load_converted(cfg)
    temp = converted_cubes.extract_cube("air_temperature")
    temp.attributes = {}

    partial_pressures = {}
    for gas in ("CH4", "H2O", "CO", "H2"):
        nd = converted_cubes.extract_cube(f"{gas} number density")
        nd.attributes = {}
        p = nd * kB * temp
        p.convert_units("Pa")
        partial_pressures[gas] = p / P_0  # dimensionless, K_p convention

    k_test = (
        partial_pressures["CH4"]
        * partial_pressures["H2O"]
        / (partial_pressures["CO"] * partial_pressures["H2"] ** 3)
    )

    with open(path_to_networks_processed / "nasa_polynomials_7_term.json") as f:
        plns = json.load(f)
    # What: written as CO+3H2->CH4+H2O (not CH4+H2O->CO+3H2).
    # Why: so K_eq lands in the same products-over-reactants form as
    # k_test above (products = CH4, H2O) - see
    # calc_gibbs_and_equilibrium_constant()'s own docstring for the
    # sign convention this relies on.
    eqn = {"reacs": ["CO", "H2", "H2", "H2"], "prods": ["CH4", "H2O"]}
    _, k_eq = calc_gibbs_and_equilibrium_constant(eqn, temp, plns)

    ratio = k_test / k_eq
    # What: k_test/k_eq is dimensionless by construction (both are
    # already-dimensionless K_p-convention quotients), and iris's own
    # unit algebra already resolves it to "1" here.
    # Why assert rather than trust that silently: makes the invariant
    # explicit, so a future change to k_test/k_eq's own units (a bug)
    # would fail loudly here instead of returning a wrongly-labelled
    # ratio downstream.
    assert ratio.units == "1", f"expected a dimensionless ratio, got units={ratio.units}"
    ratio.rename("ch4_co_k_test_over_k_eq")
    return ratio
