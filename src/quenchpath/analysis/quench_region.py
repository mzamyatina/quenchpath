"""What: ln(tau_<conversion>_<chem_kind> / tau_adv_<adv_kind>) - a
continuous, grid-resolution-independent alternative to
quenchpath.analysis.quench_level's own quench_level() and its sharp
ratio=1 "quench level" crossing (same 4 chem_kind/adv_kind
combinations, minus the Zahnle ones - see that module's own docstring
for why those stay separate too).

Why its own module: same reason quenchpath.analysis.k_test_over_k_eq/
zahnle_marley_2014 are their own modules - one function/concept per
module, downstream of quench_level() rather than folded into it.

Why it exists: see quenchpath.analysis.k_test_over_k_eq for the check
that originally motivated it (quenchpath-apps issue #16).
"""

import iris.analysis.maths
import numpy as np

from quenchpath.analysis.quench_level import quench_level
from quenchpath.config import SpeciesConversion


def quench_region(
    cfg, grid_resolution, chem_kind, adv_kind, conversion=SpeciesConversion.CH4_CO, base_dir=None
):
    r"""What: ln(tau_<conversion>_<chem_kind> / tau_adv_<adv_kind>) - a
    continuous, grid-resolution-independent alternative to a sharp
    ratio=1 "quench level" crossing (quench_level()'s own definition,
    for the same `chem_kind`/`adv_kind`/`conversion`/`base_dir` - all
    passed straight through).

    Why a continuous version: quenching is conventionally not a sharp
    crossing but that ratio being >>1 (frozen) or <<1 (still
    equilibrating) - the quench REGION is the band where it's neither,
    i.e. where |ln(ratio)| is small (0 exactly at the crossing,
    symmetric in log space either side).

    What: covers every quench_level() combination except its Zahnle &
    Marley (2014) counterparts (quench_level_zahnle_ls()/
    quench_level_zahnle_qn()).

    Why kept separate: for the same reason
    quenchpath.analysis.quench_level itself keeps them separate (see
    that module's own docstring).

    What: originally built for quenchpath-apps issue #16's finding, specifically
    at `chem_kind="net"`/`adv_kind="qn"` (see
    quenchpath.analysis.k_test_over_k_eq's own module docstring).
    `conversion` defaults to CH4_CO for the same reason (that
    investigation's own assumption), but any conversion quench_level()
    itself supports works here too, for any chem_kind/adv_kind.

    Why built: the CH4->CO pathway search's own "CH4->CO is
    net-favoured at/below the quench level" assumption breaks down
    specifically in this region, not throughout the whole
    at/below-quench volume.
    """
    conversion = SpeciesConversion(conversion)
    ratio = quench_level(cfg, grid_resolution, chem_kind, adv_kind, conversion, base_dir)
    log_ratio = iris.analysis.maths.apply_ufunc(np.log, ratio, new_unit="1")
    log_ratio.rename(f"log_tau_{conversion.value}_{chem_kind}_over_tau_adv_{adv_kind}")
    return log_ratio
