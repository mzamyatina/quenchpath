"""Stage 03: convert gas mole fractions to number density.

What: unit conversion only - every output cube stays on the model's
native height (theta) levels.

Why no reprojection: air_temperature/air_pressure/every gas cube
already share one exact grid going in, so there's nothing to
reproject.
"""

import re

import iris
from aeolus.const import init_const

from quenchpath import io
from quenchpath.stages import assert_mole_fraction

# Why anchored + unbounded: a length-capped capture group would
# silently mis-parse species names longer than the cap instead of
# erroring.
_GAS_NAME_RE = re.compile(r"^([\w\-]+) mole fraction$")


def mole_fraction_to_number_density(raw_cubes, gas_const, avogadro_const):
    r"""Convert every "<gas> mole fraction" cube in `raw_cubes` to number
    density [molecules cm-3], on the same native height levels it was
    already on.

    .. math::
        n = \frac{x \, P \, N_A}{R \, T}

    (ideal gas law: :math:`x` the gas's mole fraction, :math:`P`/:math:`T`
    air pressure/temperature, :math:`N_A` Avogadro's constant, :math:`R`
    the molar gas constant.)

    Parameters
    ----------
    raw_cubes: iris.cube.CubeList
        Must contain "air_temperature", "air_pressure", and one or more
        "<gas> mole fraction"-named cubes (as loaded from a merged raw run
        - see quenchpath.io.load_raw_run()). Cubes not matching the
        "<gas> mole fraction" naming pattern are ignored.
    gas_const: iris.cube.Cube
        Molar gas constant, e.g. `aeolus.const.init_const().molar_gas_constant`.
    avogadro_const: iris.cube.Cube
        Avogadro constant, e.g. `aeolus.const.init_const().avogadro`.

    Returns
    -------
    iris.cube.CubeList
        One "<gas> number density" cube per input gas, plus
        air_temperature and air_pressure (both passed through
        unmodified), in that order.

    Note: mutates `.units` on the gas cubes it processes - the input
    CubeList's gas cubes are not left untouched.
    """
    air_temperature = raw_cubes.extract_cube("air_temperature")
    air_pressure = raw_cubes.extract_cube("air_pressure")

    gases = iris.cube.CubeList()
    for gas_mf in raw_cubes:
        m = _GAS_NAME_RE.search(gas_mf.name())
        if m is None:
            continue
        gas_name = m.group(1)

        assert_mole_fraction(gas_mf)
        gas_nd = gas_mf * air_pressure * avogadro_const / (gas_const * air_temperature)
        gas_nd.rename(f"{gas_name} number density")
        gas_nd.convert_units("cm-3")
        gases.append(gas_nd)

    gases.append(air_temperature)
    gases.append(air_pressure)

    return gases


def convert_mole_fraction_to_number_density(cfg):
    """I/O driver: load `cfg`'s raw run, convert mole fraction to number
    density, save the result."""
    consts = init_const()
    gas_const = consts.molar_gas_constant
    avogadro_const = consts.avogadro

    raw_cubes = io.load_raw_run(cfg)
    gases = mole_fraction_to_number_density(raw_cubes, gas_const, avogadro_const)
    io.save_converted(gases, cfg)
    return gases
