"""One module per pipeline stage, filenames prefixed with each stage's
real pipeline-stage number so they sort in execution order: stage_03_
convert, stage_04_rates, stage_05_paths_serial/stage_05_paths_parallel
(two interchangeable implementations, see stage_05_paths_serial's
docstring), stage_06_merge_paths, stage_07_taus (07's calculation half).
See quenchpath.cli for the corresponding CLI subcommands.
"""


def assert_mole_fraction(cube):
    """What: fail loudly if `cube` isn't genuinely a mole fraction -
    checks both dimensionless units and a "<gas> mole fraction"-named
    cube (stage_03_convert's own naming convention).

    Why check at all, not just assume from the name: a "_mf" suffix is
    only ever as trustworthy as whoever last edited that call site.

    Why check both conditions: either alone could pass by coincidence -
    e.g. a mismatched number density cube caught by mistake (CH4's,
    meant for a pathway timescale).
    """
    assert cube.units == "1" and "mole fraction" in cube.name(), (
        f"{cube.name()!r} (units {cube.units}) isn't a mole fraction cube - expected "
        f"dimensionless units and a '<gas> mole fraction' name."
    )


def assert_number_density(cube, species=None):
    """Fail loudly if `cube` isn't genuinely a number density - same
    reasoning as assert_mole_fraction(), checking both [length]^-3
    units and a "<gas> number density"-named cube.

    What: `species`, if given (e.g. "CH4" or "CO"), additionally
    requires `cube.name() == f"{species} number density"` exactly.

    Why: catches the mixup a bare "number density" substring check
    can't - a genuine number density cube for the *wrong* gas (e.g.
    CO's, passed where CH4's was meant) has correct units and still
    contains "number density", so it would otherwise sail through
    silently.

    Why this matters more now: quenchpath.stages.stage_07_taus picks the
    species by quenchpath.config.SpeciesConversion (CH4 for CH4-to-CO,
    CO for CO-to-CH4) rather than a single hardcoded name - see GitHub
    issue #15.
    """
    assert cube.units.is_convertible("cm-3") and "number density" in cube.name(), (
        f"{cube.name()!r} (units {cube.units}) isn't a number density cube - expected "
        f"[length]^-3-convertible units and a '<gas> number density' name."
    )
    if species is not None:
        expected_name = f"{species} number density"
        assert cube.name() == expected_name, (
            f"{cube.name()!r} isn't {expected_name!r} - wrong species for this call site."
        )
