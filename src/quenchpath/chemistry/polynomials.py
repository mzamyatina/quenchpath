"""Parsing for the 7-term NASA polynomial coefficients used to compute a
species' standard chemical potential (see
quenchpath.chemistry.formulas.chemical_potential).
"""

import collections
import json
import re

_SPECIES_LINE_RE = re.compile(r"([\w\-\(\)]*)\s*([0-9\.]*)\s*([0-9\.]*)\s*([0-9\.]*)")
_COEFF_LINE_RE = re.compile(r"([0-9Ee\.\+\-\s]{,16})" * 7)

_NASA_POLYNOMIALS_RAW_FILENAME = "coeff_NASA_sc.dat"
_NASA_POLYNOMIALS_PROCESSED_FILENAME = "nasa_polynomials_7_term.json"


def parse_nasa_polynomials(raw_lines):
    """Parse 7-term NASA polynomial .dat lines into {species: {...fields}}.

    The file is fixed-width text with 3 lines per species: species name +
    temperature range, then the 7 high-temperature coefficients, then the
    7 low-temperature coefficients. If a species appears more than once,
    only its FIRST entry is kept, later entries are silently ignored -
    use find_duplicate_species() to detect which species this affects.

    Parameters
    ----------
    raw_lines: list[str]
        Lines of a coeff_NASA_sc.dat-format file (as from
        `path.read_text().splitlines()`).

    Returns
    -------
    dict[str, dict]
        Species name -> its parsed fields (see the per-species dict built
        below: temp_low/temp_mid/temp_high, a1_low..a7_low, a1_high..a7_high).
    """
    assert len(raw_lines) % 3 == 0, "raw_lines must hold a whole number of 3-line species records"
    every_three_lines = [raw_lines[i : i + 3] for i in range(0, len(raw_lines), 3)]
    plns = {}
    for species_lines in every_three_lines:
        for j, line in enumerate(species_lines):
            if j == 0:
                m = _SPECIES_LINE_RE.search(line)
                species = m.group(1)
                # header field order is low, high, mid - not low, mid, high
                temp_low = float(m.group(2))
                temp_high = float(m.group(3))
                temp_mid = float(m.group(4))
            elif j == 1:
                a1_high, a2_high, a3_high, a4_high, a5_high, a6_high, a7_high = [
                    float(n) for n in _COEFF_LINE_RE.search(line).groups()
                ]
            elif j == 2:
                a1_low, a2_low, a3_low, a4_low, a5_low, a6_low, a7_low = [
                    float(n) for n in _COEFF_LINE_RE.search(line).groups()
                ]
        if species in plns:
            continue  # first entry wins
        plns[species] = dict(
            species=species,
            temp_low=temp_low,
            temp_mid=temp_mid,
            temp_high=temp_high,
            a1_low=a1_low,
            a2_low=a2_low,
            a3_low=a3_low,
            a4_low=a4_low,
            a5_low=a5_low,
            a6_low=a6_low,
            a7_low=a7_low,
            a1_high=a1_high,
            a2_high=a2_high,
            a3_high=a3_high,
            a4_high=a4_high,
            a5_high=a5_high,
            a6_high=a6_high,
            a7_high=a7_high,
        )
    return plns


def find_duplicate_species(raw_lines):
    """Species names that appear more than once in `raw_lines` (i.e. have
    more than one set of 7-term NASA polynomials in the file)."""
    counts = species_name_counts(raw_lines)
    return [key for key, count in counts.items() if count > 1]


def species_name_counts(raw_lines):
    """collections.Counter of how many times each species name appears in
    `raw_lines` (i.e. how many sets of 7-term NASA polynomials it has)."""
    species_names = [_SPECIES_LINE_RE.search(line).group(1) for line in raw_lines[::3]]
    return collections.Counter(species_names)


def validate_network_coverage(plns, duplicate_species, network_gases):
    """Raise ValueError if any of `network_gases` has ambiguous (duplicate)
    or missing 7-term NASA polynomial data in `plns`/`duplicate_species`
    (as returned by parse_nasa_polynomials()/find_duplicate_species() on
    the same raw file).
    """
    if ispecies := set(network_gases).intersection(duplicate_species):
        raise ValueError(
            f"{ispecies} have more than one set of the 7 term NASA polynomials. Choose one set."
        )
    if ispecies := set(network_gases).difference(plns.keys()):
        raise ValueError(f"No polynomials for {ispecies} in the 7 term NASA polynomials file.")


def load_and_save_nasa_polynomials(path_to_raw_networks, path_to_networks_processed, network_gases):
    """Stage 01 driver: read `coeff_NASA_sc.dat` from `path_to_raw_networks`,
    parse it, validate coverage of `network_gases` (raise ValueError if any
    are missing/ambiguous), then save the parsed polynomials as
    `nasa_polynomials_7_term.json` under `path_to_networks_processed`.

    Parameters
    ----------
    path_to_raw_networks: pathlib.Path
    path_to_networks_processed: pathlib.Path
    network_gases: Iterable[str]
        Species that must have unambiguous polynomial coverage - typically
        `quenchpath.chemistry.network.VENOT2019["all_gases"]`.

    Returns
    -------
    dict[str, dict]
        Same as parse_nasa_polynomials() - the parsed, validated polynomials.
    """
    with open(path_to_raw_networks / _NASA_POLYNOMIALS_RAW_FILENAME) as f:
        raw_lines = f.read().splitlines()
    plns = parse_nasa_polynomials(raw_lines)
    duplicate_species = find_duplicate_species(raw_lines)
    validate_network_coverage(plns, duplicate_species, network_gases)
    with open(path_to_networks_processed / _NASA_POLYNOMIALS_PROCESSED_FILENAME, "w") as f:
        json.dump(plns, f)
    return plns
