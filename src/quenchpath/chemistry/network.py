"""What the Venot et al. (2019) reduced chemical network actually contains.

VENOT2019: the network's species lists - "all_gases" (every species) and
"bath_gases" (the 9 third-body/pressure-broadening partners used in
quenchpath.chemistry.formulas).
"""

VENOT2019 = {
    "all_gases": [
        "1CH2",
        "3CH2",
        "CH2OH",
        "CH3",
        "CH3O",
        "CH3OH",
        "CH4",
        "CN",
        "CO",
        "CO2",
        "H",
        "H2",
        "H2CN",
        "H2CO",
        "H2O",
        "HCN",
        "HCO",
        "He",
        "HNCO",
        "HOCN",
        "N2",
        "N2H2",
        "N2H3",
        "NCO",
        "NH",
        "NH2",
        "NH3",
        "NNH",
        "O-3P",
        "OH",
    ],
    "bath_gases": ["CO", "CO2", "H2O", "CH4", "H2", "N2", "He", "O-3P", "NH3"],
}
