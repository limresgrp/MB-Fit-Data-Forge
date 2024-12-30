# Make the keys available in this module
from ._keys import *  # noqa: F403, F401
from . import _keys


FOLDER_NAMES = {
    1: "monomers",
    2: "dimers",
    3: "trimers",
}

ATOM_TYPE_TO_H_DISTANCE = {
    'C': 1.086, # 1.086 C-HG, 1.092 HG-C=OO, 1.08 C=C-H
    'N': 1.022, # 1.022 N-HG,
    'O': 0.963, # 0.963 O-HG,
}

MONOMERS_DICT = {
    'POOOO-CC': {
        "composite_monomer_names": ['P-OOOO', 'O-P', 'O-CP'],
        "composite_monomer_elements": 5,
        "composite_monomer_bonds": 4,
    },
    'C__C-CCHH': {
        "composite_monomer_names": ['C-CCH'],
        "composite_monomer_elements": 2,
        "composite_monomer_bonds": 1,
    },
    'C__OO-CC': {
        "composite_monomer_names": ['O-C', 'C-COO', 'O-CC'],
        "composite_monomer_elements": 3,
        "composite_monomer_bonds": 2,
    }
}

CHARGES_DICT = {
    "POOOO-CC": -1,
    "N-CCCC": 1,
}

ATOM_SYMMETRY_NAMES_DICT_GENERAL = {
    'C-CHHH|C-4': 'A',
    'C-HHHN|C-4': 'A',
    'C-CHHH|H-1': 'B',
    'C-HHHN|H-1': 'B',
    'C-CCHH|C-4': 'C',
    'C-CHHO|C-4': 'C',
    'C-CHHN|C-4': 'C',
    'C-CCHH|H-1': 'D',
    'C-CHHO|H-1': 'D',
    'C-CHHN|H-1': 'D',
    'C__C-CCHH|C-3': 'E',
    'C__C-CCHH|H-1': 'F',
    'C__OO-CC|O-2': 'G',
    'C__OO-CC|C-3': 'H',
    'C__OO-CC|O-1': 'I',
    'C__OO-CC|H-1': 'J',
    'POOOO-CC|P-4': 'L',
    'POOOO-CC|O-1': 'M',
    'POOOO-CC|O-2': 'N',
    'POOOO-CC|H-1': 'O',
    'N-CCCC|N-4': 'P',
    'N-CCCC|H-1': 'Q',
    'C-CCHO|C-4': 'R',
    'C-CCHO|H-1': 'S',
}

# '[monomer_name]|[atom_element]-[sorted_bonded_atom_names]'
ATOM_SYMMETRY_NAMES_DICT = {
    'C__OO-CC|H-O': 'J',
    'C__OO-CC|H-C': 'K',
}