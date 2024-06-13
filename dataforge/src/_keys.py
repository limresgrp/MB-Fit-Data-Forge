from typing import Final

# Qchem optimised structure parsing
NAPPING: Final[int] = 0
READY:   Final[int] = 1
READING: Final[int] = 2

# Rule to follow if a file already exists
SKIP:      Final[int] = 0
APPEND:    Final[int] = 1
OVERWRITE: Final[int] = 2

# --------------------------------------- #

ENERGY_FILENAME  : Final[str] = "energy.csv"
TOPOLOGY_FILENAME: Final[str] = "topology.json"