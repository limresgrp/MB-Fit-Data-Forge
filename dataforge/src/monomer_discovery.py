"""Automatic monomer discovery from connectivity and bond-order evidence."""

from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


# Covalent radii in Angstrom. These are deliberately conservative: geometry
# inference is a fallback for topologies such as GROMACS TPR files that do not
# preserve bond order, and every inferred decision is written to metadata.
COVALENT_RADII = {
    "H": 0.31,
    "B": 0.85,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "F": 0.57,
    "P": 1.07,
    "S": 1.05,
    "Cl": 1.02,
    "CL": 1.02,
    "Br": 1.20,
    "BR": 1.20,
    "I": 1.39,
}


def _normalise_element(value) -> str:
    value = str(value).strip()
    if not value:
        return value
    return value[0].upper() + value[1:].lower()


def explicit_bond_orders(universe, bonds: np.ndarray) -> Tuple[Optional[np.ndarray], str]:
    """Return topology-provided bond orders when available.

    MDAnalysis exposes this as ``universe.bonds.order`` for formats that carry
    it. Older versions and some parsers expose the underlying topology
    attribute instead, so both forms are checked.
    """
    candidates = []
    try:
        candidates.append(getattr(universe.bonds, "order"))
    except Exception:
        pass
    try:
        candidates.append(getattr(universe.bonds, "bondorder"))
    except Exception:
        pass
    try:
        for attr in universe._topology.attrs:
            if getattr(attr, "attrname", "") in {"bondorder", "order"}:
                candidates.append(attr.values)
    except AttributeError:
        pass

    for candidate in candidates:
        try:
            values = np.asarray(candidate, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            continue
        if len(values) == len(bonds) and np.all(np.isfinite(values)):
            return values, "topology"
    return None, "unavailable"


def _geometry_bond_orders(
    positions: np.ndarray,
    atom_types: np.ndarray,
    bonds: np.ndarray,
    double_ratio: float,
    triple_ratio: float,
) -> Tuple[np.ndarray, List[dict]]:
    positions = np.asarray(positions, dtype=float)
    atom_types = np.asarray([_normalise_element(x) for x in atom_types])
    bond_orders = np.ones(len(bonds), dtype=float)
    evidence = []

    for bond_index, (left, right) in enumerate(np.asarray(bonds, dtype=int)):
        left_type = atom_types[left]
        right_type = atom_types[right]
        radii = COVALENT_RADII.get(left_type, 0.77) + COVALENT_RADII.get(right_type, 0.77)
        distances = np.linalg.norm(positions[:, left] - positions[:, right], axis=1)
        median_distance = float(np.median(distances))
        ratio = median_distance / radii if radii else 1.0

        order = 1
        if left_type != "H" and right_type != "H":
            if ratio <= triple_ratio:
                order = 3
            elif ratio <= double_ratio:
                order = 2
        bond_orders[bond_index] = order
        if order > 1:
            evidence.append({
                "atoms": [int(left), int(right)],
                "elements": [left_type, right_type],
                "order": order,
                "median_distance": median_distance,
                "covalent_radius_sum": radii,
                "distance_ratio": ratio,
            })
    return bond_orders, evidence


def resolve_bond_orders(
    dataset: dict,
    mode: str = "auto",
    double_ratio: float = 0.90,
    triple_ratio: float = 0.80,
) -> Tuple[np.ndarray, dict]:
    """Resolve bond orders and return both values and auditable evidence."""
    bonds = np.asarray(dataset["bond_indices"], dtype=int)
    explicit = dataset.get("bond_orders")
    if explicit is not None:
        explicit = np.asarray(explicit, dtype=float).reshape(-1)
        if len(explicit) != len(bonds):
            explicit = None

    if mode not in {"auto", "topology", "geometry"}:
        raise ValueError(f"Unknown bond-order mode {mode!r}.")

    if mode in {"auto", "topology"} and explicit is not None and np.all(np.isfinite(explicit)):
        orders = np.maximum(1, np.rint(explicit)).astype(int)
        evidence = [
            {"atoms": [int(a), int(b)], "order": int(order), "source": "topology"}
            for (a, b), order in zip(bonds, orders)
            if order > 1
        ]
        return orders, {
            "source": "topology",
            "double_ratio": double_ratio,
            "triple_ratio": triple_ratio,
            "multiple_bonds": evidence,
        }

    if mode == "topology":
        return np.ones(len(bonds), dtype=int), {
            "source": "topology-unavailable-fallback-single",
            "double_ratio": double_ratio,
            "triple_ratio": triple_ratio,
            "multiple_bonds": [],
        }

    orders, evidence = _geometry_bond_orders(
        dataset["position"],
        dataset["atom_type"],
        bonds,
        double_ratio,
        triple_ratio,
    )
    return orders, {
        "source": "geometry",
        "double_ratio": double_ratio,
        "triple_ratio": triple_ratio,
        "multiple_bonds": evidence,
    }


class _UnionFind:
    def __init__(self, values: Iterable[int]):
        self.parent = {int(value): int(value) for value in values}

    def find(self, value: int) -> int:
        value = int(value)
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int):
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def discover_monomer_groups(
    dataset: dict,
    mode: str = "auto",
    double_ratio: float = 0.90,
    triple_ratio: float = 0.80,
) -> Tuple[List[List[int]], dict]:
    """Find heavy-atom monomer cores and merge multiple-bond components.

    The returned indices are local trajectory atom indices. Hydrogens are not
    included in the core; ``Monomer`` adds all bonded hydrogens and records
    external heavy-atom connections when the core is capped.
    """
    atom_types = np.asarray([_normalise_element(x) for x in dataset["atom_type"]])
    heavy_atoms = [int(i) for i, atom_type in enumerate(atom_types) if atom_type != "H"]
    bonds = np.asarray(dataset["bond_indices"], dtype=int)
    orders, order_metadata = resolve_bond_orders(
        dataset,
        mode=mode,
        double_ratio=double_ratio,
        triple_ratio=triple_ratio,
    )

    union_find = _UnionFind(heavy_atoms)
    multiple_bonds = []
    for (left, right), order in zip(bonds, orders):
        if order >= 2 and left in union_find.parent and right in union_find.parent:
            union_find.union(left, right)
            multiple_bonds.append({"atoms": [int(left), int(right)], "order": int(order)})

    components = defaultdict(list)
    for atom in heavy_atoms:
        components[union_find.find(atom)].append(atom)

    groups = sorted((sorted(group) for group in components.values()), key=lambda group: group[0])
    group_metadata = []
    for group in groups:
        group_metadata.append({
            "heavy_atom_indices": group,
            "heavy_atom_elements": [str(atom_types[i]) for i in group],
            "contains_multiple_bond": len(group) > 1,
        })

    return groups, {
        "mode": mode,
        "bond_orders": [int(order) for order in orders],
        "bond_order_evidence": order_metadata,
        "multiple_bonds": multiple_bonds,
        "groups": group_metadata,
    }
