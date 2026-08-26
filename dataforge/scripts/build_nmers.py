import argparse
import hashlib
import logging
import os
import glob
import json
import re
from pathlib import Path
import numpy as np
import multiprocessing
from logging import Logger
from collections import Counter

from os.path import join, dirname, basename
from typing import Dict, List, Optional, Union
from itertools import combinations, zip_longest

from dataforge.src import DataDict, intersect_rows_2d, dynamic_for_loop, fix_bonds, apply_replacements_fp
from dataforge.src.qchem_utils import prepare_qchem_input
from dataforge.src.logging import get_logger
from dataforge.src.nmers import Monomer, Multimer
from dataforge.src.generic import argofyinx, read_h5_file, write_h5_file
from dataforge.src.monomer_discovery import COVALENT_RADII, discover_monomer_groups


def main(args=None):
    args = parse_command_line(args)
    aliases = load_monomer_aliases(args.monomer_aliases_json)
    merges = load_monomer_merges(args.monomer_merges_json)

    if args.mode == 'discover':
        discover_monomers(
            input_filename=args.input,
            dataset_root=args.root,
            monomer_mode=args.monomer_mode,
            bond_order_mode=args.bond_order_mode,
            monomer_aliases=aliases,
            monomer_merges=merges,
            max_nmer_degree=args.max_order,
        )
    elif args.mode == 'build':
        nmer_sampling_conf = parse_sampling_specs(args.sampling) if args.sampling else args.order
        build_nmers(
            input_filename          = args.input,
            dataset_root            = args.root,
            nmer_sampling_conf      = nmer_sampling_conf,
            keep_only_monomer_names = args.keep,
            keep_nmer_names         = args.keep_nmer_names,
            monomer_mode            = args.monomer_mode,
            bond_order_mode         = args.bond_order_mode,
            monomer_aliases         = aliases,
            monomer_merges          = merges,
            max_processes           = args.max_processes,
            suffix                  = args.suffix,
            cap_after_build         = not args.skip_cap,
        )
    elif args.mode == 'cap':
        data_root = join(args.root, "data")
        build_xyz_capped_nmers(
            nmers_root=join(data_root, "xyz" + args.suffix),
            nmers_capped_root=join(data_root, "xyz_capped" + args.suffix),
            fit_poly_root=join(args.root, "fitting" + args.suffix, "poly" + args.suffix),
            logger=get_logger('02_cap_nmers.log', level=logging.DEBUG),
            max_processes=args.max_processes,
            selected_nmer_names=args.nmer_names,
        )
    elif args.mode == 'prepare_qchem':
        charges_dict = dict(DataDict.CHARGES_DICT)
        if args.charges_json:
            with open(args.charges_json) as charges_file:
                charges_dict.update(json.load(charges_file))
        prepare_qchem(
            dataset_root            = args.root,
            suffix                  = args.suffix,
            max_processes           = args.max_processes,
            nmers_capped_root       = args.nmers_capped_root,
            qchem_in_root           = args.qchem_in_root,
            qchem_min_in_root       = args.qchem_min_in_root,
            charges_dict            = charges_dict,
            qchem_mode              = args.qchem_mode,
            selected_nmer_names     = args.nmer_names,
        )

def parse_command_line(args=None):
    parser = argparse.ArgumentParser(
        description="""
        Build and cap sampled n-mers, or prepare their QChem input files.
    """
    )
    parser.add_argument(
        "mode",
        choices=['discover', 'build', 'cap', 'prepare_qchem'],
        help=(
            "'discover' catalogs monomers/n-mer types; 'build' samples n-mers and optionally caps them; 'cap' caps existing "
            "sampled n-mers; 'prepare_qchem' prepares QChem input files."
        ),
    )
    parser.add_argument(
        "-i",
        "--input",
        help="The `.npz` file saved in the previous step (required for 'build' mode).",
        required=False,
    )
    parser.add_argument(
        "-r",
        "--root",
        help="Root folder for all the dataset components that will be created.",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--order",
        nargs='+',
        help="Ordered list of integers, indicating the order of multimers to be built (1=monomer, 2=dimers, ...).",
        required=False,
    )
    parser.add_argument(
        "--sampling",
        nargs='+',
        help="Sampling specifications such as '1=5000' or '3=10000:FPS'. Overrides --order.",
        required=False,
    )
    parser.add_argument(
        "-k",
        "--keep",
        nargs='+',
        help="Optional list with the monomer names to keep. All other monomers will be ignored.",
        default=None,
    )
    parser.add_argument(
        "--keep-nmer-names",
        nargs='+',
        help="Optional exact n-mer type names to build at every requested order.",
        default=None,
    )
    parser.add_argument(
        "--nmer-names",
        nargs='+',
        default=None,
        help="Optional exact n-mer type names to process for file-based stages.",
    )
    parser.add_argument(
        "--monomer-mode",
        choices=["auto", "legacy"],
        default="auto",
        help="Discover monomer cores from connectivity/bond orders, or use the legacy POPC rules.",
    )
    parser.add_argument(
        "--bond-order-mode",
        choices=["auto", "topology", "geometry"],
        default="auto",
        help="Use topology bond orders when available, infer them from geometry, or force single-bond fallback.",
    )
    parser.add_argument(
        "--monomer-aliases-json",
        default=None,
        help="JSON object mapping automatically generated monomer names to user names.",
    )
    parser.add_argument(
        "--monomer-merges-json",
        default=None,
        help="JSON metadata defining connected discovered monomer types to merge.",
    )
    parser.add_argument(
        "--max-order",
        type=int,
        default=3,
        help="Largest connected n-mer order to include in discovery metadata.",
    )
    parser.add_argument(
        "--skip-cap",
        action="store_true",
        help="Build sampled XYZ/HDF5 n-mers without performing the initial capping stage.",
    )
    parser.add_argument(
        "-s",
        "--suffix",
        help="Suffix to append to the paths for capped nmers, QChem input, and QChem min input.",
        default="",
    )
    parser.add_argument(
        "-p",
        "--max-processes",
        help="Maximum number of processes to use.",
        type=int,
        default=0,
    )
    parser.add_argument("--nmers-capped-root", default=None)
    parser.add_argument("--qchem-in-root", default=None)
    parser.add_argument("--qchem-min-in-root", default=None)
    parser.add_argument(
        "--charges-json",
        default=None,
        help="Optional JSON object mapping monomer names to integer QChem charges.",
    )
    parser.add_argument(
        "--qchem-mode",
        choices=["filtered", "minimization", "full"],
        default="filtered",
        help=(
            "Prepare inputs selected by .list files, one optimization per "
            "n-mer folder, or all frames already sampled into HDF5 files."
        ),
    )
    
    return parser.parse_args(args=args)


def load_monomer_aliases(filename: Optional[str]) -> Dict[str, str]:
    if not filename:
        return {}
    with open(filename) as aliases_file:
        aliases = json.load(aliases_file)
    if not isinstance(aliases, dict) or not all(
        isinstance(key, str) and isinstance(value, str) and value
        for key, value in aliases.items()
    ):
        raise ValueError("Monomer aliases must be a JSON object of non-empty string pairs.")
    invalid = [value for value in aliases.values() if re.fullmatch(r"[A-Za-z0-9_+-]+", value) is None]
    if invalid:
        raise ValueError("Monomer aliases may contain only letters, numbers, '_', '+', and '-'.")
    values = list(aliases.values())
    if len(values) != len(set(values)):
        raise ValueError("Two automatic monomer names cannot share the same alias.")
    return aliases


def load_monomer_merges(filename: Optional[str]) -> List[dict]:
    if not filename:
        return []
    with open(filename) as merge_file:
        data = json.load(merge_file)
    merges = data.get("merges", data) if isinstance(data, dict) else data
    if not isinstance(merges, list):
        raise ValueError("Monomer merges must be a list or an object containing a 'merges' list.")
    validated = []
    known_names = set()
    for entry in merges:
        if not isinstance(entry, dict):
            raise ValueError("Every monomer merge must be an object.")
        name = entry.get("name")
        members = entry.get("members")
        if re.fullmatch(r"[A-Za-z0-9_+-]+", str(name or "")) is None:
            raise ValueError(f"Invalid merged monomer name: {name!r}.")
        if (
            not isinstance(members, list)
            or not all(isinstance(member, str) and member for member in members)
            or len(set(members)) < 2
        ):
            raise ValueError(f"Merge {name!r} must contain at least two distinct member names.")
        if name in known_names:
            raise ValueError(f"Duplicate merged monomer name: {name!r}.")
        known_names.add(name)
        validated.append({"name": name, "members": list(dict.fromkeys(members))})
    return validated


def parse_sampling_specs(specs: List[str]) -> Dict[int, Union[int, tuple]]:
    """Parse CLI sampling specifications of the form ``ORDER=N[:METHOD]``."""
    result = {}
    for spec in specs:
        if '=' not in spec:
            raise ValueError(f"Invalid sampling specification {spec!r}; expected ORDER=N[:METHOD].")
        order_text, value_text = spec.split('=', 1)
        order = int(order_text)
        value_parts = value_text.split(':', 1)
        n_samples = int(value_parts[0])
        if n_samples < 1:
            raise ValueError(f"Number of samples must be positive in {spec!r}.")
        method = (value_parts[1] if len(value_parts) == 2 and value_parts[1] else 'US').upper()
        result[order] = (n_samples, method)
    return result

def build_nmers(
    input_filename: str,
    dataset_root: str,
    nmer_sampling_conf: Union[List[int], Dict[int, int]],
    keep_only_monomer_names: Optional[List[str]] = None,
    keep_nmer_names: Optional[List[str]] = None,
    monomer_mode: str = "legacy",
    bond_order_mode: str = "auto",
    monomer_aliases: Optional[Dict[str, str]] = None,
    monomer_merges: Optional[List[dict]] = None,
    max_processes: int = 0,
    suffix: str = "",
    cap_after_build: bool = True,
    **kwargs
):
    logger = get_logger('02_build_nmers.log', level=logging.DEBUG)
    
    DATA_ROOT         =  join(dataset_root, "data"           )
    NMERS_ROOT        =  join(DATA_ROOT   , "xyz"            + suffix)
    NMERS_CAPPED_ROOT =  join(DATA_ROOT   , "xyz_capped"     + suffix)

    FIT_ROOT           =                                          join(dataset_root, "fitting" + suffix)
    FIT_POLY_ROOT      = kwargs.get('FIT_POLY_ROOT',     None) or join(FIT_ROOT    , "poly"    + suffix)

    if isinstance(nmer_sampling_conf, list):
        nmer_sampling_conf = {int(k): None for k in nmer_sampling_conf}

    automatic_sampling = not all(v is None for v in nmer_sampling_conf.values())

    build_xyz_nmers(
        traj_dataset_filename   = input_filename,
        data_root               = DATA_ROOT,
        nmers_root              = NMERS_ROOT,
        monomers_dict           = DataDict.MONOMERS_DICT,
        nmer_sampling_conf      = nmer_sampling_conf,
        logger                  = logger,
        keep_only_monomer_names = keep_only_monomer_names,
        keep_nmer_names         = keep_nmer_names,
        monomer_mode            = monomer_mode,
        bond_order_mode         = bond_order_mode,
        monomer_aliases         = monomer_aliases,
        monomer_merges          = monomer_merges,
        compute_descriptors     = automatic_sampling,
        max_processes           = max_processes,
    )

    if cap_after_build:
        build_xyz_capped_nmers(
            nmers_root              = NMERS_ROOT,
            nmers_capped_root       = NMERS_CAPPED_ROOT,
            fit_poly_root           = FIT_POLY_ROOT,
            logger                  = logger,
            max_processes           = max_processes,
        )

    logger.info("- Complete!")

def prepare_qchem(
    dataset_root: str,
    suffix: str = "",
    max_processes: int = 0,
    nmers_capped_root: Optional[str] = None,
    qchem_in_root: Optional[str] = None,
    qchem_min_in_root: Optional[str] = None,
    charges_dict: Optional[dict] = None,
    qchem_mode: str = "filtered",
    selected_nmer_names: Optional[List[str]] = None,
):
    logger = get_logger('02_prepare_qchem.log', level=logging.DEBUG)
    
    DATA_ROOT         =  join(dataset_root, "data"                    )
    NMERS_CAPPED_ROOT = nmers_capped_root or join(DATA_ROOT, "xyz_capped", suffix)
    QCHEM_IN_ROOT     = qchem_in_root or join(DATA_ROOT, "qchem_input", suffix)
    QCHEM_MIN_IN_ROOT = qchem_min_in_root or join(DATA_ROOT, "qchem_min_input", suffix)

    prepare_qchem_input(
        nmers_capped_root       = NMERS_CAPPED_ROOT,
        qchem_in_root           = QCHEM_IN_ROOT,
        qchem_min_in_root       = QCHEM_MIN_IN_ROOT,
        charges_dict            = charges_dict or DataDict.CHARGES_DICT,
        max_processes           = max_processes,
        skip_if_not_frame_filter = qchem_mode == "filtered",
        minimization_only        = qchem_mode == "minimization",
        create_minimization_inputs = qchem_mode == "filtered",
        selected_nmer_names       = selected_nmer_names,
    )
    logger.info("- Complete!")

def build_xyz_nmers(
    traj_dataset_filename: str,
    data_root: str,
    nmers_root: str,
    monomers_dict: dict,
    nmer_sampling_conf: dict,
    logger: Logger,
    keep_only_monomer_names: Optional[List[str]] = None,
    keep_nmer_names: Optional[List[str]] = None,
    monomer_mode: str = "legacy",
    bond_order_mode: str = "auto",
    monomer_aliases: Optional[Dict[str, str]] = None,
    monomer_merges: Optional[List[dict]] = None,
    compute_descriptors: bool = True,
    max_processes: int = 0
):
    logger.info("- Building nmers...")

    # ------------------ L O A D    D A T A S E T -------------------- #
    # ---------------------------------------------------------------- #

    if not os.path.exists(data_root):
        os.makedirs(data_root, exist_ok=True)

    logger.info("-- Loading trajectory...")
    traj_dataset = dict(np.load(traj_dataset_filename, allow_pickle=True))

    # ---------------------------------------------------------------- #
    # ---------------------------------------------------------------- #

    logger.info("-- Building monomers...")
    if monomer_mode == "auto":
        monomers, discovery_metadata = build_auto_monomers(
            dataset=traj_dataset,
            logger=logger,
            compute_descriptors=False,
            bond_order_mode=bond_order_mode,
            aliases=monomer_aliases,
            merge_definitions=monomer_merges,
        )
        if keep_only_monomer_names is not None:
            selected_names = set(keep_only_monomer_names)
            monomers = [monomer for monomer in monomers if monomer.name in selected_names]
            discovery_metadata["selected_monomer_names"] = sorted(selected_names)
    else:
        monomers = build_monomers(
            dataset=traj_dataset,
            monomers_dict=monomers_dict,
            logger=logger,
            keep_only_monomer_names=keep_only_monomer_names,
            compute_descriptors=False,
        )
        discovery_metadata = {
            "mode": "legacy",
            "monomer_names": sorted({monomer.name for monomer in monomers}),
        }
    with open(os.path.join(data_root, "monomer_discovery.json"), "w") as discovery_file:
        json.dump(discovery_metadata, discovery_file, indent=2)

    # --- Create mapping between monomer indices and monomer names --- #
    # ---------------------------------------------------------------- #
    
    build_topology(monomers, data_root, max_nmer_degree=max(nmer_sampling_conf))

    # ---------------------------------------------------------------- #
    # ---------------------------------------------------------------- #

    logger.info("-- Building multimers...")
    build_multimers(
        nmers_root,
        nmer_sampling_conf,
        monomers,
        traj_dataset["position"],
        traj_dataset["atom_type"],
        compute_descriptors=compute_descriptors,
        logger=logger,
        max_processes=max_processes,
        keep_nmer_names=keep_nmer_names,
    )
    logger.info("- Completed building nmers!")

def _automatic_group_name(heavy_atom_indices, atom_types, local_to_name):
    heavy_atom_names = [
        local_to_name.get(index, f"{atom_types[index]}-{index}")
        for index in heavy_atom_indices
    ]
    if len(heavy_atom_indices) == 1:
        return heavy_atom_names[0], heavy_atom_names
    return "AUTO-" + "__".join(sorted(heavy_atom_names)), heavy_atom_names


def _merge_discovered_groups(groups, names, bonds, merge_definitions):
    """Apply named merges only to connected components containing every requested type."""
    records = [
        {"indices": list(group), "name": name, "merged_from": [name]}
        for group, name in zip(groups, names)
    ]
    merge_metadata = []
    for definition in merge_definitions or []:
        member_names = set(definition["members"])
        eligible = {i for i, record in enumerate(records) if record["name"] in member_names}
        atom_to_record = {
            atom: record_index
            for record_index, record in enumerate(records)
            for atom in record["indices"]
        }
        adjacency = {record_index: set() for record_index in eligible}
        for left, right in np.asarray(bonds, dtype=int):
            left_record = atom_to_record.get(int(left))
            right_record = atom_to_record.get(int(right))
            if left_record in eligible and right_record in eligible and left_record != right_record:
                adjacency[left_record].add(right_record)
                adjacency[right_record].add(left_record)

        components = []
        unseen = set(eligible)
        while unseen:
            start = unseen.pop()
            component = {start}
            stack = [start]
            while stack:
                current = stack.pop()
                neighbours = adjacency[current] & unseen
                unseen -= neighbours
                component |= neighbours
                stack.extend(neighbours)
            if member_names <= {records[i]["name"] for i in component}:
                components.append(component)

        if not components:
            raise ValueError(
                f"Merge {definition['name']!r} did not match a connected component "
                f"containing {sorted(member_names)}."
            )
        consumed = set().union(*components)
        if any(
            record["name"] == definition["name"]
            for i, record in enumerate(records)
            if i not in consumed
        ):
            raise ValueError(
                f"Merged monomer name {definition['name']!r} conflicts with an unmerged type."
            )
        new_records = [record for i, record in enumerate(records) if i not in consumed]
        for component in components:
            component_records = [records[i] for i in sorted(component)]
            new_records.append({
                "indices": sorted({atom for record in component_records for atom in record["indices"]}),
                "name": definition["name"],
                "merged_from": sorted({name for record in component_records for name in record["merged_from"]}),
            })
        records = sorted(new_records, key=lambda record: min(record["indices"]))
        merge_metadata.append({
            **definition,
            "matched_components": len(components),
            "matched_group_count": len(consumed),
        })
    return records, merge_metadata


def _infer_formal_charges(monomers, dataset, bond_orders):
    """Suggest integer formal charges from full-topology bond-order sums."""
    atom_types = np.asarray(dataset["atom_type"]).astype(str)
    normalised_types = np.asarray([
        value[0].upper() + value[1:].lower() for value in atom_types
    ])
    bonds = np.asarray(dataset["bond_indices"], dtype=int)
    neighbours = [set() for _ in atom_types]
    for left, right in bonds:
        neighbours[int(left)].add(int(right))
        neighbours[int(right)].add(int(left))
    bond_sums = np.zeros(len(atom_types), dtype=float)
    for (left, right), order in zip(bonds, bond_orders):
        bond_sums[left] += float(order)
        bond_sums[right] += float(order)
    rules = {
        "H": {1: 0}, "B": {3: 0, 4: -1}, "C": {4: 0},
        "N": {2: -1, 3: 0, 4: 1}, "O": {1: -1, 2: 0, 3: 1},
        "P": {3: 0, 4: 1, 5: 0}, "S": {1: -1, 2: 0, 4: 0, 6: 0},
        "F": {0: -1, 1: 0}, "Cl": {0: -1, 1: 0},
        "Br": {0: -1, 1: 0}, "I": {0: -1, 1: 0},
    }
    phosphate_centers = {}
    for atom_index, element in enumerate(normalised_types):
        if element != "P":
            continue
        oxygen_indices = sorted(
            neighbour for neighbour in neighbours[atom_index]
            if normalised_types[neighbour] == "O"
        )
        if len(oxygen_indices) != 4:
            continue
        neutralised_oxygen_count = sum(
            any(
                other != atom_index
                for other in neighbours[oxygen_index]
            )
            for oxygen_index in oxygen_indices
        )
        phosphate_centers[atom_index] = {
            "oxygen_indices": oxygen_indices,
            "neutralised_oxygen_count": int(neutralised_oxygen_count),
            "formal_charge": int(neutralised_oxygen_count - 3),
        }
    occurrences = {}
    evidence = []
    for monomer in monomers:
        charge = 0
        uncertain = []
        atom_evidence = []
        atom_charges = {}
        heavy_atom_indices = np.asarray(monomer.heavy_atoms_idcs, dtype=int).reshape(-1)
        for atom_index in heavy_atom_indices:
            normalised = normalised_types[atom_index]
            rounded_sum = int(round(float(bond_sums[atom_index])))
            atom_charge = rules.get(normalised, {}).get(rounded_sum)
            if atom_charge is None:
                atom_charge = 0
                uncertain.append(int(atom_index))
            atom_charges[int(atom_index)] = int(atom_charge)
            charge += atom_charge
            atom_evidence.append({
                "atom_index": int(atom_index),
                "element": str(normalised),
                "bond_order_sum": float(bond_sums[atom_index]),
                "formal_charge": int(atom_charge),
                "rule_matched": int(atom_index) not in uncertain,
            })
        special_rules = []
        covered_atoms = set()
        centers_in_monomer = [
            int(atom_index) for atom_index in heavy_atom_indices
            if int(atom_index) in phosphate_centers
        ]
        if centers_in_monomer:
            charge = 0
            for center_index in centers_in_monomer:
                rule = phosphate_centers[center_index]
                covered_atoms.add(center_index)
                covered_atoms.update(rule["oxygen_indices"])
                charge += rule["formal_charge"]
                special_rules.append({
                    "rule": "phosphate charge from protonated/substituted oxygen count",
                    "center_atom_index": center_index,
                    **rule,
                })
            charge += sum(
                atom_charge for atom_index, atom_charge in atom_charges.items()
                if atom_index not in covered_atoms
            )
            uncertain = [atom_index for atom_index in uncertain if atom_index not in covered_atoms]
            for atom_item in atom_evidence:
                if atom_item["atom_index"] in covered_atoms:
                    atom_item["covered_by_special_rule"] = True
        occurrences.setdefault(monomer.name, []).append(int(charge))
        evidence.append({
            "monomer_id": int(monomer.id), "name": monomer.name,
            "suggested_charge": int(charge),
            "confidence": "high" if not uncertain else "review",
            "atoms": atom_evidence,
            "special_rules": special_rules,
        })
    inferred = {
        name: charges[0]
        for name, charges in occurrences.items()
        if len(set(charges)) == 1
    }
    inconsistent = {
        name: sorted(set(charges)) for name, charges in occurrences.items() if len(set(charges)) > 1
    }
    return inferred, {
        "method": "formal charge from full-topology integer bond-order sums, with phosphate protonation/substitution rules",
        "review_required": bool(inconsistent) or any(item["confidence"] != "high" for item in evidence),
        "inconsistent_type_charges": inconsistent,
        "occurrences": evidence,
    }


def build_auto_monomers(
        dataset: dict,
        logger: Logger,
        compute_descriptors: bool = True,
        bond_order_mode: str = "auto",
        aliases: Optional[Dict[str, str]] = None,
        merge_definitions: Optional[List[dict]] = None,
    ):
    groups, discovery_metadata = discover_monomer_groups(
        dataset,
        mode=bond_order_mode,
    )
    atom_types = np.asarray(dataset["atom_type"])
    atom_names = np.asarray(dataset["atom_name"])
    atom_orig_indices = np.asarray(dataset["atom_orig_index"])
    local_to_name = {
        int(local_index): str(name)
        for name, local_index in zip(
            dataset["monomer_names"],
            dataset["monomer_orig_atom_index"],
        )
    }

    aliases = aliases or {}
    base_automatic_names = []
    base_heavy_names = []
    for heavy_atom_indices in groups:
        automatic_name, heavy_names = _automatic_group_name(
            heavy_atom_indices, atom_types, local_to_name
        )
        base_automatic_names.append(automatic_name)
        base_heavy_names.append(heavy_names)
    assigned_base_names = [aliases.get(name, name) for name in base_automatic_names]
    records, merge_metadata = _merge_discovered_groups(
        groups, assigned_base_names, dataset["bond_indices"], merge_definitions
    )

    monomers = []
    for monomer_id, record in enumerate(records):
        heavy_atom_indices = record["indices"]
        heavy_atom_names = [
            local_to_name.get(index, f"{atom_types[index]}-{index}")
            for index in heavy_atom_indices
        ]
        monomer = Monomer(
            id=monomer_id,
            name=record["name"],
            heavy_atoms_names=heavy_atom_names,
            heavy_atoms_idcs=heavy_atom_indices,
            orig_all_idcs=atom_orig_indices,
            orig_all_bond_idcs=dataset["bond_orig_indices"],
            orig_all_angle_idcs=dataset["angle_orig_indices"],
            orig_all_dihedral_idcs=dataset["dihedral_orig_indices"],
            orig_all_atom_names=atom_names,
            orig_all_atom_types=atom_types,
        )
        if compute_descriptors:
            monomer.compute_descriptors(orig_all_pos=dataset["position"])
        monomers.append(monomer)

    discovery_metadata["automatic_monomer_names"] = base_automatic_names
    discovery_metadata["monomer_names"] = [monomer.name for monomer in monomers]
    discovery_metadata["monomer_aliases"] = {
        name: aliases.get(name, name)
        for name in sorted(set(base_automatic_names))
        if name.startswith("AUTO-")
    }
    assigned_by_automatic_name = {
        automatic_name: aliases.get(automatic_name, automatic_name)
        for automatic_name in set(base_automatic_names)
    }
    if len(set(assigned_by_automatic_name.values())) != len(assigned_by_automatic_name):
        raise ValueError(
            "Monomer aliases collapse distinct discovered monomer types onto the same name."
        )
    discovery_metadata["monomer_count"] = len(monomers)
    discovery_metadata["monomer_merges"] = merge_metadata
    inferred_charges, charge_evidence = _infer_formal_charges(
        monomers, dataset, discovery_metadata["bond_orders"]
    )
    discovery_metadata["inferred_charges"] = inferred_charges
    discovery_metadata["charge_inference"] = charge_evidence
    logger.info(
        "--- Automatic monomer discovery: %d cores, %d multiple bonds (%s) ---",
        len(monomers),
        len(discovery_metadata["multiple_bonds"]),
        discovery_metadata["bond_order_evidence"]["source"],
    )
    return monomers, discovery_metadata

def build_monomers(
        dataset: dict,
        monomers_dict: dict,
        logger: Logger,
        keep_only_monomer_names: Optional[List[str]] = None,
        compute_descriptors: bool = True,
    ) -> List[Monomer]:

    monomer_names =           dataset["monomer_names"]
    monomer_orig_atom_index = dataset["monomer_orig_atom_index"]
    orig_all_pos =            dataset["position"]
    orig_all_idcs =           dataset["atom_orig_index"]
    orig_all_atom_names =     dataset["atom_name"]
    orig_all_atom_types =     dataset["atom_type"]
    orig_all_bond_idcs =      dataset['bond_orig_indices']
    orig_all_angle_idcs =     dataset['angle_orig_indices']
    orig_all_dihedral_idcs =  dataset['dihedral_orig_indices']

    # ----------- B U I L D   M O N O M E R S   D I C T -------------- #
    # ---------------------------------------------------------------- #

    # Work on a copy: the legacy dictionary is a package constant and should
    # not accumulate monomer types across independent workflow runs.
    monomers_dict = {
        key: dict(value)
        for key, value in monomers_dict.items()
    }

    # Configure excluded monomers
    excluded_monomers = set()

    # Update excluded monomers with monomers that appear in composite-monomers
    for composite_monomer_name in monomers_dict.keys():
        excluded_monomers.update(monomers_dict[composite_monomer_name]["composite_monomer_names"])

    # Update monomers dict with simple monomers
    for monomer_name in np.unique(monomer_names):
        if keep_only_monomer_names is not None and monomer_name not in keep_only_monomer_names:
            continue
        if monomer_name in excluded_monomers:
            continue
        monomers_dict.update({
            monomer_name: {
                "composite_monomer_names": [monomer_name],
                "composite_monomer_elements": 1,
                "composite_monomer_bonds": 0,
            }
        })
    
    if keep_only_monomer_names is not None:
        logger.info(f"--- Keeping only the following monomers: {keep_only_monomer_names}")
        monomers_dict = {k: v for k, v in monomers_dict.items() if k in keep_only_monomer_names}
    
    # ---------------------------------------------------------------- #
    # ---------------------------------------------------------------- #

    # - NOT OPTIMAL ALGORITHM, COULD PICK WRONG COMPOSITE MONOMERS (NO TOPOLOGY SPECIFIED) - #
    id = 0
    monomers: List[Monomer] = []

    for mn in monomers_dict: # mn = monomer_name
        all_candidate_monomer_idcs = np.argwhere(np.isin(monomer_names, monomers_dict[mn]["composite_monomer_names"])).flatten()
        orig_all_candidate_monomer_idcs = monomer_orig_atom_index[all_candidate_monomer_idcs]
        idx_to_monomer_idx = {k: v for k, v in zip(orig_all_candidate_monomer_idcs, all_candidate_monomer_idcs)}
        monomer_indexed_bond_idcs = dataset["bond_orig_indices"]

        for candidate_composite_monomer_idcs in list(combinations(orig_all_candidate_monomer_idcs, monomers_dict[mn]["composite_monomer_elements"])):
            number_of_heavy_bonds = 0
            for monomer_pair in combinations(candidate_composite_monomer_idcs, 2):
                monomer_pair = np.array(monomer_pair, dtype=monomer_indexed_bond_idcs.dtype)
                if len(intersect_rows_2d(monomer_indexed_bond_idcs, monomer_pair)) > 0 or \
                len(intersect_rows_2d(monomer_indexed_bond_idcs, np.ascontiguousarray(monomer_pair[::-1]))) > 0:
                    number_of_heavy_bonds += 1
                    if number_of_heavy_bonds > monomers_dict[mn]["composite_monomer_bonds"]:
                        break
            if number_of_heavy_bonds == monomers_dict[mn]["composite_monomer_bonds"]:
                candidate_composite_monomer_names = [monomer_names[idx_to_monomer_idx[idx]] for idx in candidate_composite_monomer_idcs]
                monomer = Monomer(
                    id=id,
                    name=mn,
                    heavy_atoms_names=candidate_composite_monomer_names,
                    heavy_atoms_idcs=candidate_composite_monomer_idcs,
                    orig_all_idcs=orig_all_idcs,
                    orig_all_bond_idcs=orig_all_bond_idcs,
                    orig_all_angle_idcs=orig_all_angle_idcs,
                    orig_all_dihedral_idcs=orig_all_dihedral_idcs,
                    orig_all_atom_names=orig_all_atom_names,
                    orig_all_atom_types=orig_all_atom_types,
                )
                if compute_descriptors:
                    monomer.compute_descriptors(orig_all_pos=orig_all_pos)
                monomers.append(monomer)
                id += 1
    return monomers

def monomers_are_valid(monomers_tuple):

    # --- Check if monomers_tuple is empty --- #

    if len(monomers_tuple) == 0:
        return False

    # # --- Check if same Monomer appears more than once --- #

    # if any([m1.id == m2.id for m1, m2 in combinations(monomers_tuple, 2)]):
    #     return False
    
    # --- Check if all n monomers are connected by at least one bond, otherwise ignore this nmer --- #

    monomers_connections = {}
    for monomer_1, monomer_2 in combinations(monomers_tuple, 2):
        if len(intersect_rows_2d(monomer_1.orig_bond_idcs, monomer_2.orig_bond_idcs)) > 0:
            monomers_connections[monomer_1.id] = monomers_connections.get(monomer_1.id, 0) + 1
            monomers_connections[monomer_2.id] = monomers_connections.get(monomer_2.id, 0) + 1
        else:
            monomers_connections[monomer_1.id] = monomers_connections.get(monomer_1.id, 0)
            monomers_connections[monomer_2.id] = monomers_connections.get(monomer_2.id, 0)
    if any([v == 0 for v in monomers_connections.values()]):
        return False
    return True

def build_topology(monomers: List[Monomer], data_root: str, max_nmer_degree: Optional[int] = None):
    max_nmer_degree = max_nmer_degree or max(DataDict.FOLDER_NAMES.keys())
    topology = {"monomers": {m.id: m.name for m in monomers}}

    def build_monomers_topology(monomers: List[Monomer], connections: List[str]):
        output_mapping = {"connections": connections}
        m_idcs = np.array([m.id for m in monomers])
        if len(m_idcs) != len(np.unique(m_idcs)):
            return output_mapping
        connection = '_'.join([str(id) for id in np.sort(m_idcs)])
        if connection in connections:
            return output_mapping
        for m1, m2 in zip(monomers[:-1], monomers[1:]):
            if not np.any(np.isin(m1.orig_connected_atoms_idcs, m2.orig_heavy_atoms_idcs)):
                return output_mapping
        connections.append(connection)
        return output_mapping

    all_connections = {"connections": []}
    for nmer_degree in range(2, max_nmer_degree + 1):
        connections = dynamic_for_loop(
            iterable=monomers,
            num_for_loops=nmer_degree,
            func=build_monomers_topology,
            connections=[],
        )
        all_connections["connections"].extend(connections["connections"])
    topology.update(all_connections)
    
    with open(os.path.join(data_root, DataDict.TOPOLOGY_FILENAME), "w") as topology_f:
        json.dump(topology, topology_f, indent=4)


def candidate_nmer_names(
    monomers: List[Monomer], atom_types, max_nmer_degree: int, logger: Logger
) -> Dict[str, List[str]]:
    """Catalog connected n-mer type names without allocating trajectory descriptors."""
    candidates = {}
    for order in range(1, max_nmer_degree + 1):
        names = set()
        for monomer_tuple in combinations(monomers, order):
            if order > 1 and not monomers_are_valid(monomer_tuple):
                continue
            names.add(Multimer(monomer_tuple, atom_types, logger=logger).name)
        candidates[str(order)] = sorted(names)
    return candidates


def discover_monomers(
    input_filename: str,
    dataset_root: str,
    monomer_mode: str = "auto",
    bond_order_mode: str = "auto",
    monomer_aliases: Optional[Dict[str, str]] = None,
    monomer_merges: Optional[List[dict]] = None,
    max_nmer_degree: int = 3,
):
    if not input_filename:
        raise ValueError("--input is required for discovery mode.")
    if max_nmer_degree < 1:
        raise ValueError("--max-order must be positive.")
    logger = get_logger('02_discover_monomers.log', level=logging.DEBUG)
    dataset = dict(np.load(input_filename, allow_pickle=True))
    if monomer_mode == "auto":
        monomers, metadata = build_auto_monomers(
            dataset,
            logger,
            compute_descriptors=False,
            bond_order_mode=bond_order_mode,
            aliases=monomer_aliases,
            merge_definitions=monomer_merges,
        )
    else:
        monomers = build_monomers(
            dataset, DataDict.MONOMERS_DICT, logger, compute_descriptors=False
        )
        metadata = {"mode": "legacy", "monomer_names": [m.name for m in monomers]}
    metadata["candidate_nmers"] = candidate_nmer_names(
        monomers, dataset["atom_type"], max_nmer_degree, logger
    )
    metadata["max_nmer_degree"] = max_nmer_degree
    data_root = os.path.join(dataset_root, "data")
    os.makedirs(data_root, exist_ok=True)
    output = os.path.join(data_root, "monomer_discovery.json")
    with open(output, "w") as discovery_file:
        json.dump(metadata, discovery_file, indent=2)
    logger.info("Saved monomer and n-mer catalog to %s", output)
    return metadata

def save_multimer(
    nmers_root: str,
    folder_name: str,
    multimer: Multimer,
    multimer_sampled_indices: Optional[np.ndarray],
    multimer_coords: np.ndarray,
    multimer_atom_types: np.ndarray,
    logger: Logger,
    coords_are_sampled: bool = False,
):

    # - Create folder - #
    nmer_folder = os.path.join(nmers_root, folder_name, multimer.name)
    if not os.path.isdir(nmer_folder): os.makedirs(nmer_folder, exist_ok=True)
    h5_filename = os.path.join(nmer_folder, multimer.h5_filename)
    if os.path.isfile(h5_filename): logger.warning(f"File {h5_filename} exists alreeady. Overwriting...")
    if multimer_sampled_indices is None: multimer_sampled_indices = np.arange(len(multimer_coords))
        
    # Index of all monomer atoms, relative to multimer atoms only
    severed_idcs = np.argwhere(np.isin(multimer.orig_all_atoms_idcs,multimer.orig_connected_atoms_idcs)).flatten()
    severed_names = np.zeros((len(severed_idcs),), dtype=object)
    severed_bonded_idcs = get_bonded_idcs(severed_idcs, multimer, multimer)

    info_dict = {}
    for m_id, monomer in enumerate(multimer._monomers):
        # Index of all monomer atoms, relative to multimer atoms only
        monomer_idcs = np.argwhere(np.isin(multimer.orig_all_atoms_idcs, monomer.orig_atoms_idcs)).flatten()
        for severed_atom_idx, mocai in enumerate(multimer.orig_connected_atoms_idcs):
            if mocai in monomer.orig_connected_atoms_idcs:
                severed_names[severed_atom_idx] = monomer.name
        info_dict[f"monomer_{m_id + 1}_name"] = monomer.name
        info_dict[f"monomer_{m_id + 1}_idcs"]  = monomer_idcs.tolist()
        info_dict[f"monomer_{m_id + 1}_bonded_idcs"] = get_bonded_idcs(monomer_idcs, multimer, monomer).tolist()
    
    info_dict[f"severed_name"] = severed_names.tolist()
    info_dict[f"severed_idcs"] = severed_idcs.tolist()
    info_dict[f"severed_bonded_idcs"] = severed_bonded_idcs.tolist()
    
    if coords_are_sampled:
        coords = np.asarray(multimer_coords)
    else:
        coords = np.asarray(multimer_coords[multimer_sampled_indices])
    string_dtype = getattr(np, "string_", np.bytes_)
    atom_types = np.asarray(multimer_atom_types, dtype=string_dtype)
    fullnames  = np.asarray([f"{str(frame_id)}_{multimer.fullname}" for frame_id in multimer_sampled_indices], dtype=string_dtype)

    # Save to h5 file
    write_h5_file(h5_filename, coords, atom_types, fullnames, info_dict)

def get_bonded_idcs(
    idcs: np.ndarray,
    multimer: Multimer,
    nmer: Union[Monomer, Multimer]
):
    bonded_idcs_list = []
    for orig_id in multimer.orig_all_atoms_idcs[idcs]:

        bonded_idcs = np.unique(nmer.orig_bond_idcs[np.any(nmer.orig_bond_idcs == orig_id, axis=1)])
        bonded_idcs = bonded_idcs[bonded_idcs != orig_id]
        bonded_idcs = np.argwhere(
            np.isin(
                multimer.orig_all_atoms_idcs,
                bonded_idcs
            )
        ).flatten()

        bonded_idcs_list.append(bonded_idcs.tolist())
    return np.array(list(zip_longest(*bonded_idcs_list, fillvalue=-1))).T

def build_multimer_recursively(
    nmers_root: str,
    nmer_sampling_conf: dict,
    monomers: List[Monomer],
    n: int,
    folder_name: str,
    orig_pos: np.ndarray,
    orig_all_atom_types: np.ndarray,
    logger: Logger,
    recursive_multimer_sampled_indices = None,
    compute_descriptors: bool = True,
    max_processes: int = 0,
    keep_nmer_names: Optional[List[str]] = None,
    keep_nmer_order: Optional[int] = None,
):
    if n not in nmer_sampling_conf:
        return

    n_samples, method = parse_nmer_sampling_conf(nmer_sampling_conf[n])
    
    multimers: List[Multimer] = []
    multimers_occurrence = {}
    multimers_first_occurrence = {}
    for monomers_tuple in combinations(monomers, n):
        if not monomers_are_valid(monomers_tuple):
            continue

        # - Create Multimer to join descriptors distributions of Monomers - #
        multimer = Multimer(monomers_tuple, orig_all_atom_types, logger=logger)
        if (
            keep_nmer_names is not None
            and (keep_nmer_order is None or n == keep_nmer_order)
            and multimer.name not in keep_nmer_names
        ):
            continue
        multimers.append(multimer)
        
        # - Count occurrences of each multimer - #
        counts = multimers_occurrence.get(multimer.name, 0) + 1
        multimers_occurrence[multimer.name] = counts
        multimers_first_occurrence[multimer.name] = True

    prepared_tasks = []
    for multimer in multimers:
        needs_sampling_descriptors = (
            compute_descriptors
            and recursive_multimer_sampled_indices is None
            and n_samples is not None
        )
        if needs_sampling_descriptors:
            # Compute only the current n-mer's descriptors. Keeping descriptor
            # arrays on every monomer scales with trajectory_frames * topology
            # and exhausted RAM for multi-million-frame trajectories.
            multimer.compute_descriptors(orig_all_pos=orig_pos)
        sampled_indices = select_multimer_sampled_indices(
            multimer=multimer,
            n_samples=n_samples,
            method=method,
            multimers_occurrence=multimers_occurrence,
            multimers_first_occurrence=multimers_first_occurrence,
            recursive_multimer_sampled_indices=recursive_multimer_sampled_indices,
        )
        # Sampling is complete; workers and recursion only need topology and
        # frame indices. Do not retain or pickle full-frame descriptor copies.
        multimer.bond_values = None
        multimer.angle_values = None
        multimer.dihedral_values = None
        prepared_tasks.append((multimer, sampled_indices))

    if max_processes <= 1:
        result = []
        for multimer, sampled_indices in prepared_tasks:
            # Slice one serial task at a time instead of retaining sampled
            # coordinates for every n-mer until all files have been written.
            task_coords = orig_pos if sampled_indices is None else orig_pos[sampled_indices]
            task_coords = task_coords[:, multimer.orig_all_atoms_idcs]
            res = process_multimer(
                multimer,
                nmers_root,
                folder_name,
                n_samples,
                method,
                multimers_occurrence,
                multimers_first_occurrence,
                task_coords,
                orig_all_atom_types[multimer.orig_all_atoms_idcs],
                recursive_multimer_sampled_indices,
                sampled_indices,
            )
            result.append(res)
    else:
        # Select frames before selecting atoms. This keeps worker tasks
        # proportional to the requested sample count rather than the full
        # trajectory length.
        worker_tasks = []
        for multimer, sampled_indices in prepared_tasks:
            task_coords = orig_pos if sampled_indices is None else orig_pos[sampled_indices]
            task_coords = task_coords[:, multimer.orig_all_atoms_idcs]
            worker_tasks.append((multimer, sampled_indices, task_coords))
        with multiprocessing.Pool(processes=max_processes) as pool:
            result = pool.starmap(
                process_multimer,
                [
                    (
                        multimer,
                        nmers_root,
                        folder_name,
                        n_samples,
                        method,
                        multimers_occurrence,
                        multimers_first_occurrence,
                        task_coords,
                        orig_all_atom_types[multimer.orig_all_atoms_idcs],
                        recursive_multimer_sampled_indices,
                        sampled_indices,
                    )
                    for multimer, sampled_indices, task_coords in worker_tasks
                ]
            )
    
    recursive_multimer_sampled_indices_list = []
    multimers = []
    for multimer, recursive_multimer_sampled_indices in result:
        multimers.append(multimer)
        recursive_multimer_sampled_indices_list.append(recursive_multimer_sampled_indices)
    
    if max_processes <= 1:
        for multimer, recursive_multimer_sampled_indices in zip(multimers, recursive_multimer_sampled_indices_list):
            build_multimer_recursively(
                nmers_root,
                nmer_sampling_conf,
                multimer._monomers,
                n-1,
                os.path.join(folder_name, DataDict.folder_name(n - 1)),
                orig_pos,
                orig_all_atom_types,
                logger,
                recursive_multimer_sampled_indices,
                compute_descriptors,
                0,
                keep_nmer_names,
                keep_nmer_order,
            )
    else:
        with multiprocessing.Pool(processes=max_processes) as pool:
            pool.starmap(
                build_multimer_recursively,
                [
                    (
                        nmers_root,
                        nmer_sampling_conf,
                        multimer._monomers,
                        n-1,
                        os.path.join(folder_name, DataDict.folder_name(n - 1)),
                        orig_pos,
                        orig_all_atom_types,
                        logger,
                        recursive_multimer_sampled_indices,
                        compute_descriptors,
                        0,
                        keep_nmer_names,
                        keep_nmer_order,
                    )
                    for multimer, recursive_multimer_sampled_indices in zip(multimers, recursive_multimer_sampled_indices_list)
                ]
            )

def parse_nmer_sampling_conf(x):
    if x is None:
        return None, 'ALL'
    if isinstance(x, int):
        return x, 'US'
    if isinstance(x, tuple):
        assert len(x) == 2
        assert isinstance(x[0], int)
        assert isinstance(x[1], str)
        return x
    if isinstance(x, dict):
        assert 'n' in x
        n = x['n']
        assert isinstance(n, int)
        if 'method' in x:
            method = x['method']
        else:
            method = 'US'
        assert isinstance(x['method'], str)
        return n, method
    raise Exception(f"Element of dict 'nmer_sampling_conf' with key {n} invalid. " +
                    "Should be either int, tuple of (int, str) or dict(n=n_samples(int), method=method_name(str))." +
                    "Got {type(x)}")

def configure_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(process)d - %(levelname)s - %(message)s')

def select_multimer_sampled_indices(
    multimer: Multimer,
    n_samples: Optional[int],
    method: str,
    multimers_occurrence: dict,
    multimers_first_occurrence: dict,
    recursive_multimer_sampled_indices,
):
    """Choose frame indices before any coordinate array is materialized."""
    if recursive_multimer_sampled_indices is not None:
        return recursive_multimer_sampled_indices
    if n_samples is None:
        return None

    multimer_n_samples = n_samples // multimers_occurrence.get(multimer.name)
    if multimers_first_occurrence.get(multimer.name):
        multimer_n_samples += n_samples % multimers_occurrence.get(multimer.name)
        multimers_first_occurrence[multimer.name] = False
    return multimer.sample(multimer_n_samples, method=method)

def process_multimer(
    multimer: Multimer,
    nmers_root,
    folder_name,
    n_samples,
    method,
    multimers_occurrence: dict,
    multimers_first_occurrence: dict,
    multimer_coords,
    multimer_atom_types,
    recursive_multimer_sampled_indices,
    sampled_indices_override=None,
):
    configure_logging()
    logger = logging.getLogger()
    logger.info(f"--- Saving {multimer.h5_filename} to {folder_name}...")

    if sampled_indices_override is not None:
        multimer_sampled_indices = sampled_indices_override
    elif recursive_multimer_sampled_indices is not None:
        multimer_sampled_indices = recursive_multimer_sampled_indices
    elif n_samples is None:
        multimer_sampled_indices = None
    else:
        multimer_n_samples = n_samples // multimers_occurrence.get(multimer.name)
        if multimers_first_occurrence.get(multimer.name):
            multimer_n_samples += n_samples % multimers_occurrence.get(multimer.name)
            multimers_first_occurrence[multimer.name] = False
        multimer_sampled_indices = multimer.sample(multimer_n_samples, method=method)

    save_multimer(
        nmers_root,
        folder_name,
        multimer,
        multimer_sampled_indices,
        multimer_coords,
        multimer_atom_types,
        logger,
        coords_are_sampled=True,
    )
    return multimer, multimer_sampled_indices

def build_multimers(
    nmers_root: str,
    nmer_sampling_conf: dict,
    monomers: List[Monomer],
    orig_pos: np.ndarray,
    orig_all_atom_types: np.ndarray,
    compute_descriptors: bool,
    logger: Logger,
    max_processes: int = 0,
    keep_nmer_names: Optional[List[str]] = None,
):
    # Each multiprocessing task currently receives a frame-by-atom coordinate
    # slice. For multi-gigabyte trajectories, constructing those slices for
    # many monomers at once can multiply memory use dramatically. Keep the
    # build deterministic and memory-bounded; capping/QChem stages can still
    # use their requested parallelism afterwards.
    large_trajectory_bytes = 2 * 1024**3
    if max_processes > 1 and orig_pos.nbytes > large_trajectory_bytes:
        logger.warning(
            "Trajectory coordinate array is %.2f GiB; serializing n-mer "
            "construction to avoid multiprocessing memory amplification.",
            orig_pos.nbytes / 1024**3,
        )
        max_processes = 1

    requested_orders = sorted(int(order) for order in nmer_sampling_conf)
    for n in requested_orders:
        folder_name = DataDict.folder_name(n)
        logger.info(f"--- Building Multimers of order {n}...")
        build_multimer_recursively(
            nmers_root=nmers_root,
            nmer_sampling_conf=nmer_sampling_conf,
            monomers=monomers,
            n=n,
            folder_name=folder_name,
            orig_pos=orig_pos,
            orig_all_atom_types=orig_all_atom_types,
            logger=logger,
            compute_descriptors=compute_descriptors,
            max_processes=max_processes,
            keep_nmer_names=keep_nmer_names,
            keep_nmer_order=None,
        )

def _distance_by_severed_order(info_dict: dict, capping_distances: Optional[dict]):
    severed_idcs = np.asarray(info_dict.get("severed_idcs", []), dtype=int)
    distances = [None] * len(severed_idcs)
    if capping_distances is None:
        return distances

    source_indices = capping_distances.get("source_severed_indices", [])
    measured_distances = capping_distances.get("distances", [])
    for source_index, distance in zip(source_indices, measured_distances):
        matches = np.flatnonzero(severed_idcs == int(source_index))
        if len(matches) == 1:
            distances[matches[0]] = float(distance)
    return distances


def _initial_capping_distance(atom_type: str):
    """Return a reasonable initial X-H distance and an auditable source."""
    atom_type = str(atom_type).strip()
    normalised = atom_type[:1].upper() + atom_type[1:].lower()
    if normalised in DataDict.ATOM_TYPE_TO_H_DISTANCE:
        return float(DataDict.ATOM_TYPE_TO_H_DISTANCE[normalised]), "calibrated_table"

    radius = COVALENT_RADII.get(normalised)
    hydrogen_radius = COVALENT_RADII.get("H")
    if radius is None or hydrogen_radius is None:
        supported = ", ".join(sorted(COVALENT_RADII))
        raise ValueError(
            f"Cannot choose an initial {normalised or atom_type}-H capping distance. "
            f"Add a calibrated value to DataDict.ATOM_TYPE_TO_H_DISTANCE or a "
            f"covalent radius to COVALENT_RADII. Supported elements: {supported}."
        )
    return float(radius + hydrogen_radius), "covalent_radii"


def substitute_severed_atoms(
    all_coords,
    atom_types,
    info_dict: dict,
    capping_distances: Optional[dict] = None,
    return_capping_metadata: bool = False,
):
    severed_idcs = np.asarray(info_dict.get("severed_idcs", []), dtype=int)
    nmer_idcs = np.delete(np.arange(all_coords.shape[1]), severed_idcs)
    nmer_coords = all_coords[:, nmer_idcs]
    severed_coords = all_coords[:, severed_idcs]
    distances_by_severed_order = _distance_by_severed_order(info_dict, capping_distances)
    severed_bonded_idcs = np.asarray(info_dict.get("severed_bonded_idcs", []), dtype=int)
    if severed_bonded_idcs.ndim == 1 and len(severed_bonded_idcs) > 0:
        severed_bonded_idcs = severed_bonded_idcs.reshape(1, -1)

    H_substituted_coords = np.zeros_like(severed_coords)
    applied_distances = []
    distance_sources = []
    for i, severed_atom_coords in enumerate(severed_coords.transpose(1, 0, 2)):
        distance_vectors = severed_atom_coords[:, None, :] - nmer_coords
        distances = np.linalg.norm(distance_vectors, axis=2)
        neighbour_candidates = []
        if i < len(severed_bonded_idcs):
            neighbour_candidates = [int(x) for x in severed_bonded_idcs[i] if int(x) >= 0]
        neighbour_candidates = [x for x in neighbour_candidates if x in set(nmer_idcs)]
        if neighbour_candidates:
            severed_atom_neighbour_id = int(np.flatnonzero(nmer_idcs == neighbour_candidates[0])[0])
        else:
            severed_atom_neighbour_id = np.argmin(distances[0])

        atom_type = str(atom_types[nmer_idcs][severed_atom_neighbour_id])
        atom_type_H_distance = distances_by_severed_order[i]
        if atom_type_H_distance is None:
            atom_type_H_distance, distance_source = _initial_capping_distance(atom_type)
        else:
            distance_source = "minimized_structure"
        applied_distances.append(float(atom_type_H_distance))
        distance_sources.append(distance_source)
        H_substituted_coords[:, i, :] = (
            nmer_coords[np.arange(nmer_coords.shape[0]), severed_atom_neighbour_id] +
            distance_vectors[np.arange(distance_vectors.shape[0]), severed_atom_neighbour_id] /
            distances[np.arange(distances.shape[0]), severed_atom_neighbour_id][:, None] * atom_type_H_distance
        )

    all_coords[:, severed_idcs] = H_substituted_coords
    atom_types[severed_idcs] = b'H'

    if return_capping_metadata:
        return all_coords, atom_types, {
            "capping_distances": applied_distances,
            "capping_distance_sources": distance_sources,
        }
    return all_coords, atom_types

def cap_nmer(
    h5_filepath: str,
    nmers_root: str,
    nmers_capped_root: str,
    fit_poly_root: str,
    logger: Logger,
    capping_distances: Optional[dict] = None,
):
    h5_capped_filepath = h5_filepath.replace(nmers_root, nmers_capped_root)
    h5_capped_folder = dirname(h5_capped_filepath)
    os.makedirs(h5_capped_folder, exist_ok=True)
    if os.path.exists(h5_capped_filepath):
        logger.warning(f"File {h5_capped_filepath} exists already. Overwriting...")

    # Load the H5 file saved in save_multimer
    coords, atom_types, fullnames, info_dict, _ = read_h5_file(h5_filepath)

    # Process the data
    capped_coords, capped_atom_types, capping_metadata = substitute_severed_atoms(
        coords,
        atom_types,
        info_dict,
        capping_distances=capping_distances,
        return_capping_metadata=True,
    )

    # --

    result = assign_symmetry_names_and_reorder(capped_coords, capped_atom_types, info_dict)

    nmer_poly_folder = h5_capped_folder.replace(nmers_capped_root, fit_poly_root)
    poly_generator_filename = apply_replacements_fp(os.path.join(nmer_poly_folder, "poly_generator.py"))
    save_poly_generator(result, poly_generator_filename, logger)

    # --

    # Save the capped nmers to a new H5 file

    capped_coords = result.pop('coords')
    capped_atom_types = result.pop('atom_types')
    sort_idcs = np.asarray(result["symmetry_names_argsort"], dtype=int)
    source_severed_indices = [int(x) for x in np.asarray(info_dict.get("severed_idcs", []), dtype=int)]
    source_bonded_idcs = np.asarray(info_dict.get("severed_bonded_idcs", []), dtype=int)
    capping_atom_indices = []
    capping_bonded_indices = []
    for i, source_index in enumerate(source_severed_indices):
        reordered_index = int(np.flatnonzero(sort_idcs == source_index)[0])
        capping_atom_indices.append(reordered_index)
        bonded = source_bonded_idcs[i] if i < len(source_bonded_idcs) else []
        bonded = [int(x) for x in np.asarray(bonded).reshape(-1) if int(x) >= 0]
        if bonded and np.any(sort_idcs == bonded[0]):
            capping_bonded_indices.append(int(np.flatnonzero(sort_idcs == bonded[0])[0]))
        else:
            capping_bonded_indices.append(-1)

    result.pop('symmetry_names_argsort')
    info_dict = {
        "capping_source_severed_indices": source_severed_indices,
        "capping_atom_indices": capping_atom_indices,
        "capping_bonded_indices": capping_bonded_indices,
        **capping_metadata,
    }

    write_h5_file(h5_capped_filepath, capped_coords, capped_atom_types, fullnames, info_dict, **result)

def save_poly_generator(data: dict, poly_generator_filename: str, logger: Logger):
    poly_generator_folder = dirname(poly_generator_filename)
    nmer_name = basename(poly_generator_folder)
    if os.path.isfile(poly_generator_filename):
        logger.info(f"--- Found polynomial generator for {nmer_name} ---")
        return
    # Create folder for code generating polynomials
    if not os.path.exists(poly_generator_filename):
        logger.info(f"--- Writing polynomial generator for {nmer_name} ---")
        os.makedirs(dirname(poly_generator_filename), exist_ok=True)
    
    # Read poly_generator python code template
    current_dir = Path(__file__).parent
    with open(current_dir.parent / 'templates' / 'poly_generator_template.py', 'r') as src:
        code = src.read()

    # Build symmetry names one line at a time
    symmetry_names = data.get('symmetry_names_sorted')
    per_atom_bonded_atoms = data.get('per_atom_bonded_atoms')
    add_atom_lines = []
    for symmetry_name, atom_bonded_atoms in zip(symmetry_names, per_atom_bonded_atoms):
        add_atom_lines.append([symmetry_name] + atom_bonded_atoms)
    
    add_atoms_code = """"""
    for add_atom_line in add_atom_lines:
        add_atoms_code += f'add_atom{[str(elem) for elem in add_atom_line]}\n'

    code=code.replace('[FOLDER]', nmer_name)
    code=code.replace('[NAME]', get_name_from_symmetry_names(symmetry_names))
    code=code.replace('[ADD_ATOMS]', add_atoms_code)
    code=code.replace('[MONOMER_INDICES]', '[[],]')
    code=code.replace('[NMER_INDICES]', get_nmer_indices(data.get('num_monomers')))
    with open(poly_generator_filename, 'w') as trg:
        trg.write(code)
    logger.info(f"--- Polynomial generator for {nmer_name} saved! ---")

def assign_symmetry_names_and_reorder(coords, atom_types, info_dict):
    atom_identities            = []
    atom_bonded_idcs_list      = []
    atom_bonded_index_keys     = []
    atom_identities_general    = []
    atom_name_to_symmetry_name = {}

    _atom_index_keys: List[str] = [key for key in info_dict.keys() if 'idcs' in key and 'bonded' not in key]
    for _id, _at in enumerate(atom_types):
        for _aikey in _atom_index_keys:
            _aikey_info = np.asarray(info_dict[_aikey]).reshape(-1,)
            if _id in _aikey_info:
                atom_bonded_index_key_fragments = _aikey.split('_')
                atom_bonded_index_key = '_'.join(atom_bonded_index_key_fragments[:-1]) + '_bonded_' + atom_bonded_index_key_fragments[-1]
                monomer_name_key = '_'.join(atom_bonded_index_key_fragments[:-1]) + '_name'
                atom_bonded_index_argwhere = np.argwhere(_aikey_info == _id).item()
                break
        atom_bonded_idcs = info_dict[atom_bonded_index_key][atom_bonded_index_argwhere]
        atom_bonded_idcs = atom_bonded_idcs[atom_bonded_idcs != -1]
        atom_bonded_idcs_list.append(atom_bonded_idcs)
        if isinstance(info_dict[monomer_name_key], List) or isinstance(info_dict[monomer_name_key], np.ndarray):
            monomer_name = info_dict[monomer_name_key][atom_bonded_index_argwhere]
        else:
            monomer_name = info_dict[monomer_name_key]
        assert isinstance(monomer_name, str)
        atom_identity = monomer_name + '|' + _at + '-' + ''.join(np.sort(atom_types[atom_bonded_idcs]))
        atom_identity_general = monomer_name + '|' + _at + '-' + str(len(atom_bonded_idcs))
        atom_identities.append(atom_identity)
        atom_identities_general.append(atom_identity_general)
        atom_bonded_index_keys.append(atom_bonded_index_key)
    
    atom_identities = np.array(atom_identities)
    atom_identities_general = np.array(atom_identities_general)
    sort_idcs = np.argsort(atom_identities_general)
    atom_identities_sorted = atom_identities[sort_idcs]
    atom_identities_general_sorted = atom_identities_general[sort_idcs]

    for atom_identity_sorted, atom_identity_general_sorted in zip(atom_identities_sorted, atom_identities_general_sorted):
        if atom_identity_sorted not in atom_name_to_symmetry_name:
            antosn = DataDict.ATOM_SYMMETRY_NAMES_DICT.get(atom_identity_sorted, None)
            if antosn is None:
                antosn = DataDict.ATOM_SYMMETRY_NAMES_DICT_GENERAL.get(atom_identity_general_sorted)
            if antosn is None:
                # Keep arbitrary trajectory chemistries usable. The legacy
                # table remains authoritative for known MB-Fit fragments;
                # unknown local environments receive a deterministic label
                # shared by equivalent atoms across n-mers.
                digest = hashlib.sha1(atom_identity.encode("utf-8")).hexdigest()[:10].upper()
                antosn = f"X{digest}"
            atom_name_to_symmetry_name[atom_identity_sorted] = antosn

    symmetry_names = np.array([
        atom_name_to_symmetry_name.get(an) + ('G' if 'severed' in abik else '')
        for an, abik in zip(atom_identities, atom_bonded_index_keys)
    ])
    symmetry_names_argsort = np.argsort(symmetry_names)
    per_atom_bonded_atoms = [argofyinx(symmetry_names_argsort, atom_bonded_idcs_list[sna]).tolist() for sna in symmetry_names_argsort]

    return {
        'num_monomers'          : len(_atom_index_keys) - 1,
        'symmetry_names_argsort': symmetry_names_argsort,
        'per_atom_bonded_atoms' : per_atom_bonded_atoms,
        'symmetry_names_sorted' : symmetry_names[symmetry_names_argsort],
        'coords'                : coords[:, symmetry_names_argsort],
        'atom_types'            : atom_types[symmetry_names_argsort],
    }

def get_name_from_symmetry_names(symmetry_names: np.ndarray):
    name = ""
    for k, v in Counter(symmetry_names).items():
        name += f"{k}{v}"
    return name

def get_nmer_indices(k: int):
    result = []
    for i in range(1, k+1):
        result.extend(combinations(range(k), i))
    return str([
        x for x in [list(comb) for comb in result]
        if all(x[j] == x[j - 1] + 1 for j in range(1, len(x)))
    ])

def build_xyz_capped_nmers(
    nmers_root: str,
    nmers_capped_root: str,
    fit_poly_root: str,
    logger: Logger,
    max_processes: int = 4,
    selected_nmer_names: Optional[List[str]] = None,
):
    logger.info("- Capping nmers...")
    h5_filepaths = list(glob.iglob(os.path.join(nmers_root, "**/*.h5"), recursive=True))
    if selected_nmer_names is not None:
        selected = set(selected_nmer_names)
        h5_filepaths = [path for path in h5_filepaths if basename(dirname(path)) in selected]
    if not h5_filepaths:
        raise FileNotFoundError(f"No selected n-mer HDF5 files found under {nmers_root}")

    if max_processes > 0:
        with multiprocessing.Pool(processes=max_processes) as pool:
            pool.starmap(
                cap_nmer,
                [
                    (
                        h5_filepath,
                        nmers_root,
                        nmers_capped_root,
                        fit_poly_root,
                        logger,
                    )
                    for h5_filepath in h5_filepaths
                ]
            )
        pool.join()
    else:
        for h5_filepath in h5_filepaths:
            cap_nmer(h5_filepath, nmers_root, nmers_capped_root, fit_poly_root, logger)
    logger.info("- Completed capping nmers!")


if __name__ == "__main__":
    try:
        main()
    except ValueError as error:
        raise SystemExit(f"Error: {error}") from None
