"""Measure capping-H distances from minimized structures and re-cap n-mers."""

import argparse
import glob
import json
import logging
import os
from typing import Optional

import numpy as np
from ase.io import read

from dataforge.scripts.build_dataset import save_optimized_structures
from dataforge.scripts.build_nmers import cap_nmer
from dataforge.src import DataDict, apply_replacements_fp
from dataforge.src.generic import read_h5_file
from dataforge.src.logging import get_logger


def _as_int_list(value):
    if value is None:
        return []
    if isinstance(value, bytes):
        value = value.decode()
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    return [int(x) for x in np.asarray(value).reshape(-1)]


def _optimized_path(h5_path: str, capped_root: str, optimized_root: str) -> str:
    relative = os.path.relpath(h5_path, capped_root)
    candidates = [
        os.path.join(optimized_root, os.path.dirname(relative), "nmers.opt"),
        os.path.join(
            optimized_root,
            apply_replacements_fp(os.path.dirname(relative)),
            "nmers.opt",
        ),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return candidates[0]


def _capping_atom_indices(atom_types, info_dict, extra_data):
    indices = _as_int_list(info_dict.get("capping_atom_indices"))
    if indices:
        return indices

    symmetry_names = extra_data.get("symmetry_names_sorted")
    if symmetry_names is not None:
        return [
            i for i, name in enumerate(np.asarray(symmetry_names).astype(str))
            if name.endswith("G")
        ]

    # Compatibility fallback for old capped H5 files that did not preserve
    # capping metadata. This is only safe when the molecule has no ordinary H.
    return [i for i, atom_type in enumerate(atom_types) if str(atom_type) == "H"]


def _bonded_atom_indices(atom_types, coords, cap_indices, info_dict, extra_data):
    bonded = _as_int_list(info_dict.get("capping_bonded_indices"))
    if len(bonded) == len(cap_indices):
        return bonded

    raw_bonded = extra_data.get("per_atom_bonded_atoms")
    if raw_bonded is not None:
        if isinstance(raw_bonded, bytes):
            raw_bonded = raw_bonded.decode()
        if isinstance(raw_bonded, str):
            raw_bonded = json.loads(raw_bonded)
        parsed = []
        for atom_index in cap_indices:
            values = np.asarray(raw_bonded[atom_index]).reshape(-1)
            values = [int(x) for x in values if str(x) not in {"--", "", "None"} and int(x) >= 0]
            parsed.append(values[0] if values else -1)
        if len(parsed) == len(cap_indices):
            return parsed

    heavy_indices = [i for i, atom_type in enumerate(atom_types) if str(atom_type) != "H"]
    return [
        min(heavy_indices, key=lambda j: np.linalg.norm(coords[j] - coords[i]))
        for i in cap_indices
    ]


def _legacy_cap_order(source_atom_types, source_info, cap_indices, cap_symmetry_names):
    """Align old capped files (without preserved metadata) to source cap order."""
    severed_indices = _as_int_list(source_info.get("severed_idcs"))
    severed_names = source_info.get("severed_name", [])
    if len(severed_names) != len(severed_indices):
        return cap_indices

    bonded = np.asarray(source_info.get("severed_bonded_idcs", []), dtype=int)
    if bonded.ndim == 1 and len(bonded):
        bonded = bonded.reshape(1, -1)
    available = list(cap_indices)
    ordered = []
    for i, name in enumerate(severed_names):
        name = str(name)
        bonded_indices = [int(x) for x in bonded[i] if int(x) >= 0] if i < len(bonded) else []
        atom_types = "".join(sorted(str(source_atom_types[x]) for x in bonded_indices))
        exact = DataDict.ATOM_SYMMETRY_NAMES_DICT.get(f"{name}|H-{atom_types}")
        general = DataDict.ATOM_SYMMETRY_NAMES_DICT_GENERAL.get(f"{name}|H-{len(bonded_indices)}")
        expected = (exact or general or "") + "G"
        matches = [x for x in available if str(cap_symmetry_names[x]) == expected]
        if not matches:
            return cap_indices
        chosen = matches[0]
        available.remove(chosen)
        ordered.append(chosen)
    return ordered


def extract_capping_distances(
    capped_root: str,
    optimized_root: str,
    output_filename: str,
    source_root: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
):
    logger = logger or logging.getLogger(__name__)
    files = {}
    missing = []

    for h5_path in sorted(glob.glob(os.path.join(capped_root, "**", "*.h5"), recursive=True)):
        relative = os.path.relpath(h5_path, capped_root)
        opt_path = _optimized_path(h5_path, capped_root, optimized_root)
        if not os.path.isfile(opt_path):
            missing.append(relative)
            continue

        _, atom_types, _, info_dict, extra_data = read_h5_file(h5_path, logger=logger)
        atoms = read(opt_path, index=0, format="extxyz")
        coords = np.asarray(atoms.positions, dtype=float)
        if len(coords) != len(atom_types):
            raise ValueError(
                f"Atom count mismatch for {relative}: capped H5 has {len(atom_types)}, "
                f"optimized structure has {len(coords)}."
            )

        cap_indices = _capping_atom_indices(atom_types, info_dict, extra_data)
        bonded_indices = _bonded_atom_indices(atom_types, coords, cap_indices, info_dict, extra_data)
        distances_by_cap_index = [
            float(np.linalg.norm(coords[cap_index] - coords[bonded_index]))
            for cap_index, bonded_index in zip(cap_indices, bonded_indices)
            if bonded_index >= 0
        ]
        if len(distances_by_cap_index) != len(cap_indices):
            raise ValueError(f"Could not identify a bonded heavy atom for every cap in {relative}.")

        source_severed_indices = _as_int_list(info_dict.get("capping_source_severed_indices"))
        if not source_severed_indices and source_root:
            source_path = os.path.join(source_root, relative)
            if os.path.isfile(source_path):
                _, source_atom_types, _, source_info, _ = read_h5_file(source_path, logger=logger)
                source_severed_indices = _as_int_list(source_info.get("severed_idcs"))
                if not info_dict.get("capping_source_severed_indices"):
                    symmetry_names = extra_data.get("symmetry_names_sorted")
                    if symmetry_names is not None:
                        cap_indices = _legacy_cap_order(
                            source_atom_types,
                            source_info,
                            cap_indices,
                            np.asarray(symmetry_names).astype(str),
                        )
                        bonded_indices = _bonded_atom_indices(atom_types, coords, cap_indices, info_dict, extra_data)
                        distances_by_cap_index = [
                            float(np.linalg.norm(coords[cap_index] - coords[bonded_index]))
                            for cap_index, bonded_index in zip(cap_indices, bonded_indices)
                        ]
        distances = distances_by_cap_index
        if len(source_severed_indices) != len(distances):
            source_severed_indices = []

        files[relative] = {
            "capping_atom_indices": cap_indices,
            "capping_bonded_indices": bonded_indices,
            "source_severed_indices": source_severed_indices,
            "distances": distances,
        }

    if missing:
        logger.warning("Missing minimized structures for %d capped H5 files.", len(missing))
    output = {
        "version": 1,
        "capped_root": os.path.abspath(capped_root),
        "optimized_root": os.path.abspath(optimized_root),
        "files": files,
        "missing_optimized": missing,
    }
    os.makedirs(os.path.dirname(os.path.abspath(output_filename)), exist_ok=True)
    with open(output_filename, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Saved capping distances for %d n-mer files to %s", len(files), output_filename)


def apply_capping_distances(
    source_root: str,
    destination_root: str,
    fit_poly_root: str,
    distances_filename: str,
    max_processes: int = 0,
    logger: Optional[logging.Logger] = None,
):
    logger = logger or logging.getLogger(__name__)
    with open(distances_filename) as f:
        distance_data = json.load(f)

    h5_paths = sorted(glob.glob(os.path.join(source_root, "**", "*.h5"), recursive=True))
    for h5_path in h5_paths:
        relative = os.path.relpath(h5_path, source_root)
        entry = distance_data.get("files", {}).get(relative)
        if entry is None:
            raise KeyError(f"No capping distances found for {relative}.")
        if not entry.get("source_severed_indices"):
            source_info = read_h5_file(h5_path, logger=logger)[3]
            entry = dict(entry)
            entry["source_severed_indices"] = _as_int_list(source_info.get("severed_idcs"))
        cap_nmer(
            h5_path,
            source_root,
            destination_root,
            fit_poly_root,
            logger,
            capping_distances=entry,
        )
    logger.info("Re-capped %d n-mer files into %s", len(h5_paths), destination_root)


def main(args=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    extract_parser = subparsers.add_parser("extract")
    extract_parser.add_argument("--capped-root", required=True)
    extract_parser.add_argument("--optimized-root", required=True)
    extract_parser.add_argument("--min-output-root", default=None)
    extract_parser.add_argument("--fit-poly-root", default=None)
    extract_parser.add_argument("--source-root", default=None)
    extract_parser.add_argument("--output", required=True)

    apply_parser = subparsers.add_parser("apply")
    apply_parser.add_argument("--source-root", required=True)
    apply_parser.add_argument("--destination-root", required=True)
    apply_parser.add_argument("--fit-poly-root", required=True)
    apply_parser.add_argument("--distances", required=True)
    apply_parser.add_argument("--max-processes", type=int, default=0)

    parsed = parser.parse_args(args)
    logger = get_logger("recap_nmers.log")
    if parsed.mode == "extract":
        if parsed.min_output_root:
            save_optimized_structures(
                parsed.min_output_root,
                parsed.optimized_root,
                parsed.fit_poly_root or "",
            )
        extract_capping_distances(
            parsed.capped_root,
            parsed.optimized_root,
            parsed.output,
            source_root=parsed.source_root,
            logger=logger,
        )
    else:
        apply_capping_distances(
            parsed.source_root,
            parsed.destination_root,
            parsed.fit_poly_root,
            parsed.distances,
            max_processes=parsed.max_processes,
            logger=logger,
        )


if __name__ == "__main__":
    main()
