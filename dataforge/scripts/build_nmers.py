import argparse
import logging
import os
import glob
import json
from pathlib import Path
import numpy as np
import multiprocessing
from logging import Logger
from multiprocessing import Lock
from collections import Counter

from os.path import join, dirname, basename
from typing import Dict, List, Optional, Union
from itertools import combinations, zip_longest

from dataforge.src import DataDict, intersect_rows_2d, dynamic_for_loop, fix_bonds, apply_replacements_fp
from dataforge.src.qchem_utils import prepare_qchem_input
from dataforge.src.logging import get_logger
from dataforge.src.nmers import Monomer, Multimer
from dataforge.src.generic import argofyinx, read_h5_file, write_h5_file


# Create a global lock
lock = Lock()


def main(args=None):
    args = parse_command_line(args)

    build_nmers(
        input_filename          = args.input,
        dataset_root            = args.root,
        nmer_sampling_conf      = args.order,
        keep_only_monomer_names = args.keep,
        max_processes           = args.max_processes,
    )

def parse_command_line(args=None):
    parser = argparse.ArgumentParser(
        description="""
        Read one or more trajectory files, filter and group the molecule of interest
        and savs the information as a npz dataset file.
    """
    )
    parser.add_argument(
        "-i",
        "--input",
        help="The `.npz` file saved in the previous step.",
        required=True,
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
        required=True,
    )
    parser.add_argument(
        "-k",
        "--keep",
        nargs='+',
        help="Optional list with the monomer names to keep. All other monomers will be ignored.",
        default=None,
    )
    parser.add_argument(
        "-m",
        "--max-processes",
        help="Maximum number of processes to use for building the nmers.",
        type=int,
        default=0,
    )
    
    return parser.parse_args(args=args)

def build_nmers(
    input_filename: str,
    dataset_root: str,
    nmer_sampling_conf: Union[List[int], Dict[int, int]],
    keep_only_monomer_names: Optional[List[str]] = None,
    max_processes: int = 4,
    **kwargs
):
    logger = get_logger('02_build_nmers.log', level=logging.DEBUG)
    
    DATA_ROOT         =  join(dataset_root, "data"           )
    NMERS_ROOT        =  join(DATA_ROOT   , "xyz"            )
    NMERS_CAPPED_ROOT =  join(DATA_ROOT   , "xyz_capped"     )
    QCHEM_IN_ROOT     =  join(DATA_ROOT   , "qchem_input"    )
    QCHEM_MIN_IN_ROOT =  join(DATA_ROOT   , "qchem_min_input")

    FIT_ROOT           =                                          join(dataset_root, "fitting")
    FIT_POLY_ROOT      = kwargs.get('FIT_POLY_ROOT',     None) or join(FIT_ROOT    , "poly"   )

    if isinstance(nmer_sampling_conf, list):
        nmer_sampling_conf = {int(k): None for k in nmer_sampling_conf}

    automatic_sampling = not all(v is None for v in nmer_sampling_conf.values())

    # build_xyz_nmers(
    #     traj_dataset_filename   = input_filename,
    #     data_root               = DATA_ROOT,
    #     nmers_root              = NMERS_ROOT,
    #     monomers_dict           = DataDict.MONOMERS_DICT,
    #     nmer_sampling_conf      = nmer_sampling_conf,
    #     logger                  = logger,
    #     keep_only_monomer_names = keep_only_monomer_names,
    #     compute_descriptors     = automatic_sampling,
    #     max_processes           = max_processes,
    # )

    build_xyz_capped_nmers(
        nmers_root              = NMERS_ROOT,
        nmers_capped_root       = NMERS_CAPPED_ROOT,
        fit_poly_root           = FIT_POLY_ROOT,
        logger                  = logger,
        max_processes           = max_processes,
    )

    if not automatic_sampling:
        return
    
    prepare_qchem_input(
        nmers_capped_root       = NMERS_CAPPED_ROOT,
        qchem_in_root           = QCHEM_IN_ROOT,
        qchem_min_in_root       = QCHEM_MIN_IN_ROOT,
        charges_dict            = DataDict.CHARGES_DICT,
        max_processes           = max_processes,
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
    compute_descriptors: bool = True,
    max_processes: int = 4,
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
    monomers = build_monomers(
        dataset=traj_dataset,
        monomers_dict=monomers_dict,
        logger=logger,
        keep_only_monomer_names=keep_only_monomer_names,
        compute_descriptors=compute_descriptors,
    )

    # --- Create mapping between monomer indices and monomer names --- #
    # ---------------------------------------------------------------- #
    
    build_topology(monomers, data_root)

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
    )
    logger.info("- Completed building nmers!")

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

def build_topology(monomers: List[Monomer], data_root: str):
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
        for nmer_degree in range(2, max(DataDict.FOLDER_NAMES.keys())+1):
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

def save_multimer(
    nmers_root: str,
    folder_name: str,
    multimer: Multimer,
    multimer_sampled_indices: Optional[np.ndarray],
    orig_pos: np.ndarray,
    orig_all_atom_types: np.ndarray,
    logger: Logger
):

    # - Create folder - #
    nmer_folder = os.path.join(nmers_root, folder_name, multimer.name)
    if not os.path.isdir(nmer_folder): os.makedirs(nmer_folder, exist_ok=True)
    h5_filename = os.path.join(nmer_folder, multimer.h5_filename)
    if os.path.isfile(h5_filename): logger.warning(f"File {h5_filename} exists alreeady. Overwriting...")
    if multimer_sampled_indices is None: multimer_sampled_indices = np.arange(len(orig_pos))

    # Prepare data for saving
    all_coords = []
    all_atom_types = []
    all_info_dicts = []

    for xyz, frame_id in zip(orig_pos[multimer_sampled_indices][:, multimer.orig_all_atoms_idcs], multimer_sampled_indices):
        # Index of all monomer atoms, relative to multimer atoms only
        severed_idcs = np.argwhere(np.isin(multimer.orig_all_atoms_idcs,multimer.orig_connected_atoms_idcs)).flatten()
        info_dict = {"fullname": f"{str(frame_id)}_{multimer.fullname}"}
        severed_names = np.zeros((len(severed_idcs),), dtype=object)
        severed_bonded_idcs = get_bonded_idcs(severed_idcs, multimer, multimer)

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

        all_coords.append(xyz)
        all_atom_types.append(orig_all_atom_types[multimer.orig_all_atoms_idcs])
        all_info_dicts.append(info_dict)

    all_coords = np.array(all_coords)
    all_atom_types = np.array(all_atom_types, dtype=np.string_)

    # Save to h5 file
    with lock:
        write_h5_file(h5_filename, all_coords, all_atom_types, all_info_dicts)

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
    monomers,
    n,
    folder_name,
    orig_pos: np.ndarray,
    orig_all_atom_types: np.ndarray,
    logger: Logger,
    recursive_multimer_sampled_indices = None,
    compute_descriptors: bool = True,
    max_processes: int = 4
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
        if compute_descriptors:
            multimer.compute_descriptors()
        multimers.append(multimer)
        
        # - Count occurrences of each multimer - #
        counts = multimers_occurrence.get(multimer.name, 0) + 1
        multimers_occurrence[multimer.name] = counts
        multimers_first_occurrence[multimer.name] = True

    if max_processes == 0:
        for multimer in multimers:
            process_multimer(
                multimer,
                nmers_root,
                folder_name,
                n_samples,
                method,
                multimers_occurrence,
                multimers_first_occurrence,
                orig_pos,
                orig_all_atom_types,
                logger,
                nmer_sampling_conf,
                n,
                recursive_multimer_sampled_indices,
                compute_descriptors,
                max_processes,
            )
    else:
        with multiprocessing.Pool(processes=max_processes) as pool:
            pool.starmap(
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
                        orig_pos,
                        orig_all_atom_types,
                        logger,
                        nmer_sampling_conf,
                        n,
                        recursive_multimer_sampled_indices,
                        compute_descriptors,
                        max_processes,
                    )
                    for multimer in multimers
                ]
            )
    return

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

def process_multimer(
    multimer: Multimer,
    nmers_root,
    folder_name,
    n_samples,
    method,
    multimers_occurrence: dict,
    multimers_first_occurrence: dict,
    orig_pos,
    orig_all_atom_types,
    logger: Logger,
    nmer_sampling_conf,
    n,
    recursive_multimer_sampled_indices,
    compute_descriptors,
    max_processes,
):
    logger.info(f"--- Saving {multimer.h5_filename} to {folder_name}...")

    if n_samples is None:
        multimer_sampled_indices = None
    elif recursive_multimer_sampled_indices is None:
        multimer_n_samples = n_samples // multimers_occurrence.get(multimer.name)
        if multimers_first_occurrence.get(multimer.name):
            multimer_n_samples += n_samples % multimers_occurrence.get(multimer.name)
            multimers_first_occurrence[multimer.name] = False
        multimer_sampled_indices = multimer.sample(multimer_n_samples, method=method)
    else:
        multimer_sampled_indices = recursive_multimer_sampled_indices

    save_multimer(nmers_root, folder_name, multimer, multimer_sampled_indices, orig_pos, orig_all_atom_types, logger)

    build_multimer_recursively(
        nmers_root=nmers_root,
        nmer_sampling_conf=nmer_sampling_conf,
        monomers=multimer._monomers,
        n=n-1,
        folder_name=os.path.join(folder_name, DataDict.FOLDER_NAMES.get(n-1, "")),
        orig_pos=orig_pos,
        orig_all_atom_types=orig_all_atom_types,
        logger=logger,
        recursive_multimer_sampled_indices=multimer_sampled_indices,
        compute_descriptors=compute_descriptors,
        max_processes=0, # Daemonic processes are not allowed to have children
    )

def build_multimers(
    nmers_root: str,
    nmer_sampling_conf: dict,
    monomers: List[Monomer],
    orig_pos: np.ndarray,
    orig_all_atom_types: np.ndarray,
    compute_descriptors: bool,
    logger: Logger,
    max_processes: int = 4,
):
    for n, folder_name in DataDict.FOLDER_NAMES.items():
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
            recursive_multimer_sampled_indices=None,
            compute_descriptors=compute_descriptors,
            max_processes=max_processes,
        )

def substitute_severed_atoms(all_coords, all_atom_types, info_dict: dict):
    severed_idcs = info_dict.get("severed_idcs")
    nmer_idcs = np.delete(np.arange(all_coords.shape[1]), severed_idcs)
    nmer_coords = all_coords[:, nmer_idcs]
    severed_coords = all_coords[:, severed_idcs]

    H_substituted_coords = np.zeros_like(severed_coords)
    for i, severed_atom_coords in enumerate(severed_coords.transpose(1, 0, 2)):
        distance_vectors = severed_atom_coords[:, None, :] - nmer_coords
        distances = np.linalg.norm(distance_vectors, axis=2)
        severed_atom_neighbour_id = np.argmin(distances[0])
        atom_type_H_distance = DataDict.ATOM_TYPE_TO_H_DISTANCE[all_atom_types[0, nmer_idcs][severed_atom_neighbour_id]]
        H_substituted_coords[:, i, :] = (
            nmer_coords[np.arange(nmer_coords.shape[0]), severed_atom_neighbour_id] +
            distance_vectors[np.arange(distance_vectors.shape[0]), severed_atom_neighbour_id] /
            distances[np.arange(distances.shape[0]), severed_atom_neighbour_id][:, None] * atom_type_H_distance
        )

    all_coords[:, severed_idcs] = H_substituted_coords
    all_atom_types[:, severed_idcs] = b'H'

    return all_coords, all_atom_types

def cap_nmer(h5_filepath: str, nmers_root: str, nmers_capped_root: str, fit_poly_root: str, logger: Logger):
    h5_capped_filepath = h5_filepath.replace(nmers_root, nmers_capped_root)
    h5_capped_folder = dirname(h5_capped_filepath)
    os.makedirs(h5_capped_folder, exist_ok=True)
    if os.path.exists(h5_capped_filepath):
        logger.warning(f"File {h5_capped_filepath} exists already. Overwriting...")

    # Load the H5 file saved in save_multimer
    all_coords, all_atom_types, all_info_dicts, _ = read_h5_file(h5_filepath)
    info_dict = all_info_dicts[0]

    # Process the data
    capped_coords, capped_atom_types = substitute_severed_atoms(all_coords, all_atom_types, info_dict)

    # --

    result = assign_symmetry_names_and_reorder(capped_coords, capped_atom_types, info_dict)

    nmer_poly_folder = h5_capped_folder.replace(nmers_capped_root, fit_poly_root)
    poly_generator_filename = apply_replacements_fp(os.path.join(nmer_poly_folder, "poly_generator.py"))
    save_poly_generator(result, poly_generator_filename, logger)

    # --

    # Save the capped nmers to a new H5 file

    capped_coords = result.pop('all_coords')
    capped_atom_types = result.pop('all_atom_types')
    result.pop('symmetry_names_argsort')
    all_info_dicts = [{k: v for k, v in info_dict.items() if k == 'fullname'} for info_dict in all_info_dicts]

    with lock:
        write_h5_file(h5_capped_filepath, capped_coords, capped_atom_types, all_info_dicts, **result)

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

def assign_symmetry_names_and_reorder(all_coords, all_atom_types, info_dict):
    atom_identities            = []
    atom_bonded_idcs_list      = []
    atom_bonded_index_keys     = []
    atom_identities_general    = []
    atom_name_to_symmetry_name = {}

    _atom_index_keys: List[str] = [key for key in info_dict.keys() if 'idcs' in key and 'bonded' not in key]
    for _id, _at in enumerate(all_atom_types[0]):
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
        atom_identity = monomer_name + '|' + _at + '-' + ''.join(np.sort(all_atom_types[0][atom_bonded_idcs]))
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
        'all_coords'            : all_coords    [:, symmetry_names_argsort],
        'all_atom_types'        : all_atom_types[:, symmetry_names_argsort],
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

def build_xyz_capped_nmers(nmers_root: str, nmers_capped_root: str, fit_poly_root: str, logger: Logger, max_processes: int = 4):
    logger.info("- Capping nmers...")
    h5_filepaths = list(glob.iglob(os.path.join(nmers_root, "**/*.h5"), recursive=True))

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
    main()