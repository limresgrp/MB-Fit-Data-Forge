import argparse
from logging import Logger
import os
import glob
import json
import numpy as np

from os.path import join, dirname
from typing import Dict, List, Optional, Union
from itertools import combinations, zip_longest

from dataforge.src import DataDict, intersect_rows_2d, dynamic_for_loop
from dataforge.src.qchem_utils import prepare_qchem_input
from dataforge.src.logging import get_logger
from dataforge.src.nmers import Monomer, Multimer

from ase import Atoms
from ase.io import read, write


def main(args=None):
    args = parse_command_line(args)

    build_nmers(
        input_filename          = args.input,
        dataset_root            = args.root,
        nmer_n_samples          = args.samples,
        keep_only_monomer_names = args.keep,
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
        "-s",
        "--samples",
        nargs='+',
        help="Ordered list of integers, indicating respectively the number of monomers, dimers and trimers to sample",
        required=True,
    )
    parser.add_argument(
        "-k",
        "--keep",
        nargs='+',
        help="Optional list with the monomer names to keep. All other monomers will be ignored.",
        default=None,
    )
    
    return parser.parse_args(args=args)

def build_nmers(
    input_filename: str,
    dataset_root: str,
    nmer_n_samples: Union[List[int], Dict[int, int]],
    keep_only_monomer_names: Optional[List[str]] = None,
):
    logger = get_logger('02_build_nmers.log')
    
    DATA_ROOT =          join(dataset_root, "data"           )
    NMERS_ROOT =         join(DATA_ROOT,    "xyz"            )
    NMERS_CAPPED_ROOT =  join(DATA_ROOT,    "xyz_capped"     )
    QCHEM_IN_ROOT =      join(DATA_ROOT,    "qchem_input"    )
    QCHEM_MIN_IN_ROOT =  join(DATA_ROOT,    "qchem_min_input")

    if isinstance(nmer_n_samples, list):
        nmer_n_samples = {k: int(v) for k, v in zip(range(1, len(nmer_n_samples) + 1), nmer_n_samples)}
    
    logger.info("- Building nmers...")
    build_xyz_nmers(
        traj_dataset_filename=input_filename,
        data_root               = DATA_ROOT,
        nmers_root              = NMERS_ROOT,
        monomers_dict           = DataDict.MONOMERS_DICT,
        nmer_n_samples          = nmer_n_samples,
        logger                  = logger,
        keep_only_monomer_names = keep_only_monomer_names,
    )

    logger.info("- Capping nmers...")
    build_xyz_capped_nmers(
        NMERS_ROOT,
        NMERS_CAPPED_ROOT,
    )

    logger.info("- Preparing QChem input files...")
    prepare_qchem_input(
        NMERS_CAPPED_ROOT,
        QCHEM_IN_ROOT,
        QCHEM_MIN_IN_ROOT,
        DataDict.CHARGES_DICT,
    )
    logger.info("- Complete!")

def build_xyz_nmers(
    traj_dataset_filename: str,
    data_root: str,
    nmers_root: str,
    monomers_dict: dict,
    nmer_n_samples: dict,
    logger: Logger,
    keep_only_monomer_names: Optional[List[str]] = None,
):
    # ------------------ L O A D    D A T A S E T -------------------- #
    # ---------------------------------------------------------------- #

    if not os.path.exists(data_root):
        os.makedirs(data_root)

    traj_dataset = dict(np.load(traj_dataset_filename, allow_pickle=True))

    # ---------------------------------------------------------------- #
    # ---------------------------------------------------------------- #

    logger.info("-- Building monomers...")
    monomers = build_monomers(
        dataset=traj_dataset,
        monomers_dict=monomers_dict,
        logger=logger,
        keep_only_monomer_names=keep_only_monomer_names,
    )

    # --- Create mapping between monomer indices and monomer names --- #
    # ---------------------------------------------------------------- #
    
    topology = {
        "monomers": {m.id: m.name for m in monomers}
    }

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

    # ---------------------------------------------------------------- #
    # ---------------------------------------------------------------- #

    logger.info("-- Building multimers...")
    build_multimers(
        nmers_root,
        nmer_n_samples,
        monomers,
        traj_dataset["position"],
        traj_dataset["atom_type"],
        logger=logger,
    )

def build_monomers(
        dataset: dict,
        monomers_dict: dict,
        logger: Logger,
        keep_only_monomer_names: Optional[List[str]] = None
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

def save_multimer(
    nmers_root: str,
    folder_name: str,
    multimer: Multimer,
    multimer_sampled_indices: np.ndarray,
    orig_pos: np.ndarray,
    orig_all_atom_types: np.ndarray,
):
    # - Create folder - #
    nmer_folder = os.path.join(nmers_root, folder_name, multimer.name)
    if not os.path.isdir(nmer_folder):
        os.makedirs(nmer_folder)

    # - Save xyz files - #
    for xyz, frame_id in zip(
        orig_pos[multimer_sampled_indices][:, multimer.orig_all_atoms_idcs],
        multimer_sampled_indices
    ):
        atoms = Atoms(orig_all_atom_types[multimer.orig_all_atoms_idcs], xyz)
        severed_idcs = np.argwhere(   # Index of all monomer atoms, relative to multimer atoms only
            np.isin(
                multimer.orig_all_atoms_idcs,
                multimer.orig_connected_atoms_idcs
            )
        ).flatten()
        
        atoms.info["severed_name"] = np.zeros((len(severed_idcs),), dtype=object)
        atoms.info["severed_idcs"] = severed_idcs
        atoms.info["severed_bonded_idcs"] = get_bonded_idcs(severed_idcs, multimer, multimer)

        for m_id, monomer in enumerate(multimer.monomers):
            monomer_idcs = np.argwhere(   # Index of all monomer atoms, relative to multimer atoms only
                np.isin(
                    multimer.orig_all_atoms_idcs,
                    monomer.orig_atoms_idcs
                )
            ).flatten()

            for severed_atom_idx, mocai in enumerate(multimer.orig_connected_atoms_idcs):
                if mocai in monomer.orig_connected_atoms_idcs:
                    atoms.info["severed_name"][severed_atom_idx] = monomer.name
            atoms.info[f"monomer_{m_id + 1}_name"] = monomer.name
            atoms.info[f"monomer_{m_id + 1}_idcs"] = monomer_idcs
            atoms.info[f"monomer_{m_id + 1}_bonded_idcs"] = get_bonded_idcs(monomer_idcs, multimer, monomer)
        
        fname = os.path.join(nmer_folder, f"f{str(frame_id)}-" + "_".join(multimer.monomers_idcs) + ".xyz")
        write(
            filename=fname,
            images=atoms,
            format="extxyz",
            append=False,
        )

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
    nmer_n_samples: dict,
    monomers,
    n,
    folder_name,
    orig_pos: np.ndarray,
    orig_all_atom_types: np.ndarray,
    logger: Logger,
    recursive_multimer_sampled_indices = None
):
    if n not in nmer_n_samples:
        return
    
    multimers: List[Multimer] = []
    multimers_occurrence = {}
    multimers_first_occurrence = {}
    for monomers_tuple in combinations(monomers, n):
        if not monomers_are_valid(monomers_tuple):
            continue

        # - Create Multimer to join descriptors distributions of Monomers - #
        multimer = Multimer(monomers_tuple, orig_all_atom_types)
        multimer.compute_descriptors()
        multimers.append(multimer)
        
        # - Count occurrences of each multimer - #
        counts = multimers_occurrence.get(multimer.name, 0) + 1
        multimers_occurrence[multimer.name] = counts
        multimers_first_occurrence[multimer.name] = True

    for multimer in multimers:

        # - Sample uniformly over multimer descriptors values - #
        if recursive_multimer_sampled_indices is None:
            n_samples = nmer_n_samples[n] // multimers_occurrence.get(multimer.name)
            
            # - Adjust number of samples in first occurrence of multimer to sample the exact total number of samples that was specified - #
            if multimers_first_occurrence.get(multimer.name):
                n_samples += nmer_n_samples[n] % multimers_occurrence.get(multimer.name)
                multimers_first_occurrence[multimer.name] = False
            multimer_sampled_indices = multimer.sample_uniform(n_samples)
        else:
            multimer_sampled_indices = recursive_multimer_sampled_indices

        # - Save data - #
        save_multimer(nmers_root, folder_name, multimer, multimer_sampled_indices, orig_pos, orig_all_atom_types)

        # - Recursively save lower order nmers - #
        build_multimer_recursively(
            nmers_root=nmers_root,
            nmer_n_samples=nmer_n_samples,
            monomers=multimer.monomers,
            n=n-1,
            folder_name=os.path.join(folder_name, DataDict.FOLDER_NAMES.get(n-1, "")),
            orig_pos=orig_pos,
            orig_all_atom_types=orig_all_atom_types,
            logger=logger,
            recursive_multimer_sampled_indices=multimer_sampled_indices
        )
    
    return

def build_multimers(
    nmers_root: str,
    nmer_n_samples: dict,
    monomers: List[Monomer],
    orig_pos: np.ndarray,
    orig_all_atom_types: np.ndarray,
    logger: Logger,
):
    for n, folder_name in DataDict.FOLDER_NAMES.items():
        build_multimer_recursively(
            nmers_root=nmers_root,
            nmer_n_samples=nmer_n_samples,
            monomers=monomers,
            n=n,
            folder_name=folder_name,
            orig_pos=orig_pos,
            orig_all_atom_types=orig_all_atom_types,
            logger=logger,
            recursive_multimer_sampled_indices=None,
        )

def substitute_severed_atoms(nmer):
    coords =       nmer.arrays['positions']
    atom_types =   nmer.arrays['numbers']
    severed_idcs = nmer.info["severed_idcs"]
    nmer_idcs = np.delete(np.arange(len(coords)), severed_idcs)
    nmer_coords = coords[nmer_idcs]
    severed_coords = coords[severed_idcs].reshape(-1, 3)

    H_substituted_coords = []
    for severed_atom_coords in severed_coords:
        distance_vectors = severed_atom_coords[None, ...] - nmer_coords
        distances = np.linalg.norm(distance_vectors, axis=1)
        severed_atom_neighbour_id = np.argmin(distances)
        atom_type_H_distance = DataDict.ATOM_TYPE_TO_H_DISTANCE[atom_types[nmer_idcs][severed_atom_neighbour_id]]
        H_substituted_coords.append(
            nmer_coords[severed_atom_neighbour_id] +
            distance_vectors[severed_atom_neighbour_id] / distances[severed_atom_neighbour_id] * atom_type_H_distance
        )
    H_substituted_coords = np.stack(H_substituted_coords, axis=0)
    nmer.arrays['positions'][severed_idcs] = H_substituted_coords
    nmer.arrays['numbers'][severed_idcs] = 1
    return nmer

def build_xyz_capped_nmers(nmers_root: str, nmers_capped_root: str):
    for file_path in glob.iglob(os.path.join(nmers_root, "**/*.xyz"), recursive=True):
        out_file_path = file_path.replace(nmers_root, nmers_capped_root)
        if not os.path.exists(dirname(out_file_path)):
            os.makedirs(dirname(out_file_path))
        if os.path.exists(out_file_path):
            continue
        
        assert os.path.isfile(file_path)
        nmer = read(file_path, index=":", format="extxyz")[0]
        new_nmer = substitute_severed_atoms(nmer)
        write(
            out_file_path,
            new_nmer,
            format="extxyz",
            append=False,
        )


if __name__ == "__main__":
    main()