import argparse
import os
import numpy as np
import MDAnalysis as mda

from os.path import dirname
from typing import Dict, List, Optional
from pathlib import Path
from dataforge.src import parse_slice
from dataforge.src.logging import get_logger


def main(args=None):
    args = parse_command_line(args)

    parse_trajectory(
        input_filename  = args.input,
        traj_filenames  = args.traj,
        selection       = args.selection,
        trajslice       = args.trajslice,
        output_filename = args.output,
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
        help="Filename of the reference structure+topology. It should be a `.tpr` file.",
        required=True,
    )
    parser.add_argument(
        "-t",
        "--traj",
        nargs='+',
        help="List of filenames, containing all the trajectory files to load. They could be either `.trr` or `.xtc` files.",
        required=True,
    )
    parser.add_argument(
        "-s",
        "--selection",
        help="Selection string to filter the atoms of the molecule of interest. Defaults to 'all'",
        default='all',
    )
    parser.add_argument(
        "-ts",
        "--trajslice",
        help="""Optional variable to filter the frames of the trajectory to keep.
        It could be either `None` (by default, keep all frames) or a string in the form `[start]:[stop]:[step]`.
        E.g. `:1000:2` filters the first 1000 frames, striding with a step of 2 and yielding a total of 500 frames.""",
        default=None,
    )
    parser.add_argument(
        "-o",
        "--output",
        help="""It is the filename of the output `.npz` dataset.
        It could be either a `.npz` file or a folder.
        If a folder is specified, the dataset will be saved in that folder,
        using the `INPUT_FILENAME` stem as a filename and the `.npz` suffix.
        If not specified, the dataset will be names as the `INPUT_FILENAME` but with the `.npz` suffix.""",
        default=None,
    )
    
    return parser.parse_args(args=args)

def parse_trajectory(
    input_filename: str,
    traj_filenames: List[str],
    selection: str = 'all',
    trajslice: Optional[str] = None,
    output_filename: Optional[str] = None,
    multiple_mols_per_frame: bool = True,
) -> str:
    logger = get_logger('01_parse_traj.log')

    # --- Read trajectory --- #
    logger.info("- Reading trajectory...")
    universe = mda.Universe(input_filename, *traj_filenames)

    # --- Parse trajectory --- #
    logger.info("- Parsing trajectory...")
    dataset = get_dataset(universe, selection, trajslice)
    logger.info(f"- {len(dataset['position'])} frames extracted from trajectory")
    if multiple_mols_per_frame:
        dataset = extract_single_molecule(universe, dataset)
    logger.info(f"- {len(dataset['position'])} total frames featuring a single occurrence of molecule")

    u_monomers, count = np.unique(dataset['monomer_names'], return_counts=True)
    unique_monomers_formatted = ' | '.join([f"{c}x {m}" for m, c in zip(u_monomers, count)])
    logger.info(f"- The system contains the following monomers: {unique_monomers_formatted}")
    
    # --- Save parsed trajectory as a .npz dataset --- #
    if output_filename is None:
        ip = Path(input_filename)
        output_filename = str(Path(ip.parent, ip.stem + '.npz'))
    else:
        op = Path(output_filename)
        if op.suffix != '.npz':
            ip = Path(input_filename)
            output_filename = str(Path(op.parent, op.stem, ip.stem + '.npz'))

    os.makedirs(dirname(output_filename), exist_ok=True)
    np.savez(output_filename, **dataset)
    logger.info(f"- Parsed trajectory saved! Dataset filename: {output_filename}")
    return output_filename

def get_dataset(universe: mda.Universe, selection: str, trajslice: Optional[str] = None) -> Dict[str, np.ndarray]:
    # Select atoms #
    system = universe.select_atoms(selection)
    
    # Read attributes #
    atom_name = []
    atom_type = []
    atom_resname = []
    atom_resnum = []
    atom_orig_index = []
    for atom in system.atoms:
        atom_name.append(atom.name)
        try:
            atom_type.append(atom.element)
        except:
            atom_type.append(atom.type)
        atom_resname.append(atom.resname)
        atom_resnum.append(atom.resnum)
        atom_orig_index.append(atom.index)
    
    # Read traj positions #
    if trajslice is not None:
        trajslice = parse_slice(trajslice)
    traj = universe.trajectory if trajslice is None else universe.trajectory[trajslice]
    position = []
    for ts in traj:
        position.append(system.positions)
    
    # Build dataset
    return {
        'position':        np.array(position),
        'atom_name':       np.array(atom_name),
        'atom_type':       np.array(atom_type),
        'atom_resname':    np.array(atom_resname),
        'atom_resnum':     np.array(atom_resnum, dtype=np.int32),
        'atom_orig_index': np.array(atom_orig_index, dtype=np.int32),
    }

def extract_single_molecule(universe: mda.Universe, dataset: dict) -> Dict[str, np.ndarray]:
    # Extract single molecule from whole system
    _, counts =        np.unique(dataset['atom_resnum'], return_counts=True)
    split_idcs =       np.cumsum(counts)
    position_list =    np.split(dataset['position'], split_idcs[:-1], axis=1)
    position_stacked = np.concatenate(position_list, axis=0)

    dataset['position'] = position_stacked
    n_atoms_single = position_stacked.shape[1]
    for k in ['atom_name', 'atom_type', 'atom_resname', 'atom_orig_index']:
        dataset[k] = dataset[k][:n_atoms_single]
    del dataset['atom_resnum']

    atom_orig_index = dataset['atom_orig_index']
    dataset = add_descriptor_indices('bond', dataset, atom_orig_index, universe.bonds.indices)
    dataset = add_descriptor_indices('angle', dataset, atom_orig_index, universe.angles.indices)
    dataset = add_descriptor_indices('dihedral', dataset, atom_orig_index, universe.dihedrals.indices)

    bond_interactions = {}
    for bond in dataset['bond_indices']:
        src, trg = bond
        bond_interactions[src] = bond_interactions.get(src, "") + dataset['atom_type'][trg]
        bond_interactions[trg] = bond_interactions.get(trg, "") + dataset['atom_type'][src]
    
    for bond_interaction in bond_interactions:
        bond_interactions[bond_interaction] = "".join(sorted(bond_interactions[bond_interaction]))
    
    monomer_names = []
    monomer_orig_atom_index = []
    for i, elem in enumerate(dataset['atom_type']):
        if elem == "H":
            continue
        try:
            monomer_names.append(f"{''.join(sorted(elem))}-{''.join(sorted(bond_interactions[i]))}")
            monomer_orig_atom_index.append(i)
        except:
            # If elements registered more than once bond interactions it means that we have multiple instances of the same molecule
            break
    monomer_names = np.array(monomer_names)
    monomer_orig_atom_index= np.array(monomer_orig_atom_index, dtype=np.int32)

    dataset['monomer_names'] = monomer_names
    dataset['monomer_orig_atom_index'] = monomer_orig_atom_index

    return dataset

def add_descriptor_indices(descriptor: str, dataset: dict, atom_orig_index: np.ndarray, orig_indices: np.ndarray):
    valid_orig_indices_filter = np.isin(orig_indices[:, 0], atom_orig_index)
    for d in range(1, orig_indices.shape[1]):
        valid_orig_indices_filter *= np.isin(orig_indices[:, d], atom_orig_index)
    orig_indices = orig_indices[valid_orig_indices_filter]
    unique_orig_indices = np.unique(orig_indices)
    indices = np.copy(orig_indices)
    for id, orig_id in enumerate(unique_orig_indices):
        indices[indices==orig_id] = id
    
    dataset[f'{descriptor}_orig_indices'] = orig_indices
    dataset[f'{descriptor}_indices'] = indices
    return dataset


if __name__ == "__main__":
    main()