import functools
from logging import Logger
import multiprocessing
import os
import glob
import numpy as np

from os.path import dirname, basename
from dataforge.src.generic import read_h5_file


BLUEPRINT_1 = '''$molecule
'''

BLUEPRINT_2 = '''$end

$rem
jobtype sp
method RI-MP2
basis aug-cc-pVTZ
MEM_TOTAL 12800
AUX_BASIS RIMP2-aug-cc-pVTZ
$end
'''


def _frame_filter(h5_filepath, skip_if_not_frame_filter):
    list_filepath = h5_filepath.replace('.h5', '.list')
    if os.path.exists(list_filepath):
        with open(list_filepath, 'r') as f:
            values = [int(line.strip()) for line in f if line.strip()]
        return np.asarray(values, dtype=int)
    return np.array([], dtype=int) if skip_if_not_frame_filter else None


def _nmer_charge(h5_filepath, charges_dict):
    charge = 0
    pre_name = basename(dirname(h5_filepath)).split('|')[0]
    for monomer_name in pre_name.split('.'):
        charge += charges_dict.get(monomer_name, 0)
    return charge


def _write_qchem_file(output_filename, coords, atom_types, charge, jobtype):
    input_file = BLUEPRINT_1
    input_file += f"{charge} 1\n"
    for atom_type, pos in zip(atom_types, coords):
        input_file += f'{atom_type} {" ".join(str(x) for x in pos)}\n'
    input_file += BLUEPRINT_2.replace("jobtype sp", f"jobtype {jobtype}")
    os.makedirs(dirname(output_filename), exist_ok=True)
    if not os.path.exists(output_filename):
        with open(output_filename, 'w') as f:
            f.write(input_file)


def write_qchem_input(h5_filepath: str, nmers_capped_root: str, qchem_in_root: str, charges_dict: dict, skip_if_not_frame_filter=True):
    frame_filter = _frame_filter(h5_filepath, skip_if_not_frame_filter)
    if frame_filter is not None and len(frame_filter) == 0:
        return 0
    
    qchem_in_root_folder = dirname(h5_filepath).replace(nmers_capped_root, qchem_in_root)
    os.makedirs(qchem_in_root_folder, exist_ok=True)
    
    # Load the H5 file saved in save_multimer
    all_coords, atom_types, fullnames, _, extra_info = read_h5_file(h5_filepath, rows_filter=frame_filter)
    charge = _nmer_charge(h5_filepath, charges_dict)

    print(f"Writing QChem input files for {h5_filepath}...")
    for coords, fullname in zip(all_coords, fullnames):
        splits = fullname.split('_')
        frame_id = splits[0]
        monomer_idcs = splits[-int(extra_info.get("num_monomers")):]
        output_filename = os.path.join(qchem_in_root_folder, f'f{frame_id}-{"_".join(monomer_idcs)}' + '.inp')
        _write_qchem_file(output_filename, coords, atom_types, charge, "sp")
    return len(all_coords)


def write_qchem_minimization_input(
    h5_filepath: str,
    nmers_capped_root: str,
    qchem_min_in_root: str,
    charges_dict: dict,
):
    """Write one optimization input from the first selected frame of an H5."""
    frame_filter = _frame_filter(h5_filepath, skip_if_not_frame_filter=False)
    frame_index = int(frame_filter[0]) if frame_filter is not None and len(frame_filter) else 0
    coords, atom_types, fullnames, _, extra_info = read_h5_file(
        h5_filepath,
        rows_filter=np.asarray([frame_index], dtype=int),
    )
    if len(coords) == 0:
        return 0

    relative_folder = os.path.relpath(dirname(h5_filepath), nmers_capped_root)
    output_folder = os.path.join(qchem_min_in_root, relative_folder)
    fullname = fullnames[0]
    splits = fullname.split('_')
    frame_id = splits[0]
    num_monomers = int(extra_info.get("num_monomers"))
    monomer_idcs = splits[-num_monomers:]
    output_filename = os.path.join(output_folder, f'f{frame_id}-{"_".join(monomer_idcs)}.inp')
    _write_qchem_file(
        output_filename,
        coords[0],
        atom_types,
        _nmer_charge(h5_filepath, charges_dict),
        "opt",
    )
    return 1

def write_qchem_min_input(h5_min_filepath: str, qchem_in_root: str, qchem_min_in_root: str):
    qchem_min_in_filepath = h5_min_filepath.replace(qchem_in_root, qchem_min_in_root)
    qchem_min_in_folder = dirname(qchem_min_in_filepath)
    os.makedirs(qchem_min_in_folder, exist_ok=True)
    
    with open(h5_min_filepath, 'r') as f:
        input_file = f.read()
    
    input_file = input_file.replace("jobtype sp", "jobtype opt")
    
    with open(qchem_min_in_filepath, 'w') as f:
        f.write(input_file)

def prepare_qchem_input(
        nmers_capped_root: str,
        qchem_in_root: str,
        qchem_min_in_root: str,
        charges_dict: dict,
        logger: Logger = None,
        max_processes: int = 4,
        skip_if_not_frame_filter: bool = True,
        minimization_only: bool = False,
        create_minimization_inputs: bool = True,
        selected_nmer_names=None,
):
    if logger is None:
        import logging
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
    logger.info("- Preparing QChem input files...")

    h5_filepaths = sorted(glob.iglob(os.path.join(nmers_capped_root, "**/*.h5"), recursive=True))
    if selected_nmer_names is not None:
        selected = set(selected_nmer_names)
        h5_filepaths = [
            path for path in h5_filepaths if basename(dirname(path)) in selected
        ]
    if not h5_filepaths:
        raise FileNotFoundError(f"No selected capped HDF5 files found under {nmers_capped_root}")

    if minimization_only:
        # A folder represents one n-mer type. Select one occurrence and one
        # frame per folder, matching the historical minimization workflow
        # without first materializing every single-point input.
        representative_files = []
        seen_folders = set()
        for h5_filepath in h5_filepaths:
            folder = dirname(h5_filepath)
            if folder not in seen_folders:
                representative_files.append(h5_filepath)
                seen_folders.add(folder)
        func = functools.partial(
            write_qchem_minimization_input,
            nmers_capped_root=nmers_capped_root,
            qchem_min_in_root=qchem_min_in_root,
            charges_dict=charges_dict,
        )
        if max_processes > 0:
            with multiprocessing.Pool(processes=min(max_processes, len(representative_files))) as pool:
                prepared_count = sum(pool.map(func, representative_files))
        else:
            prepared_count = sum(func(path) for path in representative_files)
        if prepared_count == 0:
            raise RuntimeError("No QChem minimization inputs were prepared.")
        logger.info("Prepared %d QChem minimization inputs in %s", prepared_count, qchem_min_in_root)
        return prepared_count

    func = functools.partial(
        write_qchem_input,
        nmers_capped_root=nmers_capped_root,
        qchem_in_root=qchem_in_root,
        charges_dict=charges_dict,
        skip_if_not_frame_filter=skip_if_not_frame_filter,
    )
    if max_processes > 0:
        with multiprocessing.Pool(processes=max_processes) as pool:
            prepared_count = sum(pool.map(func, h5_filepaths))
    else:
        prepared_count = sum(func(h5_filepath) for h5_filepath in h5_filepaths)
    if prepared_count == 0:
        qualifier = " with .list frame filters" if skip_if_not_frame_filter else ""
        raise RuntimeError(f"No QChem single-point inputs were prepared{qualifier}.")
    logger.info("Prepared %d QChem single-point inputs in %s", prepared_count, qchem_in_root)

    if not create_minimization_inputs:
        return prepared_count
    
    def get_inp_files(qchem_in_root):
        h5_filepaths = []
        for root, dirs, files in os.walk(qchem_in_root):
            for file in files:
                if file.endswith(".inp"):
                    h5_filepaths.append(os.path.join(root, file))
                    break  # Only take one file per subfolder
        return h5_filepaths
    
    h5_min_filepaths = get_inp_files(qchem_in_root)
    if max_processes > 0:
        with multiprocessing.Pool(processes=max_processes) as pool:
            pool.starmap(write_qchem_min_input, [(h5_min_filepath, qchem_in_root, qchem_min_in_root) for h5_min_filepath in h5_min_filepaths])
    else:
        for h5_min_filepath in h5_min_filepaths:
            write_qchem_min_input(h5_min_filepath, qchem_in_root, qchem_min_in_root)

    logger.info("- Completed preparing QChem input files!")
    return prepared_count
