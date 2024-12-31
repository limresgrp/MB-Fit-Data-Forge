from logging import Logger
import multiprocessing
import os
import glob

from os.path import dirname, basename
from dataforge.src.generic import parse_string_to_dict, read_h5_file


BLUEPRINT_1 = f'''```
$molecule
'''

BLUEPRINT_2 = '''$end

$rem
jobtype sp
method RI-MP2
basis aug-cc-pVTZ
MEM_TOTAL 12800
AUX_BASIS RIMP2-aug-cc-pVTZ
$end
```'''


def write_qchem_input(h5_filepath: str, nmers_capped_root: str, qchem_in_root: str, charges_dict: dict):
    qchem_in_root_folder = dirname(h5_filepath).replace(nmers_capped_root, qchem_in_root)
    os.makedirs(qchem_in_root_folder, exist_ok=True)
    
    # Load the H5 file saved in save_multimer
    all_coords, all_atom_types, all_info_strings = read_h5_file(h5_filepath)
    in_nmer_folder = basename(dirname(h5_filepath))
    
    charge = 0
    pre_name = in_nmer_folder.split('|')[0]
    for monomer_name in pre_name.split('.'):
        ch = charges_dict.get(monomer_name, None)
        if ch is not None:
            charge += ch
    multiplicity = 1

    for coords, atom_types, info_string in zip(all_coords, all_atom_types, all_info_strings):
        info_dict = parse_string_to_dict(info_string)
        output_filename = os.path.join(qchem_in_root_folder, info_dict.get('fullname') + '.inp')
        if os.path.exists(output_filename):
            continue
        
        input_file = BLUEPRINT_1
        input_file += f"{charge} {multiplicity}\n"

        for line in [
            f'{atom_type} {" ".join([str(x) for x in pos])}\n' for atom_type, pos in zip(atom_types, coords)
        ]:
            input_file += line
        input_file += BLUEPRINT_2
        
        with open(output_filename, 'w') as f:
            f.write(input_file)

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
):
    if logger is None:
        import logging
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
    logger.info("- Preparing QChem input files...")

    h5_filepaths = list(glob.iglob(os.path.join(nmers_capped_root, "**/*.h5"), recursive=True))
    if max_processes > 0:
        with multiprocessing.Pool(processes=max_processes) as pool:
            pool.starmap(write_qchem_input, [(h5_filepath, nmers_capped_root, qchem_in_root, charges_dict) for h5_filepath in h5_filepaths])
        pool.join()
    else:
        for h5_filepath in h5_filepaths:
            write_qchem_input(h5_filepath, nmers_capped_root, qchem_in_root, charges_dict)
    
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
        pool.join()
    else:
        for h5_min_filepath in h5_min_filepaths:
            write_qchem_min_input(h5_min_filepath, qchem_in_root, qchem_min_in_root)

    logger.info("- Completed preparing QChem input files!")