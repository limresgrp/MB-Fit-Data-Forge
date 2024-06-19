import os
import glob
import ase.io

from os.path import dirname, basename
from ase.symbols import chemical_symbols


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


def prepare_qchem_input(
        nmers_capped_root: str,
        qchem_in_root: str,
        qchem_min_in_root: str,
        charges_dict: dict,
):
    for filename in glob.glob(os.path.join(nmers_capped_root, "**/*.xyz"), recursive=True):
        try:
            qchem_in_root_folder = dirname(filename).replace(nmers_capped_root, qchem_in_root)
            if not os.path.exists(qchem_in_root_folder):
                os.makedirs(qchem_in_root_folder)
            
            output_filename = os.path.join(qchem_in_root_folder, basename(filename).split('.')[0] + '.inp')
            if os.path.exists(output_filename):
                continue
            
            nmer = ase.io.read(filename, index=":", format="extxyz")[0]
            in_nmer_folder = basename(dirname(filename))
            
            charge = 0
            pre_name = in_nmer_folder.split('|')[0]
            for monomer_name in pre_name.split('.'):
                ch = charges_dict.get(monomer_name, None)
                if ch is not None:
                    charge += ch
            multiplicity = 1
            
            input_file = BLUEPRINT_1
            input_file += f"{charge} {multiplicity}\n"

            for line in [
                f'{chemical_symbols[atom_type]} {" ".join([str(x) for x in pos])}\n' for atom_type, pos in zip(nmer.arrays['numbers'], nmer.arrays['positions'])
            ]:
                input_file += line
            input_file += BLUEPRINT_2
            
            with open(output_filename, 'w') as f:
                f.write(input_file)
        except Exception as e:
            print(e)
            print(f"Skipping {filename}.")
    
    prepare_qchem_min_input(
        qchem_in_root,
        qchem_min_in_root,
    )

def prepare_qchem_min_input(
        qchem_in_root: str,
        qchem_min_in_root: str,
):
    
    processed_nmer_names = set()
    
    for filename in glob.glob(os.path.join(qchem_in_root, "**/**/*.inp"), recursive=True):
        nmer_name = basename(dirname(filename))
        if nmer_name in processed_nmer_names:
            continue
        processed_nmer_names.add(nmer_name)

        qchem_min_in_filename = filename.replace(qchem_in_root, qchem_min_in_root)
        qchem_min_in_folder = dirname(qchem_min_in_filename)
        if not os.path.exists(qchem_min_in_folder):
            os.makedirs(qchem_min_in_folder)
        
        with open(filename, 'r') as f:
            input_file = f.read()
        
        input_file = input_file.replace("jobtype sp", "jobtype opt")
        
        with open(qchem_min_in_filename, 'w') as f:
            f.write(input_file)