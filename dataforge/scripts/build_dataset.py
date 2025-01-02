
import argparse
import os
import shutil
import logging
import json
import glob
import re

import pandas as pd
import numpy as np

from os.path import join, dirname, basename
from typing import Dict, List, Optional
from ase import Atoms
from ase.io import read, write
from itertools import islice, combinations
from dataforge.src import DataDict, fix_bonds, apply_replacements_fp
from dataforge.src.generic import read_h5_file, append_suffix_to_filename
from dataforge.src.logging import get_logger


def build_dataset(
    dataset_root: str,
    nmer_folder: Optional[str] = None,
    rule_if_file_exists: int = DataDict.SKIP,
    **kwargs,
):
    logger = get_logger('03_build_dataset.log')

    DATA_ROOT          =                                             join(dataset_root, "data"         )
    NMERS_CAPPED_ROOT  = kwargs.get('NMERS_CAPPED_ROOT' ,   None) or join(DATA_ROOT, "xyz_capped"      )
    QCHEM_OUT_ROOT     = kwargs.get('QCHEM_OUT_ROOT'    ,   None) or join(DATA_ROOT, "qchem_output"    )
    QCHEM_MIN_OUT_ROOT = kwargs.get('QCHEM_MIN_OUT_ROOT',   None) or join(DATA_ROOT, "qchem_min_output")

    FIT_ROOT           =                                             join(dataset_root, "fitting"      )
    FIT_DATASET_ROOT   = kwargs.get('FIT_DATASET_ROOT',     None) or join(FIT_ROOT , "dataset"         )
    FIT_OPTIM_ROOT     = kwargs.get('FIT_OPTIM_ROOT'  ,     None) or join(FIT_ROOT , "optimized"       )
    FIT_POLY_ROOT      = kwargs.get('FIT_POLY_ROOT'   ,     None) or join(FIT_ROOT , "poly"            )

    logger.info("- Reading/bulding energy.csv files of nmers...")
    energy_dict = build_energy_dict(
        qchem_out_root=QCHEM_OUT_ROOT,
        nmer_folder=nmer_folder,
        logger=logger,
    )

    logger.info("- Reading/bulding energy.csv files of energy-minimized nmers -")
    minimised_energy_dict = build_energy_dict(
        qchem_out_root=QCHEM_MIN_OUT_ROOT,
        nmer_folder=nmer_folder,
        logger=logger,
    )

    logger.info("- Computing nmers contribution energy relative to minimized structure -")
    delta_energies_dict = build_delta_energies_dict(
        qchem_out_root=QCHEM_OUT_ROOT,
        qchem_min_out_root=QCHEM_MIN_OUT_ROOT,
        energy_dict=energy_dict,
        minimised_energy_dict=minimised_energy_dict,
    )

    if len(delta_energies_dict) == 0:
        logger.error("- Failed to build a dictionary with nmers contribution energy. Maybe some energy calculations or energy minimizations are missing? -")
        return

    logger.info("- Building fitting/dataset -")
    build_fitting_dataset(
        data_root=DATA_ROOT,
        nmers_capped_root=NMERS_CAPPED_ROOT,
        qchem_out_root=QCHEM_OUT_ROOT,
        qchem_min_out_root=QCHEM_MIN_OUT_ROOT,
        fit_poly_root=FIT_POLY_ROOT,
        fit_dataset_root=FIT_DATASET_ROOT,
        fit_optimized_root=FIT_OPTIM_ROOT,
        delta_energies_dict=delta_energies_dict,
        rule_if_file_exists=rule_if_file_exists,
        logger=logger,
    )
    logger.info("- Completed building fitting/dataset! -")

def build_energy_dict(
    qchem_out_root: str,
    logger: logging.Logger,
    nmer_folder: Optional[str] = None,
) -> Dict[int, pd.DataFrame]:
    # energy_dict:
    #   key: nmer degree (1 for monomers, 2 for dimers...)
    #   value: DataFrame with energy values
    energy_dict: Dict[int, pd.DataFrame] = {}
    energy_dict_implement: Dict[int, List[pd.DataFrame]] = {}
    already_computed_files = set()

    # Create regular expression to optionally iterate only a subset of the folders
    folder_regex = f"**"
    if nmer_folder is not None:
        folder_regex = os.path.join(folder_regex, nmer_folder)
    folder_regex = os.path.join(folder_regex, "*.out")
    # -------------------------------------------------------------------------- #
    
    nmer_df = None
    nmer_df_loaded = False
    last_nmer_folder = ""
    # ------------------------- Iterate qchem output files ----------------------------- #
    for filename in glob.glob(os.path.join(qchem_out_root, folder_regex), recursive=True):
        bname = basename(filename)              # f{num_frame}-{monomer_id1}[_{monomer_id2}[_{monomer_id3}[...]]].out
        frame_id, nmer_idcs = bname.split('-')  # split frame number and nmer_idcs
        nmer_idcs = nmer_idcs.split('.')[0]     # remove extension .out
        nmer_idcs: List[int] = nmer_idcs.split('_')
        nmer_folder = dirname(filename)
        # When entering in a new folder, do any number of the following things:
        # - Save nmer_df containing nmer energies of the last folder to csv file (if any).
        #   Then append the nmer df content to the global energy df
        # - Check if the energy csv file exists in the new folder.
        #   If it exists, load it to memory (append its content to the global energy dataframe)
        if nmer_folder != last_nmer_folder:
            if nmer_df is not None and not nmer_df_loaded:
                last_nmer_energy_csv = write_nmer_df_and_update_energy_dict_implement(
                    energy_dict_implement,
                    nmer_df,
                    last_nmer_folder,
                    last_nmer
                )
                logger.info(f"-- {last_nmer_energy_csv.replace(qchem_out_root, '')} energy file saved! It contains {len(nmer_df)} entries.")
            
            nmer_df_loaded = False
            nmer_df = None
            nmer_energy_csv = os.path.join(nmer_folder, DataDict.ENERGY_FILENAME)
            if os.path.isfile(nmer_energy_csv):
                nmer_df = pd.read_csv(nmer_energy_csv)
                num_nmer_files = len(glob.glob(os.path.join(nmer_folder, "*.out")))
                if len(nmer_df) == num_nmer_files:    
                    # Append nmer_df to the list of all nmer DataFrames
                    nmer = len(nmer_idcs)
                    energy_dict_implement_value: List[pd.DataFrame] = energy_dict_implement.get(nmer, [])
                    energy_dict_implement_value.append(nmer_df)
                    energy_dict_implement[nmer] = energy_dict_implement_value
                    # Flag nmer_df as loaded for this folder
                    nmer_df_loaded = True
                    logger.info(f"-- {nmer_energy_csv.replace(qchem_out_root, '')} energy file loaded. It contains {len(nmer_df)} entries.")
                else:
                    len_df = len(nmer_df)
                    os.remove(nmer_energy_csv)
                    nmer_df = None
                    logger.warning(
                        f"-- {nmer_energy_csv.replace(qchem_out_root, '')} energy file exists, " +
                        f"but number of entries does not match the output files in the folder ({len_df} | {num_nmer_files}). " +
                         "Rebuilding. --"
                    )
            if not nmer_df_loaded:
                logger.info(f"-- Building {nmer_energy_csv.replace(qchem_out_root, '')} energy file...")
            last_nmer_folder = nmer_folder

        if nmer_df_loaded:
            continue
        if bname in already_computed_files:
            continue

        already_computed_files.add(bname)

        try:
            with open(filename, 'r') as f:
                for line in islice(f, 200, None):
                    if line.startswith("        RIMP2         total energy ="):
                        energy = float(re.findall(r"[-+]?(?:\d+\.\d+)", line)[0])

            nmer_name = basename(nmer_folder)
            row = {
                "name": nmer_name,
                "frame_id": int(frame_id[1:]),
                "energy": float(energy),
            }
            df_columns = ["name", "frame_id", "energy"]
            for j, monomer_id in enumerate(nmer_idcs):
                col_name = f"monomer_{j}"
                df_columns.append(col_name)
                row[col_name] = int(monomer_id)
            row = pd.DataFrame([row])
            if nmer_df is None:
                nmer_df = row
            else:
                nmer_df = pd.concat([nmer_df, row]).reset_index(drop=True)

            last_nmer = len(nmer_idcs) # 1 for monomers, 2 for dimers, ...
        except Exception:
            logger.error(f"-- Failed parsing file {filename}. Check if the file is corrupted or if the computation is not yet finished --")
    
    if nmer_df is not None and not nmer_df_loaded:
        last_nmer_energy_csv = write_nmer_df_and_update_energy_dict_implement(
            energy_dict_implement,
            nmer_df,
            last_nmer_folder,
            last_nmer
        )
        logger.info(f"-- {last_nmer_energy_csv.replace(qchem_out_root, '')} energy file saved! It contains {len(nmer_df)} entries.")

    for k, v in energy_dict_implement.items():
        energy_dict[k] = pd.concat(v).reset_index(drop=True)
    
    return energy_dict

def write_nmer_df_and_update_energy_dict_implement(
        energy_dict_implement: Dict[int, List[pd.DataFrame]],
        nmer_df: pd.DataFrame,
        last_nmer_folder: str,
        last_nmer: int,
    ):
        # Save nmer_df to csv file
        last_nmer_energy_csv = os.path.join(last_nmer_folder, DataDict.ENERGY_FILENAME)
        nmer_df.to_csv(last_nmer_energy_csv, index=False)
        # Append nmer_df to the list of all nmer DataFrames
        energy_dict_implement_value: List[pd.DataFrame] = energy_dict_implement.get(last_nmer, [])
        energy_dict_implement_value.append(nmer_df)
        energy_dict_implement[last_nmer] = energy_dict_implement_value
        return last_nmer_energy_csv

def build_delta_energies_dict(
    qchem_out_root: str,
    qchem_min_out_root: str,
    energy_dict: dict,
    minimised_energy_dict: dict,
):
    delta_energies_dict = {}

    for (k, minEn) in minimised_energy_dict.items():
        En = energy_dict.get(k, None)
        if En is None:
            continue
        # Associate all energies and minimised energies of monomers with the same name
        deltaEn = pd.merge(
            En, minEn,  how='inner',
            left_on=['name'] + [f'monomer_{k}' for k in range(k)],
            right_on = ['name'] + [f'monomer_{k}' for k in range(k)],
        )
        deltaEn.drop_duplicates(subset=['name', 'frame_id_x'] + [f'monomer_{k}' for k in range(k)], keep='first', inplace=True, ignore_index=True)

        # Extract monomers that have different names but same minimised structure
        monomers_associations = []
        for min_filename in glob.glob(os.path.join(qchem_min_out_root, "**/*.out"), recursive=True):
            min_monomers = basename(min_filename).split('.')[0].split('-')[-1].split('_')
            if len(min_monomers) != k:
                continue
            for filename in glob.glob(os.path.join(qchem_out_root, f"**/{basename(dirname(min_filename))}/*.out"), recursive=True):
                monomers = basename(filename).split('.')[0].split('-')[-1].split('_')
                if monomers == min_monomers:
                    continue
                monomers_association = tuple([int(x) for x in monomers + min_monomers])
                if monomers_association not in monomers_associations:
                    monomers_associations.append(monomers_association)

        # Add monomers associations
        allDeltaEn = pd.merge(En, minEn,  how='left', left_on=['name'], right_on = ['name'])
        queries = []
        for ma in monomers_associations:
            queries.append(' & '.join([f'{k} == {v}' for k, v in zip([f'monomer_{k}_x' for k in range(k)] + [f'monomer_{k}_y' for k in range(k)], ma)]))
        for query in queries:
            deltaEnToKeep = allDeltaEn.query(query)
            deltaEnToKeep = deltaEnToKeep.drop(columns=[f'monomer_{k}_y' for k in range(k)])
            deltaEnToKeep = deltaEnToKeep.rename(columns={f"monomer_{k}_x": f"monomer_{k}" for k in range(k)})
            deltaEn = pd.concat([deltaEn, deltaEnToKeep])

        # Compute delta energy
        deltaEn = deltaEn.drop_duplicates()
        deltaEn["delta_energy"] = deltaEn["energy_x"] - deltaEn["energy_y"]
        delta_energies_dict[k] = deltaEn
    
    return delta_energies_dict

def get_nmer_filename(frame: int, monomers: List[int]):
    return f"f{str(frame)}-" + '_'.join([str(m) for m in monomers]) + '.xyz'

def get_nmer_unique_name(frame: int, monomers: List[int]):
    fname = f"f{str(frame)}-"
    topname = '_'.join([str(m) for m in sorted(monomers)])
    return fname + topname, topname

def recursive_energy_calculation(
        nmer_name: str,
        frame: int,
        monomers: List[int],
        all_energies_contrib_dict: Dict[str, pd.DataFrame],
        energy: float,
        energy_contribution: float,
        k: int,
        topology: Dict,
        binding_energy_contribution: float = 0.0,
    ):
    if k == 1:
        # ---- (total energy, nmer contribution energy, binding energy) --------------------- #
        return [energy, energy_contribution, energy_contribution + binding_energy_contribution]
    
    for subset_monomers in combinations(monomers, k-1):
        nmer_unique_name, connection_name = get_nmer_unique_name(frame, subset_monomers)
        if len(subset_monomers) > 1 and connection_name not in topology["connections"]:
            continue
        subset_monomer_names_list = [topology["monomers"][str(m)] for m in subset_monomers]
        subset_nmer_name = f"{'.'.join(sorted(subset_monomer_names_list))}_{'_'.join([str(m) for m in subset_monomers])}.h5"
        nmer_energies_contrib_df = all_energies_contrib_dict[subset_nmer_name]
        if nmer_unique_name in nmer_energies_contrib_df["nmer_unique_name"].values:
            nmer_row = nmer_energies_contrib_df.loc[nmer_energies_contrib_df["nmer_unique_name"] == nmer_unique_name]
            nmer_contribution = nmer_row["nmer_energy"].iloc[0]
            energy_contribution -= nmer_contribution
            if k-1 == 1:
                binding_energy_contribution += nmer_contribution
        else:
            raise LookupError(f"{subset_nmer_name} with unique name {nmer_unique_name} is missing in the energy contribution dictionary.")

    return recursive_energy_calculation(
        nmer_name=nmer_name,
        frame=frame,
        monomers=monomers,
        all_energies_contrib_dict=all_energies_contrib_dict,
        energy=energy,
        energy_contribution=energy_contribution,
        k=k-1,
        topology=topology,
        binding_energy_contribution=binding_energy_contribution,
    )

def argofyinx(x, y):
    index = np.argsort(x)
    sorted_x = x[index]
    sorted_index = np.searchsorted(sorted_x, y)

    yindex = np.take(index, sorted_index, mode="clip")
    mask = x[yindex] != y

    return np.ma.array(yindex, mask=mask)

def build_fitting_dataset(
    data_root: str,
    nmers_capped_root: str,
    qchem_out_root: str,
    qchem_min_out_root: str,
    fit_poly_root: str,
    fit_dataset_root: str,
    fit_optimized_root: str,
    delta_energies_dict: dict,
    logger: logging.Logger,
    rule_if_file_exists: int = DataDict.SKIP,
):
    save_optimized_structures(qchem_min_out_root, fit_optimized_root, fit_poly_root)

    all_energies_contrib_dict = {}
    with open(os.path.join(data_root, DataDict.TOPOLOGY_FILENAME), 'r') as topology_f:
        topology = json.load(topology_f)
    for k in range(1, max([k for k in delta_energies_dict.keys()]) + 1):
        df = delta_energies_dict[k]
        df['fullname'] = df.apply(lambda row: '_'.join([str(row['name'])] + [str(row[f'monomer_{i}']) for i in range(k)]) + '.h5', axis=1)
        logger.info(f"-- Building  dataset for {DataDict.FOLDER_NAMES[k]} --")
        
        # ------------------------------- Work one nmer at a time ---------------------------------- #
        for nmer_name in df["fullname"].unique():
            nmer_df = df[df["fullname"] == nmer_name]
            nmer_energy_contrib_df_total: Optional[pd.DataFrame] = all_energies_contrib_dict.get(nmer_name, None)

            # -------------------------- Iterate xyz_capped folders -------------------------------- #
            nmer_capping_folder_regex = os.path.join(nmers_capped_root, "**", DataDict.FOLDER_NAMES[k], "**", nmer_name)
            for nmer_capping_filename in glob.glob(nmer_capping_folder_regex, recursive=True):
                nmer_capping_folder = dirname(nmer_capping_filename)
                out_filename = nmer_capping_filename.replace(nmers_capped_root, fit_dataset_root).replace('.h5', '.xyz')
                nmer_dataset_folder = nmer_capping_folder.replace(nmers_capped_root, fit_dataset_root)
                monomers_name = nmer_df.apply(lambda row: '_'.join([str(row[f'monomer_{i}']) for i in range(k)]), axis=1).unique()
                assert len(monomers_name) == 1
                monomers_name = monomers_name[0]
                nmer_energy_contrib_csv = os.path.join(nmer_dataset_folder, f"{monomers_name}_{DataDict.ENERGY_FILENAME}")
                nmer_energy_contrib_csv_kcal = os.path.join(nmer_dataset_folder, f"{monomers_name}_{DataDict.ENERGY_FILENAME_KCAL}")

                if os.path.isfile(nmer_energy_contrib_csv):
                    if rule_if_file_exists is DataDict.SKIP:
                        # read nmer_energy_contrib_df and append it to the
                        # list of all nmer energy contribution DataFrames
                        nmer_energy_contrib_df = pd.read_csv(nmer_energy_contrib_csv)
                        # concat nmer_energy_contrib_df to nmer_energy_contrib_df_total
                        nmer_energy_contrib_df_total = update_nmer_df(
                            nmer_energy_contrib_df_total,
                            nmer_energy_contrib_df
                        )
                        all_energies_contrib_dict[nmer_name] = nmer_energy_contrib_df_total
                        continue
                    if rule_if_file_exists not in [DataDict.APPEND, DataDict.OVERWRITE]:
                        raise ValueError()

                # ------------- Iterate xyz_capped files inside nmer folder ------------------------ #
                logger.info(f"--- Building  dataset for nmer {nmer_capping_folder.replace(nmers_capped_root, '')} ---")
                nmer_energy_contrib_df = None
                # Check if qchem output folder exists, to avoid an empty search in the nmer_df
                qchem_out_folder = nmer_capping_folder.replace(nmers_capped_root, qchem_out_root)
                if not os.path.isdir(qchem_out_folder):
                    logger.warning(f"--- Missing qchem output files for nmer {nmer_capping_folder.replace(nmers_capped_root, '')} ---")
                    continue

                all_coords, all_atom_types, all_info_dicts, _ = read_h5_file(nmer_capping_filename)
                all_fullnames = np.array([d['fullname'] for d in all_info_dicts])

                for _, row in nmer_df.iterrows():
                    frame = row.frame_id_x
                    monomers: List[int] = [int(x) for x in [row[f'monomer_{i}'] for i in range(k)]]
                    h5_fullname = f"{frame}_{row['name']}_{'_'.join([str(m) for m in monomers])}"
                    h5_index = np.argwhere(all_fullnames == h5_fullname).squeeze()
                    assert h5_index.shape == ()
                    energy = row.delta_energy
                    try:
                        value = recursive_energy_calculation(
                            nmer_name=nmer_name,
                            frame=frame,
                            monomers=monomers,
                            all_energies_contrib_dict=all_energies_contrib_dict,
                            energy=energy,
                            energy_contribution=energy,
                            k=k,
                            topology=topology,
                        )
                    except KeyError as e:
                        logger.warning(f"--- Missing energy contribution for nmer {str(e)}. Check if you are missing qchem output files ---")
                        break
                    except LookupError as e:
                        logger.warning(f"--- {str(e)} ---")
                        break
                    nmer_energy_contrib_df_row = pd.DataFrame(
                        [{
                            "name":             nmer_name,
                            "nmer_filename":    get_nmer_filename(frame, monomers),
                            "nmer_unique_name": get_nmer_unique_name(frame, monomers)[0],
                            "total_energy":     value[0],
                            "nmer_energy":      value[1],
                            "binding_energy":   value[2],
                            "frame":            frame,
                            "monomers":         monomers,
                        }]
                    )
                    nmer_energy_contrib_df = update_nmer_df(
                        nmer_energy_contrib_df,
                        nmer_energy_contrib_df_row,
                    )

                    # ---------------- Writing data/dataset file for current nmer ------------------ #

                    atoms = Atoms(positions=all_coords[h5_index], symbols=all_atom_types[h5_index])
                    atoms.info = {key: val for key, val in zip(["total_energy", "nmer_energy", "binding_energy"], value)}
                    info = f"{value[1]} {value[2]} {h5_fullname}"
                    
                    os.makedirs(nmer_dataset_folder, exist_ok=True)
                    write(
                        out_filename,
                        atoms,
                        comment=info,
                        format="extxyz",
                        append=True,
                    )

                    # ------------------------------------------------------------------------------ #

                append_connection_info_to_xyz_file(xyz_filename=out_filename, fit_poly_folder=dirname(out_filename).replace(fit_dataset_root, fit_poly_root))
                fix_bonds(out_filename)
                convert_unit(out_filename, hartrees2kcalmol)
                # ---------------- End iterate xyz_capped files inside nmer folder ----------------- #
                
                # save nmer_energy_contrib_df to csv
                if nmer_energy_contrib_df is None:
                    logger.warning(f"--- Missing qchem output files for nmer {nmer_capping_folder.replace(nmers_capped_root, '')} ---")
                else:
                    nmer_energy_contrib_df.to_csv(nmer_energy_contrib_csv, index=False)
                    
                    # Save a version of energy dict in Kcal/mol
                    nmer_energy_contrib_df_kcal = nmer_energy_contrib_df.copy()
                    # Apply the conversion function to specified columns
                    columns_to_convert = ["total_energy", "nmer_energy", "binding_energy"]
                    nmer_energy_contrib_df_kcal[columns_to_convert] = nmer_energy_contrib_df_kcal[columns_to_convert].applymap(hartrees2kcalmol)
                    nmer_energy_contrib_df_kcal.to_csv(nmer_energy_contrib_csv_kcal, index=False)

                    # concat nmer_energy_contrib_df to nmer_energy_contrib_df_total
                    nmer_energy_contrib_df_total = update_nmer_df(
                            nmer_energy_contrib_df_total,
                            nmer_energy_contrib_df
                        )
                    all_energies_contrib_dict[nmer_name] = nmer_energy_contrib_df_total

                    logger.info(f"--- Completed dataset for nmer {nmer_capping_folder.replace(nmers_capped_root, '')} ---")
                # ---------------------------------------------------------------------------------- #
        logger.info(f"-- Completed dataset for {DataDict.FOLDER_NAMES[k]} --")

def update_nmer_df(
    base_df: Optional[pd.DataFrame],
    new_df: pd.DataFrame
):
    if base_df is None:
        base_df = new_df
    else:
        base_df = pd.concat([base_df, new_df]).reset_index(drop=True)
    return base_df

def find_directories_with_name(root_folder, directory_name):

    def fast_scandir(dirname):
        subfolders= [f.path for f in os.scandir(dirname) if f.is_dir()]
        for dirname in list(subfolders):
            subfolders.extend(fast_scandir(dirname))
        return subfolders
    
    return [dir for dir in fast_scandir(root_folder) if dir.endswith(directory_name)]

def hartrees2kcalmol(energy: float):
    return energy * 627.5096080305927

def convert_unit(filename: str, func):
    out_filename = append_suffix_to_filename(filename, '.kcal')
    with open(filename, 'r') as f_in, open(out_filename, 'w') as f_out:
        for line in f_in.readlines():
            line_splits = line.split()
            if len(line_splits) >= 2: # xyz header can contain multiple info
                try:
                    nmer_energy    = func(float(line_splits[0]))
                    binding_energy = func(float(line_splits[1]))
                    line = ' '.join([str(nmer_energy), str(binding_energy)] + line_splits[2:]) + '\n'
                except ValueError:
                    # Ignore lines where conversion fails, as those are lines where first word is atom name
                    pass
            f_out.writelines(line)

def save_optimized_structures(
        qchem_min_out_root: str,
        fit_optimized_root: str,
        fit_poly_root: str,
    ):
    for filename in glob.glob(os.path.join(qchem_min_out_root, "**/*.out"), recursive=True):
        out_filename = os.path.join(dirname(filename.replace(qchem_min_out_root, fit_optimized_root)), "nmers.opt")
        out_filename = apply_replacements_fp(out_filename)
        if os.path.isfile(out_filename):
            continue
        os.makedirs(dirname(out_filename), exist_ok=True)
        
        qchem_output_file = glob.glob(f"{qchem_min_out_root}/**/{basename(dirname(filename))}/*.out", recursive=True)[0]

        pattern = re.compile(r'.+\s*(\w+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)')

        reading_state = DataDict.NAPPING
        coords_list, symbols_list = [], []
        with open(qchem_output_file, 'r') as f:
            for line in islice(f, 100, None):
                if reading_state is DataDict.NAPPING:
                    if line.startswith("             Standard Nuclear Orientation (Angstroms)"):
                        reading_state = DataDict.READY
                        coords_list.clear()
                        symbols_list.clear()
                    continue
                if reading_state is DataDict.READY and line.startswith(" ----------------------------------------------------------------"):
                    reading_state = DataDict.READING
                    continue
                if reading_state is DataDict.READING and line.startswith(" ----------------------------------------------------------------"):
                    reading_state = DataDict.NAPPING
                    continue
                if reading_state is DataDict.READING:
                    match = pattern.match(line)
                    if match:
                        x, y, z = map(float, match.groups()[1:])
                        symbol = match.groups()[0]
                        coords_list.append([x, y, z])
                        symbols_list.append(symbol)
        coords = np.array(coords_list, dtype=np.float64)
        symbols = np.array(symbols_list)
        
        atoms = Atoms(positions = coords, symbols = symbols)
        write(
            out_filename,
            atoms,
            comment="0.0 0.0",
            format="extxyz",
            append=False,
        )
        append_connection_info_to_xyz_file(xyz_filename=out_filename, fit_poly_folder=dirname(out_filename).replace(fit_optimized_root, fit_poly_root))
        fix_bonds(out_filename)

def append_connection_info_to_xyz_file(xyz_filename: str, fit_poly_folder: str):
    fit_poly_filename = apply_replacements_fp(os.path.join(fit_poly_folder, "poly_generator.py"))

    pattern = re.compile(r"'(.*?)'")
    connections_list = []
    with open(fit_poly_filename, 'r') as f:
        for line in f.readlines():
            if line.startswith("add_atom["):
                connections = pattern.findall(line)
                connections_list.append(connections)
    
    temp_filename = append_suffix_to_filename(xyz_filename, '.temp')
    with open(xyz_filename, 'r') as f, open(temp_filename, 'w') as temp_f:
        for line in f.readlines():
            splits = len(line.split())
            if splits == 1:
                assert int(line) == len(connections_list)
                atom_counter = 0
            elif splits == 4:
                line = line.rstrip('\n') + ' ' + ' '.join(connections_list[atom_counter]) + '\n'
                atom_counter += 1
            temp_f.write(line)
    
    shutil.move(temp_filename, xyz_filename)

def build_all_energies_contrib_dict(
    nmers_capped_root: str,
    dataset_root: str,
    delta_energies_dict: dict,
):
    logger = get_logger()
    
    logger.info("- Building a dictionary with all nmers energy contributions -")
    all_energies_contrib_dict = {}

    for k in sorted(delta_energies_dict.keys()):
        df = delta_energies_dict[k]
        logger.info(f"-- Building dictionary for {DataDict.FOLDER_NAMES[k]} --")
        
        # ------------------------------- Work one nmer at a time ---------------------------------- #
        for nmer_name in df["name"].unique():
            nmer_energy_contrib_df_total: Optional[pd.DataFrame] = all_energies_contrib_dict.get(nmer_name, None)

            # -------------------------- Iterate xyz_capped folders -------------------------------- #
            nmer_capping_folder_regex = os.path.join(nmers_capped_root, "**", DataDict.FOLDER_NAMES[k], nmer_name)
            for nmer_capping_folder in glob.glob(nmer_capping_folder_regex, recursive=True):
                nmer_dataset_folder = nmer_capping_folder.replace(nmers_capped_root, dataset_root)
                nmer_energy_contrib_csv = os.path.join(nmer_dataset_folder, DataDict.ENERGY_FILENAME)
                if os.path.isfile(nmer_energy_contrib_csv):
                    # read nmer_energy_contrib_df and append it to the
                    # list of all nmer energy contribution DataFrames
                    nmer_energy_contrib_df = pd.read_csv(nmer_energy_contrib_csv)
                    # concat nmer_energy_contrib_df to nmer_energy_contrib_df_total
                    nmer_energy_contrib_df_total = update_nmer_df(
                        nmer_energy_contrib_df_total,
                        nmer_energy_contrib_df
                    )
                    all_energies_contrib_dict[nmer_name] = nmer_energy_contrib_df_total
                else:
                    logger.warning(f"-- Missing nmer energy contribution csv file {nmer_energy_contrib_csv.replace(nmers_capped_root, '')} --")
    logger.info("- Dictionary completed! -")
    return all_energies_contrib_dict


if __name__ == "__main__":
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Run the build_dataset method with user-specified inputs.")

    # Required argument
    parser.add_argument('dataset_root', type=str, help="The root directory of the dataset")

    # Optional arguments
    parser.add_argument('--nmer_folder', type=str, help="The nmer folder, optional", default=None)
    parser.add_argument('--rule_if_file_exists', type=int, help="Rule for handling existing files (default is 0)", default=0)
    
    # Optional kwargs
    parser.add_argument('--NMERS_CAPPED_ROOT', type=str, help="Optional path for NMERS_CAPPED_ROOT", default=None)
    parser.add_argument('--QCHEM_OUT_ROOT', type=str, help="Optional path for QCHEM_OUT_ROOT", default=None)
    parser.add_argument('--QCHEM_MIN_OUT_ROOT', type=str, help="Optional path for QCHEM_MIN_OUT_ROOT", default=None)
    parser.add_argument('--DATASET_ROOT', type=str, help="Optional path for DATASET_ROOT", default=None)
    parser.add_argument('--FIT_DATASET_ROOT', type=str, help="Optional path for FIT_DATASET_ROOT", default=None)
    parser.add_argument('--FIT_OPTIM_ROOT', type=str, help="Optional path for FIT_OPTIM_ROOT", default=None)
    parser.add_argument('--FIT_POLY_ROOT', type=str, help="Optional path for FIT_POLY_ROOT", default=None)

    # Parse the arguments
    args = parser.parse_args()

    # Prepare kwargs from optional arguments
    kwargs = {
        'NMERS_CAPPED_ROOT': args.NMERS_CAPPED_ROOT,
        'QCHEM_OUT_ROOT': args.QCHEM_OUT_ROOT,
        'QCHEM_MIN_OUT_ROOT': args.QCHEM_MIN_OUT_ROOT,
        'DATASET_ROOT': args.DATASET_ROOT,
        'FIT_DATASET_ROOT': args.FIT_DATASET_ROOT,
        'FIT_OPTIM_ROOT': args.FIT_OPTIM_ROOT,
        'FIT_POLY_ROOT': args.FIT_POLY_ROOT
    }

    # Call the function with parsed arguments
    build_dataset(
        dataset_root=args.dataset_root,
        nmer_folder=args.nmer_folder,
        rule_if_file_exists=args.rule_if_file_exists,
        **{k: v for k, v in kwargs.items() if v is not None}
    )