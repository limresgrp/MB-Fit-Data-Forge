# MB-Fit-Data-Forge #

## Description ##

MB-Fit-Data-Forge is a tool used to prepare input data for MB-Fit, starting from molecular trajectories.

## Requirements ##

- Python >= 3.8
- Additional libraries are specified in the setup.py file and will be automatically installed when installing the `dataforge` python package.

## Installation ##

1. Clone the repository:

    ```
    git clone https://github.com/limresgrp/MB-Fit-Data-Forge.git
    ```

2. Navigate to the project directory:

    ```
    cd MB-Fit-Data-Forge/
    ```

3. Install dependencies:

    ```
    pip install -e .
    ```

Alternatively, create the project virtual environment and install the repository with all declared dependencies using:

```bash
./scripts/setup_venv.sh
source .venv/bin/activate
```

The setup script looks for `python3.11`, then `python3.10`, then `python3`. This also handles systems where interactive aliases point `python`/`python3` to a newer interpreter, since aliases are not inherited by shell scripts. If the selected interpreter is older than Python 3.10, the script reports the detected version and exits. Set `PYTHON_BIN` to select another Python executable, and use `VENV_DIR` to choose a different environment location.

## Usage ##

### 1 - Parse trajectory: `01_parse_traj.ipynb` ###

This notebook reads one or more trajectory files, filters and groups the molecule of interest and saves the information as a npz dataset file.

1. Specify the `INPUT_FILENAME` variable. It is the filename of the reference structure+topology. It should be a `.tpr` file.
2. Specify the `TRAJ_FILENAMES` variable. It is a list of filenames, containing all the trajectory files to load. They could be either `.trr` or `.xtc` files.
3. Specify the `SELECTION` variable. It is a string that should filter the atoms of the molecule of interest. E.g. `resname MOL`.
4. Optionally, specify `TRAJSLICE` variable to filter the frames of the trajectory to keep. It could be either `None` (keep all frames) or a string in the form `[start]:[stop]:[step]`. E.g. `:1000:2` filters the first 1000 frames, striding with a step of 2 and yielding a total of 500 frames.
5. Optionally, specify `OUTPUT_FILENAME`. It is the filename of the output `.npz` dataset. It could be either a `.npz` file or a folder. If a folder is specified, the dataset will be saved in that folder, using the `INPUT_FILENAME` stem as a filename and the `.npz` suffix. If not specified, the dataset will be names as the `INPUT_FILENAME` but with the `.npz` suffix.

### 2 - Build NMERS: `02_build_nmers.ipynb` ###

This notebook extracts all the nmers from the npz dataset and prepares the input files for Q-Chem.

1. Specify the `INPUT_FILENAME`. It corresponds to the `.npz` file saved in the previous step as `OUTPUT_FILENAME`.
2. Specify the `DATASET_ROOT`. It is the root folder for all the dataset components that will be created.
   ATTENTION! This folder should reside in a partition with enough disk space, and accessible to QChem software.

### 3 - Run Q-Chem calculations `dataforge-qchem`

This script runs Q-Chem QM single point energy calculations on all the nmers selected in the previous step.
To run qchem, follow these steps:

1. Load the qchem module: `source /path/to/qchem/.qcsetup`
2. Run the QChem single point evaluation on all nmers:
   `nohup dataforge-qchem -i DATASET_ROOT/data/qchem_input/ &`
3. Run the QChem energy minimization on each type of nmer:
   `nohup dataforge-qchem -i DATASET_ROOT/data/qchem_min_input/ &`

Optionally, you can run the qchem calculation on a subfolder only, e.g. with the following script:

`nohup dataforge-qchem -i DATASET_ROOT/data/qchem_input/trimers/dimers/monomers/C-CCHO/ &`

Note that you can run multiple instances of the script, they will run in parallel and distribute workload.

### 4 - Create MB-Fit dataset `03_build_dataset.ipynb`

This final script parses the QChem outputs and computes the energy contribution of each nmer, relative to the energy of the minimized system.
The script outputs the following files:

1. One `nmers.xyz` fitting dataset for each nmer. This can be found inside the `DATASET_ROOT/fitting/dataset/` folder.
2. One `nmers.opt` for each nmer. This can be found inside the `DATASET_ROOT/fitting/optimized/` folder.
3. One `poly_generator.py` for each nmer. This can be found inside the `DATASET_ROOT/fitting/poly/` folder.

### 5 - Guided n-mer workflow

Run the single interactive workflow from the repository root:

```bash
./scripts/qchem_workflow.sh
```

This is the only workflow shell script. After selecting the dataset root, it displays a numbered menu. Each invocation performs only the operation selected: trajectory parsing, monomer discovery/naming, XYZ sampling, initial capping, minimization preparation/execution, distance extraction, corrected capping, final QChem preparation/execution, or final dataset construction. After an operation finishes, the menu is displayed again. Orders above three use folders such as `4mers`, `5mers`, and so on.

Every operation that acts on n-mer types displays a numbered list. Press Enter for all types, or enter comma-separated numbers and ranges such as `3,4,6-8`. The chosen type names are passed to the underlying Python stage and recorded in its stage metadata. When building final energy contributions, include the lower-order monomers and multimers required by any selected higher-order n-mer.

XYZ sampling and initial capping are separate menu operations. For multi-gigabyte trajectories, n-mer construction automatically switches to one build process to avoid multiplying large coordinate arrays across workers. The QChem and capping stages can still use parallel workers.

Uniform sampling (`US`) is the workflow default. `FPS` uses a bounded two-stage implementation: descriptor-quantile preselection over the complete trajectory followed by furthest-point sampling on at most 50,000 candidates. It does not run DBSCAN over every trajectory frame. Descriptor arrays are computed and released one n-mer at a time, and serial builds likewise slice one sampled coordinate block at a time.

If sampling completed but initial capping was interrupted, choose operation 4 and select only the missing n-mer types. Calibrated C/N/O cap distances are retained; other supported elements use the sum of covalent radii as the initial minimization guess, and each capped HDF5 records the applied distances and their source.

The first dataset-root prompt defaults to the repository root. The selected absolute path is saved in `.dataforge_workflow_root` and becomes the default on subsequent calls, so later runs can resume the same dataset. The parsed trajectory defaults to `<dataset-root>/data/trajectory.npz`.

By default, `dataforge-build-nmers`/the workflow uses automatic monomer discovery. Each monomer starts as a heavy atom plus its bonded hydrogens; heavy atoms joined by an inferred or topology-provided double/triple bond are merged into one monomer. Operation 2 asks for a readable name for every `AUTO-*` type; pressing Enter retains the automatic name. Aliases are saved in `metadata/monomer_aliases.json`, reused on later runs, and applied before n-mer paths and metadata are generated. Discovery evidence and the connected n-mer candidate catalog are saved in `data/monomer_discovery.json`. Use `--monomer-mode legacy` only when reproducing a pre-existing configured composite-monomer definition, and provide `--bond-order-mode topology` or `geometry` when you want to force a particular bond-order source.

The corrected capped structures are written below `data/xyz_capped_minimized/`. The measured distances are recorded in `data/capping_distances.json`. Each HDF5 entry contains a `capping_atoms` record for every cap, including cap and bonded-heavy-atom indices/types, the original severed-atom index, the distance, and the explicit `angstrom` unit. Repeated subset runs merge their records into the same file.

The underlying non-interactive command is:

```bash
python -m dataforge.scripts.recap_nmers extract \
  --capped-root DATASET_ROOT/data/xyz_capped \
  --source-root DATASET_ROOT/data/xyz \
  --optimized-root DATASET_ROOT/fitting/optimized \
  --output DATASET_ROOT/data/capping_distances.json

python -m dataforge.scripts.recap_nmers apply \
  --source-root DATASET_ROOT/data/xyz \
  --destination-root DATASET_ROOT/data/xyz_capped_minimized \
  --fit-poly-root DATASET_ROOT/fitting/poly \
  --distances DATASET_ROOT/data/capping_distances.json
```

The initial minimization pass uses the element-based cap lengths to obtain minimized reference structures; only the subsequent full-dataset pass uses the measured minimized distances.

Minimization preparation writes one `jobtype opt` input per n-mer folder directly into `data/qchem_min_input`; it does not require the legacy `.list` files from `03_sample_dataset.ipynb`. The final preparation pass writes `jobtype sp` inputs for every frame already sampled into the corrected HDF5 files. Input preparation fails visibly if no capped HDF5 files or no selected frames are found.

Before launching workers, the workflow verifies that `qchem` is on `PATH`, `QCSCRATCH` is writable, and the Q-Chem executable has no unresolved shared-library dependencies. Worker failures preserve their output and are reported through each folder's `qchem-worker.log`; completed outputs are safely skipped on reruns.

Every guided stage writes a log under `metadata/logs/`, a stage manifest under `metadata/stages/`, and an append-only history in `metadata/pipeline.jsonl`. Manifests include parameters, input/output existence and sizes, hashes for small files, the Python executable, platform, and repository revision. The metadata command can also be used independently:

```bash
python -m dataforge.scripts.stage_metadata record \
  --root DATASET_ROOT --stage example \
  --inputs DATASET_ROOT/data/trajectory.npz \
  --outputs DATASET_ROOT/data/monomer_discovery.json \
  --parameters-json '{"mode": "auto"}'
```

## Configuration

Some scripts might require configuration.

## Contributing

If you'd like to contribute to this project, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature: `git checkout -b feature-name`.
3. Make your changes.
4. Test your changes.
5. Commit your changes: `git commit -m 'Added a new feature'`.
6. Push to the branch: `git push origin feature-name`.
7. Submit a pull request.

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).
