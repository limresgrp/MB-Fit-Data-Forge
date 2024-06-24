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