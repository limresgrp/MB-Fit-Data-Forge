#!/usr/bin/env python3

import os
import argparse
import textwrap
import threading
import subprocess
import glob

from os.path import dirname
from pathlib import Path

'''
# Procedure to perform single point energy evaluation and structure minimization #

source /apps/qchem6/.qcsetup
nohup dataforge-qchem -i [DATASET_ROOT]/data/qchem_input/ &
nohup dataforge-qchem -i [DATASET_ROOT]/data/qchem_min_input/ &
'''


def run_background(input: str, output: str):
    print(f"Input path: {os.path.join(input, "**/*.inp")}")
    print(f"Output path: {output}")
    for filename in glob.glob(os.path.join(input, "**/*.inp"), recursive=True):
        # If output file was already computed before, skip.
        out_filename = filename.replace(input, output).replace(".inp", ".out")
        if os.path.exists(out_filename):
            continue

        out_folder = dirname(out_filename)
        if not os.path.exists(out_folder):
            os.makedirs(out_folder, exist_ok=True)

        open(out_filename, 'a').close() # Create empty file to avoid other processes to process this same input
        print(f"qchem -nt 1 '{filename}' '{out_filename}'")
        subprocess.run([f"qchem -nt 1 '{filename}' '{out_filename}'"], shell=True, stdin=None, stdout=None, stderr=None, close_fds=True)

def check_outputs(root_dir):
    START_ROW = 250
    TARGET_STRING = "Thank you very much for using Q-Chem.  Have a nice day."

    def process_file(filepath, target_string):
        # Read the file and process the lines
        with open(filepath, 'r') as file:
            lines = file.readlines()

        if len(lines) == 0:
            print(f"File {filepath} is empty! Rerun QM calculations on it.")
            return

        new_lines = lines[:START_ROW]
        target_found = False
        skip_lines = 0

        for i, line in enumerate(lines[START_ROW:]):
            if target_found:
                if skip_lines > 0:
                    skip_lines -= 1
                else:
                    break  # Stop adding lines after skipping 2 lines
            new_lines.append(line)

            if target_string in line:
                target_found = True
                skip_lines = 2  # Skip the next 2 lines
        
        if not target_found:
            print(f"File {filepath} is either not completed or corrupted. Removing.")
            os.remove(filepath)
            return
        
        if len(lines[START_ROW:]) - i <= 5:
            return
        
        print(f"Fixing file {filepath}.")
        # Write the new content back to the file
        with open(filepath, 'w') as file:
            file.writelines(new_lines)

    def process_directory(root_dir, target_string):
        for dirpath, _, filenames in os.walk(root_dir):
            if len(filenames) > 0:
                print(f"Checking folder {dirpath}...")
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if filepath.endswith('.out'):
                    process_file(filepath, target_string)
    
    process_directory(root_dir, TARGET_STRING)

def main(args=None):
    parser = argparse.ArgumentParser(
        description=textwrap.dedent(
            """Compute the error of a model on a test set using various metrics.

            The model, metrics, dataset, etc. can specified in individual YAML config files, or a training session can be indicated with `--train-dir`.
            In order of priority, the global settings (dtype, TensorFloat32, etc.) are taken from:
              (1) the model config (for a training session),
              (2) the dataset config (for a deployed model),
              or (3) the defaults.

            Prints only the final result in `name = num` format to stdout; all other information is `logging.debug`ed to stderr.

            WARNING: Please note that results of CUDA models are rarely exactly reproducible, and that even CPU models can be nondeterministic.
            """
        )
    )

    parser.add_argument(
        "-i",
        "--input",
        help="Path to the qchem_input directory.",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Path to the qchem_output directory. Provide only if you know what you are doing.",
        type=Path,
        default=None,
    )

    args = parser.parse_args(args=args)
    _input = args.input
    _output = args.output
    if _output is None:
        _output = str(_input).replace("qchem_input", "qchem_output")

    bg_thread = threading.Thread(target=run_background, args=(str(_input), str(_output),))
    bg_thread.start()
    bg_thread.join()  # Wait for the background thread to finish

    check_outputs(_output)

if __name__ == "__main__":
    main()