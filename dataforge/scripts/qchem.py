#!/usr/bin/env python3

"""Run Q-Chem inputs recursively and fail if any calculation is incomplete."""

import argparse
import glob
import os
import shutil
import subprocess
from pathlib import Path


SUCCESS_MARKER = "Thank you very much for using Q-Chem.  Have a nice day."


def output_is_complete(filepath: str) -> bool:
    if not os.path.isfile(filepath) or os.path.getsize(filepath) == 0:
        return False
    with open(filepath, "r", errors="replace") as output_file:
        return any(SUCCESS_MARKER in line for line in output_file)


def run_qchem_inputs(input_root: str, output_root: str, threads: int = 1):
    qchem_executable = shutil.which("qchem")
    if qchem_executable is None:
        raise RuntimeError(
            "qchem is not available on PATH. Source the Q-Chem setup file before running."
        )

    input_pattern = os.path.join(input_root, "**", "*.inp")
    input_files = sorted(glob.glob(input_pattern, recursive=True))
    if not input_files:
        raise FileNotFoundError(f"No Q-Chem input files found under {input_root}")

    print(f"Input path: {input_pattern}", flush=True)
    print(f"Output path: {output_root}", flush=True)
    failures = []
    for input_filename in input_files:
        relative = os.path.relpath(input_filename, input_root)
        output_filename = os.path.join(output_root, os.path.splitext(relative)[0] + ".out")
        if output_is_complete(output_filename):
            print(f"Complete output exists; skipping {input_filename}", flush=True)
            continue

        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        if os.path.exists(output_filename):
            os.remove(output_filename)
        command = [
            qchem_executable,
            "-nt",
            str(threads),
            input_filename,
            output_filename,
        ]
        print("Running: " + " ".join(command), flush=True)
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0 or not output_is_complete(output_filename):
            failures.append((input_filename, output_filename, completed.returncode))
            print(
                f"FAILED (exit {completed.returncode}): {input_filename}; "
                f"inspect {output_filename}",
                flush=True,
            )

    return failures


def main(args=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--input", type=Path, required=True)
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument("-t", "--threads", type=int, default=1)
    parsed = parser.parse_args(args=args)

    if parsed.threads < 1:
        parser.error("--threads must be positive")
    input_root = str(parsed.input)
    output_root = str(parsed.output) if parsed.output is not None else input_root.replace(
        "qchem_input", "qchem_output"
    )
    failures = run_qchem_inputs(input_root, output_root, threads=parsed.threads)
    if failures:
        raise SystemExit(f"{len(failures)} Q-Chem calculation(s) failed or were incomplete.")


if __name__ == "__main__":
    main()
