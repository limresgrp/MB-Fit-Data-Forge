#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$PROJECT_ROOT"
PYTHON=${PYTHON:-python3}
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

ask_default() {
    local prompt=$1 default=$2 answer
    read -r -p "$prompt [$default] " answer
    printf '%s' "${answer:-$default}"
}

ask_yes_no() {
    local prompt=$1 default=${2:-n} answer suffix='y/N'
    [[ "$default" == y ]] && suffix='Y/n'
    read -r -p "$prompt [$suffix] " answer
    [[ "$answer" == y || "$answer" == Y ]] || [[ -z "$answer" && "$default" == y ]]
}

run_logged() {
    local stage=$1 command_status
    shift
    mkdir -p "$DATASET_ROOT/metadata/logs"
    set +e
    "$@" 2>&1 | tee -a "$DATASET_ROOT/metadata/logs/${stage}.log"
    command_status=${PIPESTATUS[0]}
    set -e
    return "$command_status"
}

record_stage_metadata() {
    local stage=$1 parameters=$2 input_line=$3 output_line=$4 command=${5:-} status=${6:-completed}
    local -a inputs=() outputs=() metadata_args
    local old_ifs=$IFS
    IFS='|'; read -r -a inputs <<<"$input_line"; read -r -a outputs <<<"$output_line"; IFS=$old_ifs
    metadata_args=(--root "$DATASET_ROOT" --stage "$stage" --status "$status" --parameters-json "$parameters")
    [[ -n "$command" ]] && metadata_args+=(--command "$command")
    ((${#inputs[@]})) && metadata_args+=(--inputs "${inputs[@]}")
    ((${#outputs[@]})) && metadata_args+=(--outputs "${outputs[@]}")
    "$PYTHON" -m dataforge.scripts.stage_metadata record "${metadata_args[@]}" >/dev/null
}

json_parameters() {
    local -a pairs=("$@")
    PARAMETER_PAIRS=$(printf '%s\n' "${pairs[@]}") "$PYTHON" -c '
import json, os
values = {}
for pair in os.environ.get("PARAMETER_PAIRS", "").splitlines():
    key, value = pair.split("=", 1)
    values[key] = value
print(json.dumps(values))
'
}

run_qchem_parallel() {
    local input_root=$1 output_root=$2 workers=$3 label=$4
    local folder relative output_folder log_file
    local master_log="$DATASET_ROOT/metadata/logs/${label}.log"
    local -a folders=() pids=()
    local failed=0
    [[ "$workers" =~ ^[1-9][0-9]*$ ]] || { echo "QChem workers must be positive." >&2; return 2; }
    mapfile -t folders < <(find "$input_root" -type f -name '*.inp' -printf '%h\n' | sort -u)
    ((${#folders[@]})) || { echo "No QChem inputs found under $input_root" >&2; return 1; }
    mkdir -p "$(dirname "$master_log")"
    echo "Running $label QChem jobs in $workers parallel folder workers (${#folders[@]} folders)." | tee -a "$master_log"
    for folder in "${folders[@]}"; do
        relative=${folder#"$input_root"/}
        output_folder="$output_root/$relative"
        log_file="$output_folder/qchem-worker.log"
        mkdir -p "$output_folder"
        ("$PYTHON" -m dataforge.scripts.qchem --input "$folder" --output "$output_folder" >"$log_file" 2>&1) &
        pids+=("$!")
        if ((${#pids[@]} >= workers)); then
            wait "${pids[0]}" || failed=1
            pids=("${pids[@]:1}")
        fi
    done
    for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
    if ((failed)); then echo "$label QChem stage failed." | tee -a "$master_log"; else echo "$label QChem stage completed." | tee -a "$master_log"; fi
    return "$failed"
}

echo "General n-mer QChem workflow"
echo "This script guides trajectory parsing, sampling, minimization, re-capping, QChem, and dataset generation."

WORKFLOW_ROOT_FILE="$PROJECT_ROOT/.dataforge_workflow_root"
DEFAULT_DATASET_ROOT="$PROJECT_ROOT"
if [[ -s "$WORKFLOW_ROOT_FILE" ]]; then
    read -r SAVED_DATASET_ROOT <"$WORKFLOW_ROOT_FILE"
    [[ -n "${SAVED_DATASET_ROOT:-}" ]] && DEFAULT_DATASET_ROOT="$SAVED_DATASET_ROOT"
fi
DATASET_ROOT=$(ask_default "Dataset root" "$DEFAULT_DATASET_ROOT")
if [[ "$DATASET_ROOT" != /* ]]; then DATASET_ROOT="$PROJECT_ROOT/$DATASET_ROOT"; fi
mkdir -p "$DATASET_ROOT/data" "$DATASET_ROOT/fitting"
DATASET_ROOT=$(cd "$DATASET_ROOT" && pwd)
printf '%s\n' "$DATASET_ROOT" >"$WORKFLOW_ROOT_FILE"

TRAJ_DATASET=$(ask_default "Parsed trajectory .npz" "$DATASET_ROOT/data/trajectory.npz")
if [[ "$TRAJ_DATASET" != /* ]]; then TRAJ_DATASET="$DATASET_ROOT/$TRAJ_DATASET"; fi
PARSE_TRAJECTORY=false
if [[ -f "$TRAJ_DATASET" ]]; then
    if ask_yes_no "Recreate the parsed trajectory as stage 1?" n; then
        PARSE_TRAJECTORY=true
    fi
else
    if ask_yes_no "Create the parsed trajectory as stage 1?" y; then
        PARSE_TRAJECTORY=true
    fi
fi

if [[ "$PARSE_TRAJECTORY" == true ]]; then
    REFERENCE=$(ask_default "Reference topology/structure" "")
    TRAJ_LINE=$(ask_default "Trajectory file(s), space-separated" "")
    SELECTION=$(ask_default "Atom selection" "all")
    TRAJSLICE=$(ask_default "Trajectory slice" ":")
    read -r -a TRAJ_FILES <<<"$TRAJ_LINE"
    run_logged trajectory_parse "$PYTHON" -m dataforge.scripts.parse_traj \
        --input "$REFERENCE" --traj "${TRAJ_FILES[@]}" --selection "$SELECTION" \
        --trajslice "$TRAJSLICE" --output "$TRAJ_DATASET"
    record_stage_metadata trajectory_parse \
        "$(json_parameters selection="$SELECTION" trajslice="$TRAJSLICE")" \
        "$REFERENCE|${TRAJ_FILES[*]}" "$TRAJ_DATASET" "parse trajectory"
else
    test -f "$TRAJ_DATASET" || { echo "Missing parsed trajectory: $TRAJ_DATASET" >&2; exit 1; }
    record_stage_metadata trajectory_parse "{}" "$TRAJ_DATASET" "$TRAJ_DATASET" "reuse parsed trajectory" reused
fi

MONOMER_MODE=$(ask_default "Monomer discovery mode (auto or legacy)" auto)
BOND_ORDER_MODE=$(ask_default "Bond-order mode (auto, topology, or geometry)" auto)
ORDER=$(ask_default "N-mer orders to build" "1 2 3")
SAMPLE_COUNT=$(ask_default "Samples per requested n-mer name" 5000)
SAMPLE_METHOD=$(ask_default "Sampling method (US or bounded FPS)" US)
BUILD_WORKERS=$(ask_default "DataForge build worker processes" 4)
KEEP_NMER_LINE=$(ask_default "Optional exact n-mer names, space-separated (empty means all)" "")
KEEP_NMER_NAMES=()
if [[ -n "$KEEP_NMER_LINE" ]]; then read -r -a KEEP_NMER_NAMES <<<"$KEEP_NMER_LINE"; fi
SAMPLING_SPECS=()
for order in $ORDER; do SAMPLING_SPECS+=("${order}=${SAMPLE_COUNT}:${SAMPLE_METHOD}"); done

BUILD_NMERS=false
EXISTING_NMER_FILE=""
if [[ -d "$DATASET_ROOT/data/xyz" ]]; then
    EXISTING_NMER_FILE=$(find "$DATASET_ROOT/data/xyz" -type f -name '*.h5' -print -quit)
fi
if [[ -n "$EXISTING_NMER_FILE" ]]; then
    if ask_yes_no "Existing n-mer files found. Recompute them?" n; then
        BUILD_NMERS=true
    fi
else
    if ask_yes_no "Build and initially cap the sampled n-mers?" y; then
        BUILD_NMERS=true
    fi
fi

if [[ "$BUILD_NMERS" == true ]]; then
    BUILD_ARGS=(-m dataforge.scripts.build_nmers build --input "$TRAJ_DATASET" --root "$DATASET_ROOT" --sampling "${SAMPLING_SPECS[@]}" --monomer-mode "$MONOMER_MODE" --bond-order-mode "$BOND_ORDER_MODE" --max-processes "$BUILD_WORKERS")
    if ((${#KEEP_NMER_NAMES[@]})); then BUILD_ARGS+=(--keep-nmer-names "${KEEP_NMER_NAMES[@]}"); fi
    run_logged build_nmers "$PYTHON" "${BUILD_ARGS[@]}"
    record_stage_metadata build_nmers \
        "$(json_parameters orders="${ORDER// /,}" samples="$SAMPLE_COUNT" method="$SAMPLE_METHOD" monomer_mode="$MONOMER_MODE" bond_order_mode="$BOND_ORDER_MODE")" \
        "$TRAJ_DATASET" "$DATASET_ROOT/data/xyz|$DATASET_ROOT/data/xyz_capped|$DATASET_ROOT/data/monomer_discovery.json|$DATASET_ROOT/data/topology.json" "build n-mers"
else
    SOURCE_NMER_COUNT=0
    CAPPED_NMER_COUNT=0
    if [[ -d "$DATASET_ROOT/data/xyz" ]]; then
        SOURCE_NMER_COUNT=$(find "$DATASET_ROOT/data/xyz" -type f -name '*.h5' | wc -l)
    fi
    if [[ -d "$DATASET_ROOT/data/xyz_capped" ]]; then
        CAPPED_NMER_COUNT=$(find "$DATASET_ROOT/data/xyz_capped" -type f -name '*.h5' | wc -l)
    fi
    CAP_DEFAULT=n
    CAP_PROMPT="Re-cap existing n-mer files?"
    if ((CAPPED_NMER_COUNT < SOURCE_NMER_COUNT)); then
        CAP_DEFAULT=y
        CAP_PROMPT="Initial capped dataset is incomplete (${CAPPED_NMER_COUNT}/${SOURCE_NMER_COUNT}). Finish capping existing n-mers?"
    fi
    if ((SOURCE_NMER_COUNT > 0)) && ask_yes_no "$CAP_PROMPT" "$CAP_DEFAULT"; then
        run_logged cap_nmers "$PYTHON" -m dataforge.scripts.build_nmers cap --root "$DATASET_ROOT" --max-processes "$BUILD_WORKERS"
        record_stage_metadata cap_nmers \
            "$(json_parameters source_count="$SOURCE_NMER_COUNT" previous_capped_count="$CAPPED_NMER_COUNT")" \
            "$DATASET_ROOT/data/xyz" "$DATASET_ROOT/data/xyz_capped" "cap existing n-mers"
    fi
fi

INITIAL_CAPPED="$DATASET_ROOT/data/xyz_capped"
INITIAL_QCHEM_IN="$DATASET_ROOT/data/qchem_input_minimized"
INITIAL_QCHEM_MIN_IN="$DATASET_ROOT/data/qchem_min_input"
INITIAL_QCHEM_MIN_OUT="$DATASET_ROOT/data/qchem_min_output"
FIT_POLY="$DATASET_ROOT/fitting/poly"
OPTIMIZED="$DATASET_ROOT/fitting/optimized"
CHARGES_JSON=$(ask_default "Optional monomer-charge JSON (empty uses built-in charges)" "")
CHARGE_ARGS=()
if [[ -n "$CHARGES_JSON" ]]; then CHARGE_ARGS+=(--charges-json "$CHARGES_JSON"); fi

if ask_yes_no "Prepare QChem inputs for the minimization pass?"; then
    run_logged prepare_qchem_minimized "$PYTHON" -m dataforge.scripts.build_nmers prepare_qchem --root "$DATASET_ROOT" --nmers-capped-root "$INITIAL_CAPPED" --qchem-in-root "$INITIAL_QCHEM_IN" --qchem-min-in-root "$INITIAL_QCHEM_MIN_IN" --max-processes "$BUILD_WORKERS" "${CHARGE_ARGS[@]}"
    record_stage_metadata prepare_qchem_minimized "$(json_parameters charges_json="$CHARGES_JSON")" "$INITIAL_CAPPED" "$INITIAL_QCHEM_IN|$INITIAL_QCHEM_MIN_IN" "prepare minimization QChem inputs"
fi

if ask_yes_no "Load QChem and run minimized structures in parallel?"; then
    QC_SETUP=$(ask_default "QChem setup file (empty if already loaded)" "")
    if [[ -n "$QC_SETUP" ]]; then source "$QC_SETUP"; fi
    QC_WORKERS=$(ask_default "Parallel QChem folder workers" 8)
    run_qchem_parallel "$INITIAL_QCHEM_MIN_IN" "$INITIAL_QCHEM_MIN_OUT" "$QC_WORKERS" minimization
    record_stage_metadata qchem_minimization "$(json_parameters workers="$QC_WORKERS")" "$INITIAL_QCHEM_MIN_IN" "$INITIAL_QCHEM_MIN_OUT" "run minimized QChem calculations"
fi

DISTANCES="$DATASET_ROOT/data/capping_distances.json"
if ask_yes_no "Measure capping distances from minimized structures?"; then
    run_logged extract_capping_distances "$PYTHON" -m dataforge.scripts.recap_nmers extract --capped-root "$INITIAL_CAPPED" --source-root "$DATASET_ROOT/data/xyz" --optimized-root "$OPTIMIZED" --min-output-root "$INITIAL_QCHEM_MIN_OUT" --fit-poly-root "$FIT_POLY" --output "$DISTANCES"
    record_stage_metadata extract_capping_distances "$(json_parameters source=optimized_structures)" "$INITIAL_CAPPED|$INITIAL_QCHEM_MIN_OUT" "$OPTIMIZED|$DISTANCES" "extract minimized cap distances"
fi
test -f "$DISTANCES" || { echo "Missing capping-distance file: $DISTANCES" >&2; exit 1; }

CORRECTED_CAPPED="$DATASET_ROOT/data/xyz_capped_minimized"
if ask_yes_no "Apply measured distances and create corrected capped n-mers?"; then
    run_logged apply_capping_distances "$PYTHON" -m dataforge.scripts.recap_nmers apply --source-root "$DATASET_ROOT/data/xyz" --destination-root "$CORRECTED_CAPPED" --fit-poly-root "$FIT_POLY" --distances "$DISTANCES" --max-processes "$BUILD_WORKERS"
    record_stage_metadata apply_capping_distances "$(json_parameters distance_file="$DISTANCES")" "$DATASET_ROOT/data/xyz|$DISTANCES" "$CORRECTED_CAPPED" "apply minimized cap distances"
fi

FINAL_QCHEM_IN="$DATASET_ROOT/data/qchem_input"
FINAL_QCHEM_MIN_IN="$DATASET_ROOT/data/qchem_min_input_final"
FINAL_QCHEM_OUT="$DATASET_ROOT/data/qchem_output"
if ask_yes_no "Prepare QChem inputs for the complete corrected dataset?"; then
    run_logged prepare_qchem_final "$PYTHON" -m dataforge.scripts.build_nmers prepare_qchem --root "$DATASET_ROOT" --nmers-capped-root "$CORRECTED_CAPPED" --qchem-in-root "$FINAL_QCHEM_IN" --qchem-min-in-root "$FINAL_QCHEM_MIN_IN" --max-processes "$BUILD_WORKERS" "${CHARGE_ARGS[@]}"
    record_stage_metadata prepare_qchem_final "$(json_parameters charges_json="$CHARGES_JSON")" "$CORRECTED_CAPPED" "$FINAL_QCHEM_IN|$FINAL_QCHEM_MIN_IN" "prepare final QChem inputs"
fi

if ask_yes_no "Run whole-dataset QChem single points in parallel?"; then
    QC_WORKERS=$(ask_default "Parallel QChem folder workers" 8)
    run_qchem_parallel "$FINAL_QCHEM_IN" "$FINAL_QCHEM_OUT" "$QC_WORKERS" single-point
    record_stage_metadata qchem_single_point "$(json_parameters workers="$QC_WORKERS")" "$FINAL_QCHEM_IN" "$FINAL_QCHEM_OUT" "run final QChem calculations"
fi

if ask_yes_no "Compute n-mer contributions and build the final MB-Fit dataset?"; then
    run_logged build_dataset "$PYTHON" -m dataforge.scripts.build_dataset "$DATASET_ROOT" --NMERS_CAPPED_ROOT "$CORRECTED_CAPPED" --QCHEM_OUT_ROOT "$FINAL_QCHEM_OUT" --QCHEM_MIN_OUT_ROOT "$INITIAL_QCHEM_MIN_OUT" --FIT_OPTIM_ROOT "$OPTIMIZED" --FIT_POLY_ROOT "$FIT_POLY"
    record_stage_metadata build_dataset "$(json_parameters corrected_capped_root="$CORRECTED_CAPPED")" "$CORRECTED_CAPPED|$FINAL_QCHEM_OUT|$INITIAL_QCHEM_MIN_OUT" "$DATASET_ROOT/fitting" "build final MB-Fit dataset"
fi

echo "Workflow complete. Dataset root: $DATASET_ROOT"
