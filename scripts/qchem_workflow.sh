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

require_file() { [[ -f "$1" ]] || { echo "Missing file: $1" >&2; return 1; }; }
require_dir() { [[ -d "$1" ]] || { echo "Missing directory: $1" >&2; return 1; }; }

selection_summary() {
    if [[ "$SELECT_ALL" == true ]]; then
        printf 'all'
    else
        local IFS=,
        printf '%s' "${SELECTED_NMER_NAMES[*]}"
    fi
}

choose_numbered_entries() {
    local prompt=$1 answer token start end index entry
    shift
    local -a entries=("$@")
    local -A chosen=()
    SELECTED_NMER_NAMES=()
    SELECT_ALL=true
    ((${#entries[@]})) || { echo "No n-mer types are available." >&2; return 1; }
    echo "$prompt"
    for index in "${!entries[@]}"; do printf '  %3d) %s\n' "$((index + 1))" "${entries[$index]}"; done
    read -r -p "Select all, or enter numbers/ranges (example: 3,4,6-8) [all] " answer
    answer=${answer:-all}
    [[ "${answer,,}" == all || "$answer" == '*' ]] && return 0
    answer=${answer//,/ }
    for token in $answer; do
        if [[ "$token" =~ ^([0-9]+)-([0-9]+)$ ]]; then
            start=${BASH_REMATCH[1]}; end=${BASH_REMATCH[2]}
            ((start <= end)) || { echo "Invalid range: $token" >&2; return 1; }
            for ((index=start; index<=end; index++)); do chosen[$index]=1; done
        elif [[ "$token" =~ ^[0-9]+$ ]]; then
            chosen[$token]=1
        else
            echo "Invalid selection token: $token" >&2; return 1
        fi
    done
    for index in $(printf '%s\n' "${!chosen[@]}" | sort -n); do
        ((index >= 1 && index <= ${#entries[@]})) || { echo "Selection $index is out of range." >&2; return 1; }
        entry=${entries[$((index - 1))]}
        SELECTED_NMER_NAMES+=("${entry#*] }")
    done
    ((${#SELECTED_NMER_NAMES[@]})) || { echo "Select at least one n-mer type." >&2; return 1; }
    SELECT_ALL=false
}

select_nmers_from_root() {
    local root=$1 extension=$2 prompt=$3
    local -a entries=()
    require_dir "$root" || return 1
    mapfile -t entries < <("$PYTHON" - "$root" "$extension" <<'PY'
import glob, os, re, sys
root, extension = sys.argv[1:]
orders = {}
for path in glob.iglob(os.path.join(root, "**", f"*.{extension}"), recursive=True):
    folder = os.path.dirname(path)
    name = os.path.basename(folder)
    labels = [p for p in os.path.relpath(folder, root).split(os.sep)
              if re.fullmatch(r"monomers|dimers|trimers|[0-9]+mers", p)]
    orders.setdefault(name, set()).update(labels[-1:] or ["n-mer"])
for name in sorted(orders):
    print(f"[{','.join(sorted(orders[name]))}] {name}")
PY
    )
    choose_numbered_entries "$prompt" "${entries[@]}"
}

select_nmers_from_catalog() {
    local catalog=$1 prompt=$2
    local -a entries=()
    require_file "$catalog" || return 1
    mapfile -t entries < <("$PYTHON" - "$catalog" <<'PY'
import json, sys
with open(sys.argv[1]) as stream:
    data = json.load(stream)
for order, names in sorted(data.get("candidate_nmers", {}).items(), key=lambda item: int(item[0])):
    for name in names:
        print(f"[{order}-mer] {name}")
PY
    )
    choose_numbered_entries "$prompt" "${entries[@]}"
}

nmer_cli_args() {
    NMER_ARGS=()
    [[ "$SELECT_ALL" == true ]] || NMER_ARGS=(--nmer-names "${SELECTED_NMER_NAMES[@]}")
}

estimate_sample_target() {
    "$PYTHON" - "$1" <<'PY'
import collections, glob, os, sys
import h5py
totals = collections.Counter()
for path in glob.iglob(os.path.join(sys.argv[1], "*", "*", "*.h5")):
    try:
        with h5py.File(path, "r") as handle:
            totals[os.path.dirname(path)] += int(handle["coords"].shape[0])
    except (OSError, KeyError):
        pass
print(max(totals.values(), default=0))
PY
}

validate_qchem_environment() {
    local qchem_executable missing_libraries qc_program
    qchem_executable=$(command -v qchem || true)
    [[ -n "$qchem_executable" ]] || { echo "Q-Chem executable not found on PATH." >&2; return 1; }
    qc_program="${QC:-}/exe/qcprog.exe"
    if [[ -n "${QC:-}" && -x "$qc_program" ]] && command -v ldd >/dev/null; then
        missing_libraries=$(ldd "$qc_program" 2>/dev/null | awk '/not found/{print $1}')
        [[ -z "$missing_libraries" ]] || { echo "Q-Chem is missing shared libraries:" >&2; printf '  %s\n' $missing_libraries >&2; return 1; }
    fi
    [[ -z "${QCSCRATCH:-}" || -d "$QCSCRATCH" ]] || mkdir -p "$QCSCRATCH"
    [[ -z "${QCSCRATCH:-}" || -w "$QCSCRATCH" ]] || { echo "QCSCRATCH is not writable: $QCSCRATCH" >&2; return 1; }
    echo "Q-Chem environment validated: $qchem_executable"
}

load_qchem() {
    local setup
    setup=$(ask_default "QChem setup file (empty if already loaded)" "/apps/qchem6/.qcsetup")
    [[ -z "$setup" ]] || source "$setup"
    validate_qchem_environment
}

run_qchem_parallel() {
    local input_root=$1 output_root=$2 workers=$3 label=$4
    shift 4
    local folder relative output_folder log_file folder_name i
    local master_log="$DATASET_ROOT/metadata/logs/${label}.log"
    local -a requested_names=("$@") folders=() pids=() pid_logs=() failed_logs=()
    local failed=0
    local -A selected=()
    for folder_name in "${requested_names[@]}"; do selected["$folder_name"]=1; done
    [[ "$workers" =~ ^[1-9][0-9]*$ ]] || { echo "QChem workers must be positive." >&2; return 2; }
    while IFS= read -r folder; do
        folder_name=$(basename "$folder")
        ((${#requested_names[@]} == 0)) || [[ -n "${selected[$folder_name]:-}" ]] || continue
        folders+=("$folder")
    done < <(find "$input_root" -type f -name '*.inp' -printf '%h\n' | sort -u)
    ((${#folders[@]})) || { echo "No selected QChem inputs found under $input_root" >&2; return 1; }
    mkdir -p "$(dirname "$master_log")"
    echo "Running $label QChem jobs in $workers parallel folder workers (${#folders[@]} folders)." | tee -a "$master_log"
    for folder in "${folders[@]}"; do
        relative=${folder#"$input_root"/}; output_folder="$output_root/$relative"; log_file="$output_folder/qchem-worker.log"
        mkdir -p "$output_folder"
        ("$PYTHON" -m dataforge.scripts.qchem --input "$folder" --output "$output_folder" >"$log_file" 2>&1) &
        pids+=("$!"); pid_logs+=("$log_file")
        if ((${#pids[@]} >= workers)); then
            if ! wait "${pids[0]}"; then failed=1; failed_logs+=("${pid_logs[0]}"); fi
            pids=("${pids[@]:1}"); pid_logs=("${pid_logs[@]:1}")
        fi
    done
    for i in "${!pids[@]}"; do
        if ! wait "${pids[$i]}"; then failed=1; failed_logs+=("${pid_logs[$i]}"); fi
    done
    if ((failed)); then
        echo "$label QChem stage failed. Failed worker logs:" | tee -a "$master_log"
        printf '  %s\n' "${failed_logs[@]}" | tee -a "$master_log"
    else
        echo "$label QChem stage completed." | tee -a "$master_log"
    fi
    return "$failed"
}

stage_parse_trajectory() {
    local reference traj_line selection trajslice
    local -a traj_files
    if [[ -f "$TRAJ_DATASET" ]] && ! ask_yes_no "Parsed trajectory exists. Recreate it?" n; then echo "Keeping $TRAJ_DATASET"; return 0; fi
    reference=$(ask_default "Reference topology/structure" "")
    traj_line=$(ask_default "Trajectory file(s), space-separated" "")
    selection=$(ask_default "Atom selection" "all")
    trajslice=$(ask_default "Trajectory slice" ":")
    read -r -a traj_files <<<"$traj_line"
    run_logged trajectory_parse "$PYTHON" -m dataforge.scripts.parse_traj --input "$reference" --traj "${traj_files[@]}" --selection "$selection" --trajslice "$trajslice" --output "$TRAJ_DATASET"
    record_stage_metadata trajectory_parse "$(json_parameters selection="$selection" trajslice="$trajslice")" "$reference|${traj_files[*]}" "$TRAJ_DATASET" "parse trajectory"
}

stage_discover_monomers() {
    local monomer_mode bond_order_mode max_order discovery aliases_file merges_file charges_file raw_name current_name requested_name merge_name inferred_charge current_charge answer
    local -a raw_auto_names=() discovery_args=() MERGE_ENTRIES=() CHARGE_ENTRIES=()
    require_file "$TRAJ_DATASET" || return 1
    monomer_mode=$(ask_default "Monomer discovery mode (auto or legacy)" auto)
    bond_order_mode=$(ask_default "Bond-order mode (auto, topology, or geometry)" auto)
    max_order=$(ask_default "Largest n-mer order to catalog" 3)
    discovery="$DATASET_ROOT/data/monomer_discovery.json"
    aliases_file="$DATASET_ROOT/metadata/monomer_aliases.json"
    merges_file="$DATASET_ROOT/metadata/monomer_merges.json"
    charges_file="$DATASET_ROOT/metadata/monomer_charges.json"
    mkdir -p "$DATASET_ROOT/metadata"
    run_logged discover_monomers "$PYTHON" -m dataforge.scripts.build_nmers discover --input "$TRAJ_DATASET" --root "$DATASET_ROOT" --monomer-mode "$monomer_mode" --bond-order-mode "$bond_order_mode" --max-order "$max_order"
    if [[ "$monomer_mode" == auto ]]; then
        mapfile -t raw_auto_names < <("$PYTHON" - "$discovery" <<'PY'
import json, sys
with open(sys.argv[1]) as stream: data = json.load(stream)
print("\n".join(sorted({n for n in data.get("automatic_monomer_names", []) if n.startswith("AUTO-")})))
PY
        )
        ALIAS_PAIRS=""
        for raw_name in "${raw_auto_names[@]}"; do
            current_name=$("$PYTHON" - "$aliases_file" "$raw_name" <<'PY'
import json, os, sys
path, name = sys.argv[1:]
data = json.load(open(path)) if os.path.isfile(path) else {}
print(data.get(name, name))
PY
            )
            while true; do
                requested_name=$(ask_default "Name for automatic monomer '$raw_name' (keep AUTO name if unchanged)" "$current_name")
                if [[ ! "$requested_name" =~ ^[A-Za-z0-9_+-]+$ ]]; then echo "Use only letters, numbers, '_', '+', and '-' in monomer names." >&2; continue; fi
                break
            done
            ALIAS_PAIRS+="$raw_name"$'\t'"$requested_name"$'\n'
        done
        ALIAS_PAIRS="$ALIAS_PAIRS" "$PYTHON" - "$aliases_file" <<'PY'
import json, os, sys
path = sys.argv[1]
aliases = json.load(open(path)) if os.path.isfile(path) else {}
for line in os.environ.get("ALIAS_PAIRS", "").splitlines():
    source, alias = line.split("\t", 1); aliases[source] = alias
if len(set(aliases.values())) != len(aliases): raise SystemExit("Two AUTO monomers cannot use the same name.")
os.makedirs(os.path.dirname(path), exist_ok=True)
with open(path, "w") as stream: json.dump(aliases, stream, indent=2, sort_keys=True); stream.write("\n")
PY
        if [[ -f "$merges_file" ]] && ! ask_yes_no "Keep existing connected-monomer merge definitions?" y; then
            "$PYTHON" - "$merges_file" <<'PY'
import json, sys
with open(sys.argv[1], "w") as stream:
    json.dump({"version": 1, "merges": []}, stream, indent=2)
    stream.write("\n")
PY
        fi
        [[ -f "$merges_file" ]] || "$PYTHON" - "$merges_file" <<'PY'
import json, sys
with open(sys.argv[1], "w") as stream:
    json.dump({"version": 1, "merges": []}, stream, indent=2)
    stream.write("\n")
PY
        discovery_args=(--monomer-aliases-json "$aliases_file" --monomer-merges-json "$merges_file")
        run_logged discover_monomers "$PYTHON" -m dataforge.scripts.build_nmers discover --input "$TRAJ_DATASET" --root "$DATASET_ROOT" --monomer-mode "$monomer_mode" --bond-order-mode "$bond_order_mode" --max-order "$max_order" "${discovery_args[@]}"

        while ask_yes_no "Merge connected inferred monomer types into a larger monomer?" n; do
            mapfile -t MERGE_ENTRIES < <("$PYTHON" - "$discovery" <<'PY'
import json, sys
with open(sys.argv[1]) as stream: data = json.load(stream)
charges = data.get("inferred_charges", {})
for name in data.get("candidate_nmers", {}).get("1", []):
    print(f"[suggested charge {charges.get(name, 'review')}] {name}")
PY
            )
            choose_numbered_entries "Choose at least two connected monomer types to merge:" "${MERGE_ENTRIES[@]}" || return 1
            if [[ "$SELECT_ALL" == true ]]; then
                SELECTED_NMER_NAMES=()
                for answer in "${MERGE_ENTRIES[@]}"; do SELECTED_NMER_NAMES+=("${answer#*] }"); done
            fi
            ((${#SELECTED_NMER_NAMES[@]} >= 2)) || { echo "A merge requires at least two distinct monomer types." >&2; continue; }
            while true; do
                merge_name=$(ask_default "Name for the merged monomer" "")
                [[ "$merge_name" =~ ^[A-Za-z0-9_+-]+$ ]] && break
                echo "Use only letters, numbers, '_', '+', and '-' in monomer names." >&2
            done
            MERGE_MEMBERS=$(printf '%s\n' "${SELECTED_NMER_NAMES[@]}") MERGE_NAME="$merge_name" "$PYTHON" - "$merges_file" <<'PY'
import json, os, sys
path = sys.argv[1]
data = json.load(open(path))
members = list(dict.fromkeys(os.environ["MERGE_MEMBERS"].splitlines()))
name = os.environ["MERGE_NAME"]
if any(entry["name"] == name for entry in data.get("merges", [])):
    raise SystemExit(f"A merge named {name!r} already exists.")
data.setdefault("merges", []).append({"name": name, "members": members})
with open(path, "w") as stream:
    json.dump(data, stream, indent=2)
    stream.write("\n")
PY
            run_logged discover_monomers "$PYTHON" -m dataforge.scripts.build_nmers discover --input "$TRAJ_DATASET" --root "$DATASET_ROOT" --monomer-mode "$monomer_mode" --bond-order-mode "$bond_order_mode" --max-order "$max_order" "${discovery_args[@]}"
        done

        mapfile -t CHARGE_ENTRIES < <("$PYTHON" - "$discovery" "$charges_file" <<'PY'
import json, os, sys
with open(sys.argv[1]) as stream: data = json.load(stream)
confirmed = json.load(open(sys.argv[2])) if os.path.isfile(sys.argv[2]) else {}
inferred = data.get("inferred_charges", {})
for name in data.get("candidate_nmers", {}).get("1", []):
    suggestion = inferred.get(name, 0)
    current = confirmed.get(name, suggestion)
    print(f"{name}\t{suggestion}\t{current}")
PY
        )
        echo "Suggested monomer charges from full-topology bond-order sums:"
        for answer in "${CHARGE_ENTRIES[@]}"; do
            IFS=$'\t' read -r raw_name inferred_charge current_charge <<<"$answer"
            printf '  %-45s suggested=%s  selected=%s\n' "$raw_name" "$inferred_charge" "$current_charge"
        done
        CHARGE_PAIRS=""
        if ask_yes_no "Accept the selected charges shown above?" y; then
            for answer in "${CHARGE_ENTRIES[@]}"; do
                IFS=$'\t' read -r raw_name inferred_charge current_charge <<<"$answer"
                CHARGE_PAIRS+="$raw_name"$'\t'"$current_charge"$'\n'
            done
        else
            for answer in "${CHARGE_ENTRIES[@]}"; do
                IFS=$'\t' read -r raw_name inferred_charge current_charge <<<"$answer"
                while true; do
                    requested_name=$(ask_default "Integer charge for '$raw_name' (auto suggestion: $inferred_charge)" "$current_charge")
                    [[ "$requested_name" =~ ^-?[0-9]+$ ]] && break
                    echo "Enter an integer charge." >&2
                done
                CHARGE_PAIRS+="$raw_name"$'\t'"$requested_name"$'\n'
            done
        fi
        CHARGE_PAIRS="$CHARGE_PAIRS" "$PYTHON" - "$charges_file" <<'PY'
import json, os, sys
charges = {}
for line in os.environ.get("CHARGE_PAIRS", "").splitlines():
    name, charge = line.split("\t", 1)
    charges[name] = int(charge)
with open(sys.argv[1], "w") as stream:
    json.dump(charges, stream, indent=2, sort_keys=True)
    stream.write("\n")
PY
    fi
    record_stage_metadata discover_monomers "$(json_parameters monomer_mode="$monomer_mode" bond_order_mode="$bond_order_mode" max_order="$max_order")" "$TRAJ_DATASET" "$discovery|$aliases_file|$merges_file|$charges_file" "discover, merge, name, and assign charges to monomers"
}

stage_build_xyz() {
    local order_line sample_count sample_default sample_method workers max_order monomer_mode bond_order_mode aliases_file merges_file discovery current_target selection order missing_aliases
    local -a sampling_specs=() build_args=()
    require_file "$TRAJ_DATASET" || return 1
    order_line=$(ask_default "N-mer orders to build" "1 2 3"); max_order=$(tr ' ' '\n' <<<"$order_line" | sort -n | tail -1)
    monomer_mode=$(ask_default "Monomer discovery mode (auto or legacy)" auto); bond_order_mode=$(ask_default "Bond-order mode (auto, topology, or geometry)" auto)
    aliases_file="$DATASET_ROOT/metadata/monomer_aliases.json"; merges_file="$DATASET_ROOT/metadata/monomer_merges.json"; discovery="$DATASET_ROOT/data/monomer_discovery.json"
    build_args=(--monomer-mode "$monomer_mode" --bond-order-mode "$bond_order_mode")
    [[ ! -f "$aliases_file" || "$monomer_mode" != auto ]] || build_args+=(--monomer-aliases-json "$aliases_file")
    [[ ! -f "$merges_file" || ! -f "$aliases_file" || "$monomer_mode" != auto ]] || build_args+=(--monomer-merges-json "$merges_file")
    run_logged discover_monomers "$PYTHON" -m dataforge.scripts.build_nmers discover --input "$TRAJ_DATASET" --root "$DATASET_ROOT" --max-order "$max_order" "${build_args[@]}"
    if [[ "$monomer_mode" == auto ]]; then
        missing_aliases=$("$PYTHON" - "$discovery" "$aliases_file" <<'PY'
import json, os, sys
discovery, aliases_path = sys.argv[1:]
data = json.load(open(discovery))
aliases = json.load(open(aliases_path)) if os.path.isfile(aliases_path) else {}
automatic = {name for name in data.get("automatic_monomer_names", []) if name.startswith("AUTO-")}
print(len(automatic - set(aliases)))
PY
        )
        if ((missing_aliases > 0)); then
            echo "$missing_aliases AUTO monomer type(s) still need a naming decision."
            stage_discover_monomers || return 1
            require_file "$aliases_file" || { echo "Automatic builds require naming decisions from operation 2." >&2; return 1; }
            build_args=(--monomer-mode auto --bond-order-mode "$bond_order_mode" --monomer-aliases-json "$aliases_file")
            [[ ! -f "$merges_file" ]] || build_args+=(--monomer-merges-json "$merges_file")
            run_logged discover_monomers "$PYTHON" -m dataforge.scripts.build_nmers discover --input "$TRAJ_DATASET" --root "$DATASET_ROOT" --max-order "$max_order" "${build_args[@]}"
        fi
    fi
    select_nmers_from_catalog "$discovery" "Available connected n-mer types:" || return 1
    selection=$(selection_summary)
    [[ "$SELECT_ALL" == true ]] || build_args+=(--keep-nmer-names "${SELECTED_NMER_NAMES[@]}")
    current_target=$(estimate_sample_target "$XYZ_ROOT"); sample_default=5000; ((current_target == 0)) || sample_default=$((current_target * 2))
    sample_count=$(ask_default "Target samples per selected n-mer type" "$sample_default"); sample_method=$(ask_default "Sampling method (US or bounded FPS)" US); workers=$(ask_default "DataForge worker processes" 4)
    for order in $order_line; do sampling_specs+=("${order}=${sample_count}:${sample_method}"); done
    run_logged build_nmers "$PYTHON" -m dataforge.scripts.build_nmers build --input "$TRAJ_DATASET" --root "$DATASET_ROOT" --sampling "${sampling_specs[@]}" --max-processes "$workers" --skip-cap "${build_args[@]}"
    record_stage_metadata build_nmers "$(json_parameters orders="${order_line// /,}" samples="$sample_count" method="$sample_method" selection="$selection")" "$TRAJ_DATASET|$discovery" "$XYZ_ROOT|$DATASET_ROOT/data/topology.json" "sample selected XYZ n-mers"
}

stage_initial_cap() {
    local workers selection
    select_nmers_from_root "$XYZ_ROOT" h5 "Sampled n-mer types available for initial capping:" || return 1; nmer_cli_args
    workers=$(ask_default "Capping worker processes" 4); selection=$(selection_summary)
    run_logged cap_nmers "$PYTHON" -m dataforge.scripts.build_nmers cap --root "$DATASET_ROOT" --max-processes "$workers" "${NMER_ARGS[@]}"
    record_stage_metadata cap_nmers "$(json_parameters workers="$workers" selection="$selection")" "$XYZ_ROOT" "$INITIAL_CAPPED" "initially cap selected n-mers"
}

charge_args() {
    local charges_json default_charges=""
    CHARGE_ARGS=()
    [[ ! -f "$DATASET_ROOT/metadata/monomer_charges.json" ]] || default_charges="$DATASET_ROOT/metadata/monomer_charges.json"
    charges_json=$(ask_default "Monomer-charge JSON ('builtin' uses package defaults)" "$default_charges")
    [[ "$charges_json" == builtin || -z "$charges_json" ]] || {
        require_file "$charges_json" || return 1
        CHARGE_ARGS=(--charges-json "$charges_json")
    }
}

stage_prepare_minimization() {
    local workers selection
    select_nmers_from_root "$INITIAL_CAPPED" h5 "Capped n-mer types available for minimization input preparation:" || return 1; nmer_cli_args; charge_args || return 1
    workers=$(ask_default "Input-preparation worker processes" 4); selection=$(selection_summary)
    run_logged prepare_qchem_minimized "$PYTHON" -m dataforge.scripts.build_nmers prepare_qchem --root "$DATASET_ROOT" --nmers-capped-root "$INITIAL_CAPPED" --qchem-min-in-root "$INITIAL_QCHEM_MIN_IN" --qchem-mode minimization --max-processes "$workers" "${CHARGE_ARGS[@]}" "${NMER_ARGS[@]}"
    record_stage_metadata prepare_qchem_minimized "$(json_parameters workers="$workers" selection="$selection")" "$INITIAL_CAPPED" "$INITIAL_QCHEM_MIN_IN" "prepare selected minimization inputs"
}

stage_run_minimization() {
    local workers selection
    select_nmers_from_root "$INITIAL_QCHEM_MIN_IN" inp "N-mer types with minimization inputs:" || return 1; load_qchem || return 1
    workers=$(ask_default "Parallel QChem folder workers" 8); selection=$(selection_summary)
    run_qchem_parallel "$INITIAL_QCHEM_MIN_IN" "$INITIAL_QCHEM_MIN_OUT" "$workers" minimization "${SELECTED_NMER_NAMES[@]}"
    record_stage_metadata qchem_minimization "$(json_parameters workers="$workers" selection="$selection")" "$INITIAL_QCHEM_MIN_IN" "$INITIAL_QCHEM_MIN_OUT" "run selected minimizations"
}

stage_extract_distances() {
    local selection
    select_nmers_from_root "$INITIAL_CAPPED" h5 "N-mer types whose minimized cap distances will be measured:" || return 1; nmer_cli_args; selection=$(selection_summary)
    run_logged extract_capping_distances "$PYTHON" -m dataforge.scripts.recap_nmers extract --capped-root "$INITIAL_CAPPED" --source-root "$XYZ_ROOT" --optimized-root "$OPTIMIZED" --min-output-root "$INITIAL_QCHEM_MIN_OUT" --fit-poly-root "$FIT_POLY" --output "$DISTANCES" "${NMER_ARGS[@]}"
    record_stage_metadata extract_capping_distances "$(json_parameters selection="$selection" distance_unit=angstrom per_atom_metadata=true)" "$INITIAL_CAPPED|$INITIAL_QCHEM_MIN_OUT" "$OPTIMIZED|$DISTANCES" "measure selected per-atom cap distances"
}

stage_apply_distances() {
    local workers selection
    require_file "$DISTANCES" || return 1; select_nmers_from_root "$XYZ_ROOT" h5 "N-mer types to re-cap with minimized distances:" || return 1; nmer_cli_args
    workers=$(ask_default "Re-capping worker processes" 4); selection=$(selection_summary)
    run_logged apply_capping_distances "$PYTHON" -m dataforge.scripts.recap_nmers apply --source-root "$XYZ_ROOT" --destination-root "$CORRECTED_CAPPED" --fit-poly-root "$FIT_POLY" --distances "$DISTANCES" --max-processes "$workers" "${NMER_ARGS[@]}"
    record_stage_metadata apply_capping_distances "$(json_parameters workers="$workers" selection="$selection")" "$XYZ_ROOT|$DISTANCES" "$CORRECTED_CAPPED" "re-cap selected n-mers"
}

stage_prepare_final() {
    local workers selection
    select_nmers_from_root "$CORRECTED_CAPPED" h5 "Corrected n-mer types available for single-point input preparation:" || return 1; nmer_cli_args; charge_args || return 1
    workers=$(ask_default "Input-preparation worker processes" 4); selection=$(selection_summary)
    run_logged prepare_qchem_final "$PYTHON" -m dataforge.scripts.build_nmers prepare_qchem --root "$DATASET_ROOT" --nmers-capped-root "$CORRECTED_CAPPED" --qchem-in-root "$FINAL_QCHEM_IN" --qchem-mode full --max-processes "$workers" "${CHARGE_ARGS[@]}" "${NMER_ARGS[@]}"
    record_stage_metadata prepare_qchem_final "$(json_parameters workers="$workers" selection="$selection")" "$CORRECTED_CAPPED" "$FINAL_QCHEM_IN" "prepare selected single-point inputs"
}

stage_run_final() {
    local workers selection
    select_nmers_from_root "$FINAL_QCHEM_IN" inp "N-mer types with single-point inputs:" || return 1; load_qchem || return 1
    workers=$(ask_default "Parallel QChem folder workers" 8); selection=$(selection_summary)
    run_qchem_parallel "$FINAL_QCHEM_IN" "$FINAL_QCHEM_OUT" "$workers" single-point "${SELECTED_NMER_NAMES[@]}"
    record_stage_metadata qchem_single_point "$(json_parameters workers="$workers" selection="$selection")" "$FINAL_QCHEM_IN" "$FINAL_QCHEM_OUT" "run selected single points"
}

stage_build_dataset() {
    local selection
    select_nmers_from_root "$CORRECTED_CAPPED" h5 "N-mer types to include in final energy contributions/datasets:" || return 1
    echo "Include every lower-order n-mer needed by selected higher-order contributions."
    nmer_cli_args; selection=$(selection_summary)
    run_logged build_dataset "$PYTHON" -m dataforge.scripts.build_dataset "$DATASET_ROOT" --NMERS_CAPPED_ROOT "$CORRECTED_CAPPED" --QCHEM_OUT_ROOT "$FINAL_QCHEM_OUT" --QCHEM_MIN_OUT_ROOT "$INITIAL_QCHEM_MIN_OUT" --FIT_OPTIM_ROOT "$OPTIMIZED" --FIT_POLY_ROOT "$FIT_POLY" "${NMER_ARGS[@]}"
    record_stage_metadata build_dataset "$(json_parameters selection="$selection")" "$CORRECTED_CAPPED|$FINAL_QCHEM_OUT|$INITIAL_QCHEM_MIN_OUT" "$DATASET_ROOT/fitting" "build selected final datasets"
}

echo "General n-mer QChem workflow"
WORKFLOW_ROOT_FILE="$PROJECT_ROOT/.dataforge_workflow_root"; DEFAULT_DATASET_ROOT="$PROJECT_ROOT"
if [[ -s "$WORKFLOW_ROOT_FILE" ]]; then read -r saved_root <"$WORKFLOW_ROOT_FILE"; [[ -z "$saved_root" ]] || DEFAULT_DATASET_ROOT="$saved_root"; fi
DATASET_ROOT=$(ask_default "Dataset root" "$DEFAULT_DATASET_ROOT"); [[ "$DATASET_ROOT" == /* ]] || DATASET_ROOT="$PROJECT_ROOT/$DATASET_ROOT"
mkdir -p "$DATASET_ROOT/data" "$DATASET_ROOT/fitting" "$DATASET_ROOT/metadata"; DATASET_ROOT=$(cd "$DATASET_ROOT" && pwd); printf '%s\n' "$DATASET_ROOT" >"$WORKFLOW_ROOT_FILE"

TRAJ_DATASET="$DATASET_ROOT/data/trajectory.npz"; XYZ_ROOT="$DATASET_ROOT/data/xyz"; INITIAL_CAPPED="$DATASET_ROOT/data/xyz_capped"
INITIAL_QCHEM_MIN_IN="$DATASET_ROOT/data/qchem_min_input"; INITIAL_QCHEM_MIN_OUT="$DATASET_ROOT/data/qchem_min_output"
DISTANCES="$DATASET_ROOT/data/capping_distances.json"; CORRECTED_CAPPED="$DATASET_ROOT/data/xyz_capped_minimized"
FINAL_QCHEM_IN="$DATASET_ROOT/data/qchem_input"; FINAL_QCHEM_OUT="$DATASET_ROOT/data/qchem_output"
FIT_POLY="$DATASET_ROOT/fitting/poly"; OPTIMIZED="$DATASET_ROOT/fitting/optimized"
SELECTED_NMER_NAMES=(); SELECT_ALL=true

while true; do
    echo
    echo "Choose one operation:"
    echo "   1) Parse or recreate trajectory NPZ"
    echo "   2) Discover monomers, name AUTO monomers, and catalog n-mers"
    echo "   3) Sample selected n-mers into data/xyz"
    echo "   4) Initially cap selected sampled n-mers"
    echo "   5) Prepare minimization QChem inputs"
    echo "   6) Run minimization QChem jobs"
    echo "   7) Extract per-atom cap distances from minimized structures"
    echo "   8) Re-cap selected n-mers with minimized distances"
    echo "   9) Prepare final single-point QChem inputs"
    echo "  10) Run final single-point QChem jobs"
    echo "  11) Compute contributions and build final datasets"
    echo "   0) Exit"
    operation=$(ask_default "Operation number" 0)
    case "$operation" in
        1) stage_parse_trajectory || echo "Operation failed; inspect the stage log." >&2 ;;
        2) stage_discover_monomers || echo "Operation failed; inspect the stage log." >&2 ;;
        3) stage_build_xyz || echo "Operation failed; inspect the stage log." >&2 ;;
        4) stage_initial_cap || echo "Operation failed; inspect the stage log." >&2 ;;
        5) stage_prepare_minimization || echo "Operation failed; inspect the stage log." >&2 ;;
        6) stage_run_minimization || echo "Operation failed; inspect the stage log." >&2 ;;
        7) stage_extract_distances || echo "Operation failed; inspect the stage log." >&2 ;;
        8) stage_apply_distances || echo "Operation failed; inspect the stage log." >&2 ;;
        9) stage_prepare_final || echo "Operation failed; inspect the stage log." >&2 ;;
        10) stage_run_final || echo "Operation failed; inspect the stage log." >&2 ;;
        11) stage_build_dataset || echo "Operation failed; inspect the stage log." >&2 ;;
        0) echo "Dataset root: $DATASET_ROOT"; break ;;
        *) echo "Enter a number from 0 to 11." >&2 ;;
    esac
done
