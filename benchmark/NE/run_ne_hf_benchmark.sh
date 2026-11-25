#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${PYTHON_BIN:-/home/ubuntu/miniconda3/envs/AQLM-AWQ/bin/python}"
model_root="${MODEL_ROOT:-/home/ubuntu/data/models/NE}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export TOKENIZERS_PARALLELISM=false

if (( $# > 1 )); then
    printf 'Expected at most one workload argument.\n' >&2
    exit 2
fi

case "${1:-smoke}" in
    smoke) workloads=(128:128) ;;
    quick) workloads=(128:128 512:512 2048:2048) ;;
    full) workloads=(128:128 512:128 2048:128 128:512 128:2048 512:512 2048:2048) ;;
    *) printf 'Usage: %s [smoke|quick|full]\n' "$0" >&2 && exit 2 ;;
esac

models=(
    'FP16|base|Qwen2-1.5B'
    'AWQ-4bit|awq|Qwen2-1.5B-AutoAWQ-4Bit-v2|--no-awq-fuse-layers'
    'AWQ-4bit-Opt|awq|Qwen2-1.5B-AutoAWQ-4Bit-v2|--awq-fuse-layers'
    'AQLM-2bit|aqlm|Qwen2-1.5B-AQLM-2x8-v2-e2e-HF'
    'AQLM-PV-2bit|aqlm|Qwen2-1.5B-AQLM-2x8-200k-v2-ft-PV-HF'
    'CRVQ-PV|crvq|Qwen2-1.5B-CRVQ-2x8+8+8-0.03-200k-appended-20pct-v4-ft-PV-HF'
)

results_root="${OUTPUT_ROOT:-${script_dir}/results}"
mkdir -p "${results_root}/memory" "${results_root}/latency" "${results_root}/logs"
while :
do
    run_id="ne_hf_benchmark.$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S')"
    memory_run="${results_root}/memory/${run_id}"
    latency_run="${results_root}/latency/${run_id}"
    logs_run="${results_root}/logs/${run_id}"
    if [[ ! -e "$latency_run" && ! -L "$latency_run" &&
          ! -e "$logs_run" && ! -L "$logs_run" ]]; then
        if mkdir -m 700 "$memory_run" 2>/dev/null; then
            break
        fi
        if [[ ! -e "$memory_run" && ! -L "$memory_run" ]]; then
            printf 'Cannot create memory results directory: %s\n' "$memory_run" >&2
            exit 1
        fi
    fi
    sleep 1
done
if ! mkdir "$latency_run"; then
    rmdir "$memory_run"
    exit 1
fi
if ! mkdir "$logs_run"; then
    rmdir "$memory_run" "$latency_run"
    exit 1
fi
printf 'Memory results: %s\n' "$memory_run"
printf 'Latency results: %s\n' "$latency_run"
printf 'Logs: %s\n' "$logs_run"

case_names=()
for workload in "${workloads[@]}"
do
    IFS=: read -r input_length new_tokens <<< "$workload"
    case_name="input_${input_length}_new_${new_tokens}"
    case_names+=("$case_name")
    memory_case="${memory_run}/${case_name}"
    latency_case="${latency_run}/${case_name}"
    logs_case="${logs_run}/${case_name}"
    mkdir "$memory_case" "$latency_case" "$logs_case"
    memory_results=()
    latency_results=()
    for model in "${models[@]}"
    do
        IFS='|' read -r label kind directory awq_option <<< "$model"
        model_options=()
        if [[ -n "$awq_option" ]]; then
            model_options+=("$awq_option")
        fi
        memory_result="${memory_case}/${label}.json"
        latency_result="${latency_case}/${label}.json"
        log="${logs_case}/${label}.log"
        printf 'Input %s, output %s, model %s\n' "$input_length" "$new_tokens" "$label"
        if ! "$python_bin" "${script_dir}/hf_generate_benchmark.py" \
            --model-label "$label" --model-kind "$kind" --model-path "${model_root}/${directory}" "${model_options[@]}" \
            --input-length "$input_length" --new-tokens "$new_tokens" \
            --batch-size 1 --dtype float16 --attn-implementation flash_attention_2 --seed 0 \
            --warmup-iters "${WARMUP_ITERS:-1}" --benchmark-iters "${BENCHMARK_ITERS:-3}" \
            --memory-output-json "$memory_result" --latency-output-json "$latency_result" > "$log" 2>&1
        then
            tail -n 50 "$log" >&2
            printf 'Failed: %s\n' "$log" >&2
            exit 1
        fi
        memory_results+=("$memory_result")
        latency_results+=("$latency_result")
    done
    "$python_bin" "${script_dir}/summarize_hf_generate_memory.py" \
        --inputs "${memory_results[@]}" --output-json "${memory_case}/summary.json"
    "$python_bin" "${script_dir}/summarize_hf_generate_latency.py" \
        --inputs "${latency_results[@]}" --output-json "${latency_case}/summary.json"
done

"$python_bin" - "$memory_run" "$latency_run" "${case_names[@]}" <<'PY'
import json
import sys
from pathlib import Path

memory_run, latency_run, *case_names = sys.argv[1:]
for run in map(Path, (memory_run, latency_run)):
    workloads = [json.loads((run / case / "summary.json").read_text()) for case in case_names]
    output = run / "summary.json"
    output.write_text(json.dumps({"run_id": run.name, "workloads": workloads}, indent=2) + "\n")
    print(f"Run summary: {output}")
PY
