#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${PYTHON_BIN:-/home/ubuntu/miniconda3/envs/AQLM-AWQ/bin/python}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5}"

exec "$python_bin" -u "${script_dir}/hf_generate_energy_benchmark.py" \
    --input-length 128 --new-tokens 128 \
    --requests-per-window 10 --windows 3 \
    "$@"
