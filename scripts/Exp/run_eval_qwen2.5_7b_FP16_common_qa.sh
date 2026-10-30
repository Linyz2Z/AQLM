#!/usr/bin/env bash

set -euo pipefail

# chmod +x './scripts/Exp/run_eval_qwen2.5_7b_FP16_common_qa.sh'
# CUDA_VISIBLE_DEVICES=0 './scripts/Exp/run_eval_qwen2.5_7b_FP16_common_qa.sh'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AQLM_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/home/ubuntu/data/hf_datasets_cache}"

PYTHON_BIN="${PYTHON_BIN:-/home/ubuntu/miniconda3/envs/AQLM/bin/python}"
MODEL_PATH="${MODEL_PATH:-/home/ubuntu/data/models/Qwen/Qwen2.5-7B}"
OUTPUT_PATH="${OUTPUT_PATH:-${MODEL_PATH}/lmeval-fp16-common-qa}"
TASKS="${TASKS:-winogrande,piqa,hellaswag,arc_easy,arc_challenge,boolq}"
DTYPE="${DTYPE:-float16}"

mkdir -p "${OUTPUT_PATH}"

echo "开始执行 Qwen2.5-7B FP16 Common QA 评测..."
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo "Python: ${PYTHON_BIN}"
echo "模型路径: ${MODEL_PATH}"
echo "输出路径: ${OUTPUT_PATH}"
echo "评测任务: ${TASKS}"
echo "计算类型: ${DTYPE}"

cd "${AQLM_ROOT}"

"${PYTHON_BIN}" -u lmeval.py \
  --model hf \
  --model_args pretrained="${MODEL_PATH}",dtype="${DTYPE}" \
  --tasks "${TASKS}" \
  --batch_size auto \
  --output_path "${OUTPUT_PATH}"
