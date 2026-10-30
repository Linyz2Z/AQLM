#!/usr/bin/env bash

set -euo pipefail

# chmod +x './scripts/Exp/run_eval_llama3.2_3b_AQLM_common_qa_2x8_e2e.sh'
# CUDA_VISIBLE_DEVICES=0 './scripts/Exp/run_eval_llama3.2_3b_AQLM_common_qa_2x8_e2e.sh'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AQLM_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/home/ubuntu/data/hf_datasets_cache}"

PYTHON_BIN="${PYTHON_BIN:-/home/ubuntu/miniconda3/envs/AQLM_llama32/bin/python}"
MODEL_PATH="${MODEL_PATH:-/home/ubuntu/data/models/llama3/Llama-3.2-3B}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/home/ubuntu/data/models/Exp/Llama-3.2-3B-AQLM-2x8-e2e}"
OUTPUT_PATH="${OUTPUT_PATH:-${CHECKPOINT_PATH}/lmeval-common-qa}"
TASKS="${TASKS:-winogrande,piqa,hellaswag,arc_easy,arc_challenge,boolq}"
DTYPE="${DTYPE:-float16}"

mkdir -p "${OUTPUT_PATH}"

echo "开始执行 Llama-3.2-3B AQLM 2x8 E2E checkpoint 的 Common QA 评测..."
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo "Python: ${PYTHON_BIN}"
echo "模型路径: ${MODEL_PATH}"
echo "checkpoint 路径: ${CHECKPOINT_PATH}"
echo "输出路径: ${OUTPUT_PATH}"
echo "评测任务: ${TASKS}"
echo "计算类型: ${DTYPE}"

cd "${AQLM_ROOT}"

"${PYTHON_BIN}" -u lmeval.py \
  --model hf \
  --model_args pretrained="${MODEL_PATH}",dtype="${DTYPE}" \
  --tasks "${TASKS}" \
  --batch_size auto \
  --output_path "${OUTPUT_PATH}" \
  --aqlm_checkpoint_path "${CHECKPOINT_PATH}"
