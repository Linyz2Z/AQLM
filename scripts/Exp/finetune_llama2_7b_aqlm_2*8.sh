#!/usr/bin/env bash

set -euo pipefail

# chmod +x './scripts/Exp/finetune_llama2_7b_aqlm_2*8.sh'
# mkdir -p ./logs/Exp
# CUDA_VISIBLE_DEVICES=0 nohup './scripts/Exp/finetune_llama2_7b_aqlm_2*8.sh' > './logs/Exp/finetune_llama2_7b_aqlm_2*8.log' 2>&1 &
# tail -f './logs/Exp/finetune_llama2_7b_aqlm_2*8.log'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AQLM_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/home/ubuntu/data/hf_datasets_cache}"

PYTHON_BIN="${PYTHON_BIN:-/home/ubuntu/miniconda3/envs/AQLM/bin/python}"
MODEL_PATH="${MODEL_PATH:-/home/ubuntu/data/models/llama2/Llama-2-7B}"
DATASET_NAME="${DATASET_NAME:-pajama}"
SAVE_ROOT="${SAVE_ROOT:-/home/ubuntu/data/models/Exp}"
QUANT_MODEL_PATH="${QUANT_MODEL_PATH:-${SAVE_ROOT}/Llama-2-7B-AQLM-2x8}"
SAVE_PATH="${SAVE_PATH:-${SAVE_ROOT}/Llama-2-7B-AQLM-2x8-e2e}"

mkdir -p "${SAVE_ROOT}" "${SAVE_PATH}"

echo "开始执行 Llama2-7B AQLM 2x8 E2E 微调任务..."
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo "Python: ${PYTHON_BIN}"
echo "基础模型路径: ${MODEL_PATH}"
echo "量化模型路径: ${QUANT_MODEL_PATH}"
echo "数据集: ${DATASET_NAME}"
echo "保存路径: ${SAVE_PATH}"

cd "${AQLM_ROOT}"

"${PYTHON_BIN}" -u finetune_e2e.py \
  --base_model "${MODEL_PATH}" \
  --quant_model "${QUANT_MODEL_PATH}" \
  --dataset "${DATASET_NAME}" \
  --model_seqlen 4096 \
  --eval_model_seqlen 4096 \
  --nsamples 2048 \
  --val_size 128 \
  --lr 1e-5 \
  --adam_beta1 0.90 \
  --adam_beta2 0.999 \
  --epochs 3 \
  --early_stop 3 \
  --batch_size 16 \
  --microbatch_size 1 \
  --save "${SAVE_PATH}" \
  --gradient_checkpointing \
  --amp \
  --device_map auto \
  --use_fast_tokenizer
