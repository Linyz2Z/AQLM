#!/usr/bin/env bash

set -euo pipefail

# chmod +x './scripts/NE/run_qwen2_1p5b_aqlm_2*8_v1.sh'
# mkdir -p ./logs/NE
# CUDA_VISIBLE_DEVICES=3 nohup './scripts/NE/run_qwen2_1p5b_aqlm_2*8_v1.sh' > './logs/NE/run_qwen2_1p5b_aqlm_2*8_v1.log' 2>&1 &
# tail -f './logs/NE/run_qwen2_1p5b_aqlm_2*8_v1.log'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AQLM_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/home/ubuntu/data/hf_datasets_cache}"

PYTHON_BIN="${PYTHON_BIN:-/home/ubuntu/miniconda3/envs/AQLM/bin/python}"
MODEL_PATH="${MODEL_PATH:-/home/ubuntu/data/models/NE/Qwen2-1.5B}"
DATASET_NAME="${DATASET_NAME:-pajama}"
SAVE_ROOT="${SAVE_ROOT:-/home/ubuntu/data/models/NE}"
SAVE_PATH="${SAVE_PATH:-${SAVE_ROOT}/Qwen2-1.5B-AQLM-2x8-v1}"

mkdir -p "${SAVE_ROOT}" "${SAVE_PATH}"

echo "开始执行 Qwen2-1.5B AQLM 2x8 v1 量化任务..."
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo "Python: ${PYTHON_BIN}"
echo "模型路径: ${MODEL_PATH}"
echo "数据集: ${DATASET_NAME}"
echo "保存路径: ${SAVE_PATH}"

cd "${AQLM_ROOT}"

"${PYTHON_BIN}" -u main.py "${MODEL_PATH}" "${DATASET_NAME}" \
  --nsamples 1024 \
  --model_seqlen 1024 \
  --val_size 128 \
  --seed 42 \
  --num_codebooks 2 \
  --nbits_per_codebook 8 \
  --out_group_size 1 \
  --in_group_size 8 \
  --beam_size 4 \
  --max_epochs 1000 \
  --steps_per_epoch 100 \
  --relative_mse_tolerance 0.01 \
  --finetune_lr 1e-5 \
  --finetune_batch_size 32 \
  --local_batch_size 4 \
  --finetune_keep_best \
  --finetune_max_epochs 10 \
  --finetune_early_stop 3 \
  --use_checkpointing \
  --use_fast_tokenizer \
  --offload_activations \
  --save "${SAVE_PATH}"
