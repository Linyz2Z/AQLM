#!/usr/bin/env bash

set -euo pipefail

# chmod +x './scripts/Exp/run_llama2_7b_aqlm_2*8.sh'
# mkdir -p ./logs/Exp
# CUDA_VISIBLE_DEVICES=0 nohup './scripts/Exp/run_llama2_7b_aqlm_2*8.sh' > './logs/Exp/run_llama2_7b_aqlm_2*8.log' 2>&1 &
# tail -f './logs/Exp/run_llama2_7b_aqlm_2*8.log'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AQLM_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/home/ubuntu/data/hf_datasets_cache}"

PYTHON_BIN="${PYTHON_BIN:-/home/ubuntu/miniconda3/envs/AQLM/bin/python}"
MODEL_PATH="${MODEL_PATH:-/home/ubuntu/data/models/llama2/Llama-2-7B}"
DATASET_NAME="${DATASET_NAME:-pajama}"
SAVE_ROOT="${SAVE_ROOT:-/home/ubuntu/data/models/Exp}"
SAVE_PATH="${SAVE_PATH:-${SAVE_ROOT}/Llama-2-7B-AQLM-2x8}"

mkdir -p "${SAVE_ROOT}" "${SAVE_PATH}"

echo "开始执行 Llama2-7B AQLM 2x8 量化任务..."
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo "Python: ${PYTHON_BIN}"
echo "模型路径: ${MODEL_PATH}"
echo "数据集: ${DATASET_NAME}"
echo "保存路径: ${SAVE_PATH}"

cd "${AQLM_ROOT}"

"${PYTHON_BIN}" -u main.py "${MODEL_PATH}" "${DATASET_NAME}" \
  --nsamples 2048 \
  --val_size 128 \
  --model_seqlen 4096 \
  --seed 42 \
  --num_codebooks 2 \
  --nbits_per_codebook 8 \
  --out_group_size 1 \
  --in_group_size 8 \
  --beam_size 4 \
  --max_epochs 100 \
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
