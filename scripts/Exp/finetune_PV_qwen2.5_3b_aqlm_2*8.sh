#!/usr/bin/env bash

set -euo pipefail

# chmod +x './scripts/Exp/finetune_PV_qwen2.5_3b_aqlm_2*8.sh'
# mkdir -p ./logs/Exp
# CUDA_VISIBLE_DEVICES=1,3 nohup bash './scripts/Exp/finetune_PV_qwen2.5_3b_aqlm_2*8.sh' > './logs/Exp/finetune_PV_qwen2.5_3b_aqlm_2*8.log' 2>&1 &
# tail -f './logs/Exp/finetune_PV_qwen2.5_3b_aqlm_2*8.log'
# 训练完成后，在 AQLM 根目录将 best_model 转换为普通 AQLM checkpoint：
# /home/ubuntu/miniconda3/envs/AQLM/bin/python -u convert_legacy_model_format.py --base_model '/home/ubuntu/data/models/Qwen/Qwen2.5-3B' --quantized_model '/home/ubuntu/data/models/Exp/Qwen2.5-3B-AQLM-2x8' --pv_fsdp_dir '/home/ubuntu/data/models/Exp/Qwen2.5-3B-AQLM-2x8-200k-4096-ft-PV/best_model' --load_dtype bfloat16 --code_dtype uint16 --attn_implementation sdpa --trust_remote_code --save '/home/ubuntu/data/models/Exp/Qwen2.5-3B-AQLM-2x8-200k-4096-ft-PV/best_model_aqlm'
# Edit paths below when needed; GPU count follows CUDA_VISIBLE_DEVICES.
# An existing SAVE_PATH is automatically used by AQLM for resuming.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AQLM_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HOME=/home/ubuntu/data/huggingface
export HF_DATASETS_CACHE=/home/ubuntu/data/hf_datasets_cache
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=16

PYTHON_BIN=/home/ubuntu/miniconda3/envs/AQLM/bin/python
# Count visible devices (also supports comma-separated GPU UUIDs).
IFS=',' read -r -a VISIBLE_GPUS <<< "${CUDA_VISIBLE_DEVICES}"
NUM_GPUS="${#VISIBLE_GPUS[@]}"
MODEL_PATH="/home/ubuntu/data/models/Qwen/Qwen2.5-3B"
QUANT_MODEL_PATH="/home/ubuntu/data/models/Exp/Qwen2.5-3B-AQLM-2x8"
DATASET_PATH="/home/ubuntu/data/datasets/Exp/redpajama_tokenized_qwen2_5_3b_200k_4096"
SAVE_PATH="/home/ubuntu/data/models/Exp/Qwen2.5-3B-AQLM-2x8-200k-4096-ft-PV"
CACHE_DIR="${HF_DATASETS_CACHE}"

# Do not pre-create SAVE_PATH: finetune.py treats it as a checkpoint.
mkdir -p "$(dirname "${SAVE_PATH}")" "${CACHE_DIR}"
cd "${AQLM_ROOT}"

"${PYTHON_BIN}" -u -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node="${NUM_GPUS}" \
  finetune.py \
  --base_model "${MODEL_PATH}" \
  --quantized_model "${QUANT_MODEL_PATH}" \
  --model_seqlen 4096 \
  --block_type Qwen2DecoderLayer \
  --load_dtype bfloat16 \
  --amp_dtype bfloat16 \
  --code_dtype uint16 \
  --dataset_name "${DATASET_PATH}" \
  --split none \
  --seed 42 \
  --preprocessing_chunk_length 100000 \
  --cache_dir "${CACHE_DIR}" \
  --trust_remote_code \
  --update_codes \
  --update_codebooks_and_scales \
  --update_non_quantized_parameters \
  --lamb \
  --debias \
  --lr 3e-4 \
  --adam_beta1 0.90 \
  --adam_beta2 0.95 \
  --max_code_change_per_step 1e-2 \
  --code_lr 1e-2 \
  --code_beta1 0.0 \
  --code_beta2 0.95 \
  --beam_size 5 \
  --delta_decay 0 \
  --batch_size 128 \
  --microbatch_size 1 \
  --max_epochs 1 \
  --gradient_checkpointing \
  --print_every_steps 1 \
  --verbose_optimizer \
  --eval_every_steps 10 \
  --keep_best_model \
  --save "${SAVE_PATH}" \
  --save_every_steps 100 \
  --loss_tokens_per_chunk 128 \
  --limit_all_gathers \
  --attn_implementation sdpa
