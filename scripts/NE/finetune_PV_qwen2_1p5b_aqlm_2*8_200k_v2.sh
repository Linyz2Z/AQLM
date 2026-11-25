#!/usr/bin/env bash

set -euo pipefail

# chmod +x './scripts/NE/finetune_PV_qwen2_1p5b_aqlm_2*8_200k_v2.sh'
# mkdir -p ./logs/NE
# CUDA_VISIBLE_DEVICES=0 nohup bash './scripts/NE/finetune_PV_qwen2_1p5b_aqlm_2*8_200k_v2.sh' > './logs/NE/finetune_PV_qwen2_1p5b_aqlm_2*8_200k_v2.log' 2>&1 &
# tail -f './logs/NE/finetune_PV_qwen2_1p5b_aqlm_2*8_200k_v2.log'
# 训练完成后，在 AQLM 根目录将 best_model 转换为普通 AQLM checkpoint：
# /home/ubuntu/miniconda3/envs/AQLM/bin/python -u convert_legacy_model_format.py --base_model '/home/ubuntu/data/models/NE/Qwen2-1.5B' --quantized_model '/home/ubuntu/data/models/NE/Qwen2-1.5B-AQLM-2x8-v2' --pv_fsdp_dir '/home/ubuntu/data/models/NE/Qwen2-1.5B-AQLM-2x8-200k-v2-ft-PV/best_model' --load_dtype bfloat16 --code_dtype uint16 --attn_implementation sdpa --trust_remote_code --save '/home/ubuntu/data/models/NE/Qwen2-1.5B-AQLM-2x8-200k-v2-ft-PV/best_model_aqlm'
# GPU count follows CUDA_VISIBLE_DEVICES; keep the same GPU count when resuming.
# AQLM automatically resumes when SAVE_PATH contains a complete training snapshot.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AQLM_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HOME="${HF_HOME:-/home/ubuntu/data/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/home/ubuntu/data/hf_datasets_cache}"
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-/home/ubuntu/.cache/huggingface/modules}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"

PYTHON_BIN="${PYTHON_BIN:-/home/ubuntu/miniconda3/envs/AQLM/bin/python}"
# Count visible devices (also supports comma-separated GPU UUIDs).
IFS=',' read -r -a VISIBLE_GPUS <<< "${CUDA_VISIBLE_DEVICES}"
NUM_GPUS="${#VISIBLE_GPUS[@]}"
MODEL_PATH="${MODEL_PATH:-/home/ubuntu/data/models/NE/Qwen2-1.5B}"
QUANT_MODEL_PATH="${QUANT_MODEL_PATH:-/home/ubuntu/data/models/NE/Qwen2-1.5B-AQLM-2x8-v2}"
DATASET_PATH="${DATASET_PATH:-/home/ubuntu/data/datasets/NE/redpajama_tokenized_qwen2_1p5b_200k_4096}"
SAVE_PATH="${SAVE_PATH:-/home/ubuntu/data/models/NE/Qwen2-1.5B-AQLM-2x8-200k-v2-ft-PV}"
CACHE_DIR="${CACHE_DIR:-${HF_DATASETS_CACHE}}"
BATCH_SIZE="${BATCH_SIZE:-128}"
EVAL_EVERY_STEPS="${EVAL_EVERY_STEPS:-10}"
SAVE_EVERY_STEPS="${SAVE_EVERY_STEPS:-100}"
NUM_WORKERS="${NUM_WORKERS:-8}"

for required_path in "${PYTHON_BIN}" "${MODEL_PATH}" "${QUANT_MODEL_PATH}" "${DATASET_PATH}"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "错误：所需路径不存在：${required_path}" >&2
    exit 1
  fi
done

# Do not pre-create SAVE_PATH; finetune.py manages training snapshots and resuming.
mkdir -p "$(dirname "${SAVE_PATH}")" "${CACHE_DIR}"

echo "开始执行 Qwen2-1.5B AQLM 2x8 200k v2 PV-Tuning..."
echo "GPU: ${CUDA_VISIBLE_DEVICES}（${NUM_GPUS} 个进程）"
echo "基础模型: ${MODEL_PATH}"
echo "量化模型: ${QUANT_MODEL_PATH}"
echo "训练数据: ${DATASET_PATH}"
echo "序列长度: 4096"
echo "训练范围: 200k RedPajama 预处理数据集的全部序列"
echo "全局 batch size: ${BATCH_SIZE}; 每卡 microbatch size: 1"
echo "PPL 验证频率: 每 ${EVAL_EVERY_STEPS} 个 optimizer steps"
echo "状态保存频率: 每 ${SAVE_EVERY_STEPS} 个 optimizer steps"
echo "保存路径: ${SAVE_PATH}"

cd "${AQLM_ROOT}"

"${PYTHON_BIN}" -u -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node="${NUM_GPUS}" \
  finetune.py \
  --base_model "${MODEL_PATH}" \
  --quantized_model "${QUANT_MODEL_PATH}" \
  --model_seqlen 4096 \
  --block_type Qwen2DecoderLayer \
  --load_dtype bfloat16 \
  --amp_dtype bfloat16 \
  --master_dtype bfloat16 \
  --embed_dtype bfloat16 \
  --code_dtype uint16 \
  --dataset_name "${DATASET_PATH}" \
  --split none \
  --seed 42 \
  --preprocessing_chunk_length 100000 \
  --num_workers "${NUM_WORKERS}" \
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
  --batch_size "${BATCH_SIZE}" \
  --microbatch_size 1 \
  --max_epochs 1 \
  --gradient_checkpointing \
  --print_every_steps 1 \
  --verbose_optimizer \
  --eval_every_steps "${EVAL_EVERY_STEPS}" \
  --keep_best_model \
  --save "${SAVE_PATH}" \
  --save_every_steps "${SAVE_EVERY_STEPS}" \
  --attn_implementation sdpa
