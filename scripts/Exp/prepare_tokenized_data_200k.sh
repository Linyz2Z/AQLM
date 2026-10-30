#!/usr/bin/env bash

set -euo pipefail

# Prepare the first 200k RedPajama records for AQLM PV-Tuning.
# Default: Qwen2.5-1.5B tokenizer, 1024 tokens per sequence.
#   ./scripts/Exp/prepare_tokenized_data_200k.sh
#   SEQLEN=4096 ./scripts/Exp/prepare_tokenized_data_200k.sh
# For another tokenizer, override both MODEL_PATH and OUTPUT_PATH.
# finetune.py requires CUDA/distributed initialization even for preprocessing.
# No model weights are loaded before --save_dataset_and_exit returns.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AQLM_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HOME="${HF_HOME:-/home/ubuntu/data/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/home/ubuntu/data/hf_datasets_cache}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"

PYTHON_BIN="${PYTHON_BIN:-/home/ubuntu/miniconda3/envs/AQLM/bin/python}"
MODEL_PATH="${MODEL_PATH:-/home/ubuntu/data/models/Qwen/Qwen2.5-1.5B}"
BLOCK_TYPE="${BLOCK_TYPE:-Qwen2DecoderLayer}"
DATASET_NAME="${DATASET_NAME:-ZengXiangyu/RedPajama-Data-1T-Sample}"
DATASET_CONFIG_NAME="${DATASET_CONFIG_NAME:-}"
DATASET_SPLIT="${DATASET_SPLIT:-train[:200000]}"
SEQLEN="${SEQLEN:-1024}"
OUTPUT_PATH="${OUTPUT_PATH:-/home/ubuntu/data/datasets/Exp/redpajama_tokenized_qwen2_5_1p5b_200k_${SEQLEN}}"
CACHE_DIR="${CACHE_DIR:-${HF_DATASETS_CACHE}}"
PREPROCESSING_NUM_WORKERS="${PREPROCESSING_NUM_WORKERS:-64}"
DOWNLOAD_NUM_WORKERS="${DOWNLOAD_NUM_WORKERS:-1}"
PREPROCESSING_CHUNK_LENGTH="${PREPROCESSING_CHUNK_LENGTH:-100000}"

if [[ -e "${OUTPUT_PATH}" ]]; then
  echo "错误：输出路径已经存在：${OUTPUT_PATH}" >&2
  echo "请指定新的 OUTPUT_PATH，或先自行备份并移走已有目录。" >&2
  exit 1
fi

mkdir -p "$(dirname "${OUTPUT_PATH}")" "${CACHE_DIR}" "${HF_HOME}" "${HF_DATASETS_CACHE}"

echo "开始准备 AQLM PV-Tuning 200k 数据集..."
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo "Python: ${PYTHON_BIN}"
echo "Tokenizer 模型: ${MODEL_PATH}"
echo "数据集: ${DATASET_NAME}"
echo "数据集配置: ${DATASET_CONFIG_NAME:-<默认配置>}"
echo "数据集 split: ${DATASET_SPLIT}"
echo "序列长度: ${SEQLEN}"
echo "预处理进程数: ${PREPROCESSING_NUM_WORKERS}"
echo "缓存目录: ${CACHE_DIR}"
echo "输出目录: ${OUTPUT_PATH}"
echo "注意：200k 指原始记录数，tokenize 后的定长序列数可能不同。"

cd "${AQLM_ROOT}"

DATASET_CONFIG_ARGS=()
if [[ -n "${DATASET_CONFIG_NAME}" ]]; then
  DATASET_CONFIG_ARGS+=(--dataset_config_name "${DATASET_CONFIG_NAME}")
fi

# Use the selected Python environment's torchrun with an automatic local port.
# quantized_model and block_type are required parser arguments, unused here.
# README's --dtype is outdated; the current parser accepts --load_dtype.
"${PYTHON_BIN}" -u -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=1 \
  finetune.py \
  --base_model "${MODEL_PATH}" \
  --quantized_model ./doesnt_matter \
  --block_type "${BLOCK_TYPE}" \
  --load_dtype auto \
  --dataset_name "${DATASET_NAME}" \
  "${DATASET_CONFIG_ARGS[@]}" \
  --split "${DATASET_SPLIT}" \
  --model_seqlen "${SEQLEN}" \
  --cache_dir "${CACHE_DIR}" \
  --download_num_workers "${DOWNLOAD_NUM_WORKERS}" \
  --preprocessing_num_workers "${PREPROCESSING_NUM_WORKERS}" \
  --preprocessing_chunk_length "${PREPROCESSING_CHUNK_LENGTH}" \
  --trust_remote_code \
  --save_dataset_and_exit "${OUTPUT_PATH}"

echo "200k 数据集准备完成：${OUTPUT_PATH}"
echo "PV-Tuning 时设置：--dataset_name ${OUTPUT_PATH} --model_seqlen ${SEQLEN}"
