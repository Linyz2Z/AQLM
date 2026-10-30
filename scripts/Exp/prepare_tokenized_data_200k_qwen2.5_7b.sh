#!/usr/bin/env bash

set -euo pipefail

# Prepare 200k raw RedPajama records using this model's own tokenizer.
# Usage: CUDA_VISIBLE_DEVICES=0 SEQLEN=4096 bash scripts/Exp/prepare_tokenized_data_200k_qwen2.5_7b.sh
# Override MODEL_PATH and OUTPUT_PATH together when changing models.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

export MODEL_PATH="${MODEL_PATH:-/home/ubuntu/data/models/Qwen/Qwen2.5-7B}"
export BLOCK_TYPE="${BLOCK_TYPE:-Qwen2DecoderLayer}"
export SEQLEN="${SEQLEN:-1024}"
export OUTPUT_PATH="${OUTPUT_PATH:-/home/ubuntu/data/datasets/Exp/redpajama_tokenized_qwen2_5_7b_200k_${SEQLEN}}"

exec bash "${SCRIPT_DIR}/prepare_tokenized_data_200k.sh"
