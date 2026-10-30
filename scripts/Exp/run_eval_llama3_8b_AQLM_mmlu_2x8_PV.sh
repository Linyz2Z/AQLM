#!/usr/bin/env bash

set -euo pipefail

# chmod +x './scripts/Exp/run_eval_llama3_8b_AQLM_mmlu_2x8_PV.sh'
# CUDA_VISIBLE_DEVICES=0 './scripts/Exp/run_eval_llama3_8b_AQLM_mmlu_2x8_PV.sh'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AQLM_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HOME="${HF_HOME:-/home/ubuntu/data/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/home/ubuntu/data/hf_datasets_cache}"
export TOKENIZERS_PARALLELISM=false

PYTHON_BIN="${PYTHON_BIN:-/home/ubuntu/miniconda3/envs/AQLM/bin/python}"
MODEL_PATH="${MODEL_PATH:-/home/ubuntu/data/models/llama3/llama3-8b}"
PV_SAVE_PATH="${PV_SAVE_PATH:-/home/ubuntu/data/models/Exp/Llama3-8B-AQLM-2x8-200k-4096-ft-PV}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${PV_SAVE_PATH}/best_model_aqlm}"
OUTPUT_PATH="${OUTPUT_PATH:-${PV_SAVE_PATH}/lmeval-mmlu}"
TASKS="${TASKS:-mmlu}"
NUM_FEWSHOT="${NUM_FEWSHOT:-5}"
DTYPE="${DTYPE:-float16}"
BATCH_SIZE="${BATCH_SIZE:-auto}"

if [[ ! -f "${CHECKPOINT_PATH}/0.pth" || ! -f "${CHECKPOINT_PATH}/not_quantized_weights.pt" ]]; then
  echo "错误：未找到转换后的普通 AQLM checkpoint：${CHECKPOINT_PATH}" >&2
  echo "请先运行对应 finetune PV 脚本顶部的 convert_legacy_model_format.py 命令。" >&2
  exit 1
fi
mkdir -p "${OUTPUT_PATH}"

echo "开始执行 Llama3-8B AQLM 2x8 PV-Tuning checkpoint 的 MMLU 评测..."
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo "Python: ${PYTHON_BIN}"
echo "模型路径: ${MODEL_PATH}"
echo "转换后 checkpoint 路径: ${CHECKPOINT_PATH}"
echo "输出路径: ${OUTPUT_PATH}"
echo "评测任务: ${TASKS}"
echo "few-shot: ${NUM_FEWSHOT}"
echo "计算类型: ${DTYPE}"
echo "batch size: ${BATCH_SIZE}"

cd "${AQLM_ROOT}"

"${PYTHON_BIN}" -u lmeval.py \
  --model hf \
  --model_args pretrained="${MODEL_PATH}",dtype="${DTYPE}" \
  --tasks "${TASKS}" \
  --num_fewshot "${NUM_FEWSHOT}" \
  --batch_size "${BATCH_SIZE}" \
  --output_path "${OUTPUT_PATH}" \
  --aqlm_checkpoint_path "${CHECKPOINT_PATH}"

"${PYTHON_BIN}" - "${OUTPUT_PATH}/results.json" <<'PY'
import json
import sys

results_path = sys.argv[1]
with open(results_path, "r", encoding="utf-8") as f:
    data = json.load(f)

groups = data.get("groups", {})
results = data.get("results", {})


def get_entry(name):
    return groups.get(name) or results.get(name)


def get_acc(entry):
    if not entry:
        return None
    for key in ("acc,none", "acc"):
        if key in entry:
            return entry[key]
    for key, value in entry.items():
        if key.startswith("acc,") and not key.startswith("acc_stderr"):
            return value
    return None


items = [
    ("mmlu_stem", "MMLU-STEM"),
    ("mmlu_humanities", "MMLU-Humanities"),
    ("mmlu_social_sciences", "MMLU-Social Sciences"),
    ("mmlu_other", "MMLU-Other"),
    ("mmlu", "MMLU-Overall"),
]

missing = []
print("\nMMLU accuracy summary")
print("subset\taccuracy")
for key, label in items:
    acc = get_acc(get_entry(key))
    if acc is None:
        missing.append(key)
        print(f"{label}\tN/A")
    else:
        print(f"{label}\t{acc:.4f} ({acc * 100:.2f}%)")

if missing:
    raise SystemExit(f"Missing MMLU metrics in {results_path}: {', '.join(missing)}")
PY
