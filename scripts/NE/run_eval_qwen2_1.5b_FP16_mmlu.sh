#!/usr/bin/env bash

set -euo pipefail

# chmod +x './scripts/NE/run_eval_qwen2_1.5b_FP16_mmlu.sh'
# CUDA_VISIBLE_DEVICES=0 './scripts/NE/run_eval_qwen2_1.5b_FP16_mmlu.sh'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AQLM_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/home/ubuntu/data/hf_datasets_cache}"

PYTHON_BIN="${PYTHON_BIN:-/home/ubuntu/miniconda3/envs/AQLM/bin/python}"
MODEL_PATH="${MODEL_PATH:-/home/ubuntu/data/models/NE/Qwen2-1.5B}"
OUTPUT_PATH="${OUTPUT_PATH:-${MODEL_PATH}/lmeval-fp16-mmlu}"
TASKS="${TASKS:-mmlu}"
NUM_FEWSHOT="${NUM_FEWSHOT:-5}"
DTYPE="${DTYPE:-float16}"

mkdir -p "${OUTPUT_PATH}"

echo "开始执行 Qwen2-1.5B FP16 MMLU 评测..."
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo "Python: ${PYTHON_BIN}"
echo "模型路径: ${MODEL_PATH}"
echo "数据类型: ${DTYPE}"
echo "输出路径: ${OUTPUT_PATH}"
echo "评测任务: ${TASKS}"
echo "few-shot: ${NUM_FEWSHOT}"

cd "${AQLM_ROOT}"

"${PYTHON_BIN}" -u lmeval.py \
  --model hf \
  --model_args pretrained="${MODEL_PATH}",dtype="${DTYPE}" \
  --tasks "${TASKS}" \
  --num_fewshot "${NUM_FEWSHOT}" \
  --batch_size auto \
  --output_path "${OUTPUT_PATH}"

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
