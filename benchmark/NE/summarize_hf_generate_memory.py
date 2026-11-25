#!/usr/bin/env python3
"""Validate and summarize the memory results from hf_generate_benchmark.py."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    results = [json.loads(Path(path).read_text()) for path in args.inputs]
    reference_config = results[0]["configuration"]
    for result in results[1:]:
        if result["configuration"] != reference_config:
            raise ValueError(
                f"Configuration mismatch between {results[0]['model_kind']} and {result['model_kind']}."
            )

    models = []
    for result in results:
        memory = result["gpu_memory"]
        model = {
            "model_kind": result["model_kind"],
            "model_label": result.get("model_label", result["model_kind"]),
            "model_path": result["model_path"],
            "load_peak_allocated_mib": memory["load_peak_allocated_mib"],
            "steady_allocated_mib": memory["steady_allocated_mib"],
            "inference_peak_allocated_mib": memory["inference_peak_allocated_mib"],
            "inference_incremental_allocated_mib": memory["inference_incremental_allocated_mib"],
            "nvidia_smi_inference_peak_mib": memory["nvidia_smi_inference_peak_mib"],
        }
        if result["model_kind"] == "awq":
            model["awq_fuse_layers"] = result.get("awq_fuse_layers", reference_config.get("awq_fuse_layers"))
        models.append(model)
    summary = {"configuration": reference_config, "models": models}

    output_path = Path(args.output_json).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    columns = (
        "model",
        "load peak MiB",
        "steady MiB",
        "inference peak MiB",
        "increment MiB",
        "nvidia-smi peak MiB",
    )
    rows = []
    for model in summary["models"]:
        rows.append(
            (
                str(model["model_label"]),
                str(model["load_peak_allocated_mib"]),
                str(model["steady_allocated_mib"]),
                str(model["inference_peak_allocated_mib"]),
                str(model["inference_incremental_allocated_mib"]),
                str(model["nvidia_smi_inference_peak_mib"]),
            )
        )

    widths = [max(len(columns[index]), *(len(row[index]) for row in rows)) for index in range(len(columns))]
    print("  ".join(value.ljust(widths[index]) for index, value in enumerate(columns)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))
    print(f"Summary written to: {output_path}")


if __name__ == "__main__":
    main()
