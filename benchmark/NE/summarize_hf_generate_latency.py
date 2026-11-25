#!/usr/bin/env python3
"""Summarize Hugging Face streaming latency and throughput for one workload."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median, quantiles


def summarize_metric(values: list[float]) -> dict[str, float]:
    p95 = quantiles(values, n=100, method="inclusive")[94] if len(values) > 1 else values[0]
    return {"p50": round(median(values), 4), "p95": round(p95, 4)}


def read_e2e_throughput(result: dict, path: Path) -> float | None:
    if "e2e_throughput_tokens_per_second" in result:
        return result["e2e_throughput_tokens_per_second"]

    # Older NE runs recorded the same aggregate in the paired memory JSON.
    path = path.resolve()
    if len(path.parents) >= 3 and path.parents[2].name == "latency":
        latency_dir = path.parents[2]
        memory_path = latency_dir.with_name("memory") / path.relative_to(latency_dir)
        if memory_path.is_file():
            memory_result = json.loads(memory_path.read_text())
            if (
                memory_result.get("model_path") == result["model_path"]
                and memory_result.get("configuration") == result["configuration"]
            ):
                return memory_result.get("timing", {}).get("generated_tokens_per_second")
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    results = [json.loads(path.read_text()) for path in args.inputs]
    configuration = results[0]["configuration"]
    timing_scope = results[0]["timing_scope"]
    expected_samples = configuration["benchmark_iters"]
    expected_tokens = configuration["new_tokens"]
    input_tokens = configuration["input_length"]

    models = []
    for path, result in zip(args.inputs, results):
        label = result["model_label"]
        if result["configuration"] != configuration or result["timing_scope"] != timing_scope:
            raise ValueError(f"Latency measurement configuration mismatch for {label}.")
        requests = result["requests"]
        if len(requests) != expected_samples or any(req["output_tokens"] != expected_tokens for req in requests):
            raise ValueError(f"Unexpected request count or output token count for {label}.")

        ttft_ms = [req["ttft_ms"] for req in requests]
        tpot_ms = [req["tpot_ms"] for req in requests]
        model = {
            "model_kind": result["model_kind"],
            "model_label": label,
            "model_path": result["model_path"],
            "sample_count": len(requests),
            "ttft_ms": summarize_metric(ttft_ms),
            "tpot_ms": summarize_metric(tpot_ms),
            "prefill_throughput_tokens_per_second": summarize_metric([input_tokens * 1000 / ms for ms in ttft_ms]),
            "decode_throughput_tokens_per_second": summarize_metric([1000 / ms for ms in tpot_ms]),
            "e2e_throughput_tokens_per_second": read_e2e_throughput(result, path),
        }
        if result["model_kind"] == "awq":
            model["awq_fuse_layers"] = result.get("awq_fuse_layers", configuration.get("awq_fuse_layers"))
        models.append(model)

    summary = {"configuration": configuration, "timing_scope": timing_scope, "models": models}
    output = args.output_json.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    print(
        f"{'Model':<14} {'N':>2}  {'TTFT ms':>8}  {'TPOT ms':>8}"
        f"  {'Prefill throughput token/s':>26}  {'Decode throughput token/s':>25}  {'E2E throughput token/s':>22}"
    )
    for model in models:
        e2e_throughput = model["e2e_throughput_tokens_per_second"]
        e2e_display = "—" if e2e_throughput is None else f"{e2e_throughput:.2f}"
        print(
            f"{model['model_label']:<14} {model['sample_count']:>2}"
            f"  {model['ttft_ms']['p50']:>8.2f}  {model['tpot_ms']['p50']:>8.2f}"
            f"  {model['prefill_throughput_tokens_per_second']['p50']:>26.2f}"
            f"  {model['decode_throughput_tokens_per_second']['p50']:>25.2f}"
            f"  {e2e_display:>22}"
        )
    print(f"Latency summary: {output}")


if __name__ == "__main__":
    main()
