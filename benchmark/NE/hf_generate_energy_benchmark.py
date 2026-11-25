#!/usr/bin/env python3
"""Measure NVML GPU energy per request for the NE single-stream workloads."""

from __future__ import annotations

import argparse
import ctypes
import gc
import os
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from statistics import median
from zoneinfo import ZoneInfo

import torch

from hf_generate_benchmark import (
    ATTENTION_IMPLEMENTATION,
    generate_once,
    load_model,
    make_inputs,
    synchronize,
    validate_model_path,
    verify_decode_kernel,
    write_result,
)


MODELS = (
    ("FP16", "base", "Qwen2-1.5B", False),
    ("AWQ-4bit", "awq", "Qwen2-1.5B-AutoAWQ-4Bit-v2", False),
    ("AWQ-4bit-Opt", "awq", "Qwen2-1.5B-AutoAWQ-4Bit-v2", True),
    ("AQLM-2bit", "aqlm", "Qwen2-1.5B-AQLM-2x8-v2-e2e-HF", False),
    ("AQLM-PV-2bit", "aqlm", "Qwen2-1.5B-AQLM-2x8-200k-v2-ft-PV-HF", False),
    ("CRVQ-PV", "crvq", "Qwen2-1.5B-CRVQ-2x8+8+8-0.03-200k-appended-20pct-v4-ft-PV-HF", False),
)
SHANGHAI = ZoneInfo("Asia/Shanghai")


@contextmanager
def gpu_energy_reader(gpu_uuid: str) -> Iterator[Callable[[], int]]:
    """Bind the cumulative millijoule counter to the model's physical GPU."""
    nvml = ctypes.CDLL("libnvidia-ml.so.1")
    nvml.nvmlErrorString.argtypes = [ctypes.c_int]
    nvml.nvmlErrorString.restype = ctypes.c_char_p
    nvml.nvmlDeviceGetHandleByUUID.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]
    nvml.nvmlDeviceGetTotalEnergyConsumption.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulonglong)]

    def check(status: int, operation: str) -> None:
        if status:
            message = nvml.nvmlErrorString(status).decode()
            raise RuntimeError(f"{operation} failed for {gpu_uuid}: {message}")

    check(nvml.nvmlInit_v2(), "nvmlInit_v2")
    try:
        handle = ctypes.c_void_p()
        check(nvml.nvmlDeviceGetHandleByUUID(gpu_uuid.encode("ascii"), ctypes.byref(handle)), "NVML GPU lookup")

        def read_mj() -> int:
            energy = ctypes.c_ulonglong()
            status = nvml.nvmlDeviceGetTotalEnergyConsumption(handle, ctypes.byref(energy))
            check(status, "NVML total energy query")
            return energy.value

        yield read_mj
    finally:
        nvml.nvmlShutdown()


def measure_window(
    model: torch.nn.Module,
    inputs: dict[str, torch.Tensor],
    new_tokens: int,
    requests: int,
    read_mj: Callable[[], int],
) -> dict[str, int | float]:
    input_length = inputs["input_ids"].shape[-1]
    output_tokens = 0
    synchronize()
    start_mj = read_mj()
    started = time.perf_counter()
    for _ in range(requests):
        output, _ = generate_once(model, inputs, new_tokens)
        synchronize()
        output_tokens += output.shape[-1] - input_length
        del output
    elapsed = time.perf_counter() - started
    end_mj = read_mj()
    energy_j = (end_mj - start_mj) / 1000
    if energy_j <= 0:
        raise RuntimeError(f"Invalid NVML energy delta: start={start_mj} mJ, end={end_mj} mJ.")
    return {
        "requests": requests,
        "output_tokens": output_tokens,
        "elapsed_seconds": elapsed,
        "energy_start_mj": start_mj,
        "energy_end_mj": end_mj,
        "gpu_energy_j": energy_j,
        "gpu_j_per_request": energy_j / requests,
        "gpu_j_per_1000_output_tokens": energy_j * 1000 / output_tokens,
        "average_gpu_power_w": energy_j / elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-length", type=int, default=128)
    parser.add_argument("--new-tokens", type=int, default=128)
    parser.add_argument("--requests-per-window", type=int, default=20)
    parser.add_argument("--windows", type=int, default=3)
    parser.add_argument("--warmup-iters", type=int, default=1)
    parser.add_argument("--model-root", type=Path, default=Path(os.getenv("MODEL_ROOT", "/home/ubuntu/data/models/NE")))
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parent / "results" / "energy")
    args = parser.parse_args()
    if min(args.input_length, args.warmup_iters, args.requests_per_window, args.windows) < 1 or args.new_tokens < 2:
        parser.error("Use positive lengths and counts, with at least two generated tokens.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the GPU energy benchmark.")

    torch.cuda.set_device(0)
    torch.manual_seed(0)
    gpu = torch.cuda.get_device_properties(0)
    gpu_uuid = f"GPU-{str(gpu.uuid).removeprefix('GPU-')}"
    config = {
        "batch_size": 1,
        "input_length": args.input_length,
        "new_tokens": args.new_tokens,
        "warmup_iters": args.warmup_iters,
        "requests_per_window": args.requests_per_window,
        "windows": args.windows,
        "dtype": "float16",
        "attn_implementation": ATTENTION_IMPLEMENTATION,
        "use_cache": True,
        "do_sample": False,
        "num_beams": 1,
        "seed": 0,
        "stream_output": True,
    }

    with gpu_energy_reader(gpu_uuid) as read_mj:
        read_mj()
        args.output_root.mkdir(parents=True, exist_ok=True)
        while True:
            run = args.output_root / f"ne_hf_energy.{datetime.now(SHANGHAI):%Y%m%d_%H%M%S}"
            try:
                run.mkdir()
            except FileExistsError:
                time.sleep(1)
            else:
                break

        metadata = {
            "run_id": run.name,
            "energy_scope": "nvml_gpu_total_energy",
            "environment": {"gpu": gpu.name, "gpu_uuid": gpu_uuid, "torch": torch.__version__, "cuda": torch.version.cuda},
            "configuration": config,
        }
        print(f"Energy results: {run.resolve()}", flush=True)
        models = []
        for label, kind, directory, awq_fuse_layers in MODELS:
            path = validate_model_path(args.model_root / directory, kind)
            print(f"Measuring {label} on {gpu.name}", flush=True)
            model = load_model(
                path,
                kind,
                ATTENTION_IMPLEMENTATION,
                max_seq_len=args.input_length + args.new_tokens,
                awq_fuse_layers=awq_fuse_layers,
            )
            if kind in ("awq", "aqlm", "crvq"):
                verify_decode_kernel(model, kind, seed=0)
            inputs = make_inputs(model, batch_size=1, input_length=args.input_length, seed=0)
            for _ in range(args.warmup_iters):
                generate_once(model, inputs, args.new_tokens)
                synchronize()

            windows = []
            for index in range(1, args.windows + 1):
                window = measure_window(model, inputs, args.new_tokens, args.requests_per_window, read_mj)
                windows.append({"window_index": index, **window})
                print(f"  Window {index}/{args.windows}: {window['gpu_j_per_request']:.2f} J/request", flush=True)

            row = {
                "model_label": label,
                "model_path": str(path),
                "sample_count": len(windows),
                "gpu_j_per_request_p50": median(w["gpu_j_per_request"] for w in windows),
                "gpu_j_per_1000_output_tokens_p50": median(w["gpu_j_per_1000_output_tokens"] for w in windows),
                "average_gpu_power_w_p50": median(w["average_gpu_power_w"] for w in windows),
            }
            if kind == "awq":
                row["awq_fuse_layers"] = awq_fuse_layers
            write_result(run / f"{label}.json", {**metadata, **row, "windows": windows})
            models.append(row)
            del model, inputs
            gc.collect()
            torch.cuda.empty_cache()

    energy_by_model = {row["model_label"]: row["gpu_j_per_request_p50"] for row in models}
    for row in models:
        energy = row["gpu_j_per_request_p50"]
        row["gpu_energy_per_request_reduction_vs_fp16_pct"] = 100 * (1 - energy / energy_by_model["FP16"])
        row["gpu_energy_per_request_reduction_vs_awq_pct"] = 100 * (1 - energy / energy_by_model["AWQ-4bit"])
    write_result(run / "summary.json", {**metadata, "models": models})


if __name__ == "__main__":
    main()
