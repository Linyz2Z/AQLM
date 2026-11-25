#!/usr/bin/env python3
"""Benchmark Hugging Face generate() latency and peak GPU memory in one process."""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import torch
from torch.autograd import DeviceType
from torch.profiler import ProfilerActivity, profile
from transformers import AutoConfig, AutoModelForCausalLM


MIB = 1024**2
GIB = 1024**3
ATTENTION_IMPLEMENTATION = "flash_attention_2"


class NvidiaSmiProcessMonitor:
    """Poll this process's GPU memory as reported by nvidia-smi."""

    def __init__(self, interval_seconds: float = 0.1) -> None:
        self.interval_seconds = interval_seconds
        self.pid = os.getpid()
        self.peak_mib: int | None = None
        self.error: str | None = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _poll(self) -> None:
        command = [
            "nvidia-smi",
            "--query-compute-apps=pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ]
        while not self._stop_event.is_set():
            try:
                result = subprocess.run(command, capture_output=True, text=True, timeout=3, check=False)
                if result.returncode != 0:
                    self.error = result.stderr.strip() or f"nvidia-smi exited with {result.returncode}"
                else:
                    for line in result.stdout.splitlines():
                        fields = [field.strip() for field in line.split(",")]
                        if len(fields) != 2:
                            continue
                        try:
                            process_id, used_mib = int(fields[0]), int(fields[1])
                        except ValueError:
                            continue
                        if process_id == self.pid:
                            self.peak_mib = used_mib if self.peak_mib is None else max(self.peak_mib, used_mib)
            except (FileNotFoundError, subprocess.SubprocessError) as exc:
                self.error = str(exc)
                return
            self._stop_event.wait(self.interval_seconds)


class TokenLatencyStreamer:
    """Record the first and last generated token arrivals in the host process."""

    def __init__(self) -> None:
        self.start_ns = 0
        self.first_token_ns: int | None = None
        self.last_token_ns = 0
        self.output_tokens = 0
        self.prompt_pending = True

    def put(self, tokens: torch.Tensor) -> None:
        if self.prompt_pending:
            # generate() sends the complete prompt before any output tokens.
            self.prompt_pending = False
            return
        received_ns = time.perf_counter_ns()
        if self.first_token_ns is None:
            self.first_token_ns = received_ns
        self.last_token_ns = received_ns
        self.output_tokens += tokens.numel()

    def end(self) -> None:
        pass

    def sample(self, expected_tokens: int) -> dict[str, int | float]:
        if self.first_token_ns is None or self.output_tokens != expected_tokens:
            raise RuntimeError(f"Expected {expected_tokens} streamed output tokens, got {self.output_tokens}.")
        return {
            "output_tokens": self.output_tokens,
            "ttft_ms": (self.first_token_ns - self.start_ns) / 1e6,
            "tpot_ms": (self.last_token_ns - self.first_token_ns) / (self.output_tokens - 1) / 1e6,
            "end_to_end_ms": (self.last_token_ns - self.start_ns) / 1e6,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure GPU memory and per-request latency for Hugging Face streaming generation."
    )
    parser.add_argument("--model-kind", required=True, choices=("base", "aqlm", "awq", "crvq"))
    parser.add_argument("--model-label", help="Optional display name for this checkpoint.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--awq-fuse-layers", action=argparse.BooleanOptionalAction, default=False,
        help="Enable AutoAWQ module fusion for the AWQ checkpoint.",
    )
    parser.add_argument(
        "--source-model-path",
        default=None,
        help="Optional original checkpoint path recorded as metadata (used for the legacy AQLM source).",
    )
    parser.add_argument("--memory-output-json", type=Path, required=True)
    parser.add_argument("--latency-output-json", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--input-length", type=int, default=2048)
    parser.add_argument("--new-tokens", type=int, default=128)
    parser.add_argument("--warmup-iters", type=int, default=1)
    parser.add_argument("--benchmark-iters", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", choices=("float16",), default="float16")
    parser.add_argument("--attn-implementation", choices=(ATTENTION_IMPLEMENTATION,), default=ATTENTION_IMPLEMENTATION)
    parser.add_argument("--nvml-poll-interval", type=float, default=0.1)
    return parser.parse_args()


def package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def mib(value: int) -> float:
    return round(value / MIB, 2)


def gib(value: int) -> float:
    return round(value / GIB, 4)


def validate_args(args: argparse.Namespace) -> Path:
    if args.awq_fuse_layers and args.model_kind != "awq":
        raise ValueError("--awq-fuse-layers requires --model-kind awq.")
    if args.batch_size != 1:
        raise ValueError("This benchmark fixes batch_size=1 so AQLM decode selects its fused GEMV CUDA kernel.")
    if args.input_length < 1 or args.new_tokens < 2:
        raise ValueError("input_length must be >= 1 and new_tokens must be >= 2.")
    if args.warmup_iters < 1 or args.benchmark_iters < 1:
        raise ValueError("warmup_iters and benchmark_iters must both be >= 1.")
    if args.memory_output_json.resolve() == args.latency_output_json.resolve():
        raise ValueError("Memory and latency JSON paths must be different.")

    return validate_model_path(Path(args.model_path), args.model_kind)


def validate_model_path(model_path: Path, model_kind: str) -> Path:
    model_path = model_path.resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {model_path}")
    if not (model_path / "config.json").is_file():
        raise FileNotFoundError(f"Missing model config: {model_path / 'config.json'}")

    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    quantization_config = getattr(config, "quantization_config", None) or {}
    quant_method = quantization_config.get("quant_method")
    expected_method = {"base": None, "aqlm": "aqlm", "awq": "awq", "crvq": None}[model_kind]
    if quant_method != expected_method:
        raise ValueError(
            f"Expected quant_method={expected_method!r} for {model_kind}, found {quant_method!r} in {model_path}."
        )

    if model_kind == "aqlm":
        expected = {
            "num_codebooks": 2,
            "nbits_per_codebook": 8,
            "in_group_size": 8,
            "out_group_size": 1,
        }
        actual = {key: quantization_config.get(key) for key in expected}
        if actual != expected:
            raise ValueError(f"AQLM CUDA code2x8 kernel requires {expected}, found {actual}.")

    if model_kind == "awq" and quantization_config.get("version") != "gemm":
        raise ValueError(f"AWQ-GEMM/CUDA requires a GEMM checkpoint in {model_path}.")

    if model_kind == "crvq" and not (
        getattr(config, "crvq_modules", None)
        and getattr(config, "auto_map", {}).get("AutoModelForCausalLM")
    ):
        raise ValueError(f"Expected a CRVQ HF export with a module manifest and custom model class in {model_path}.")

    return model_path


def synchronize() -> None:
    torch.cuda.synchronize(torch.device("cuda:0"))


def make_inputs(model: torch.nn.Module, batch_size: int, input_length: int, seed: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    input_ids = torch.randint(
        low=0,
        high=model.config.vocab_size,
        size=(batch_size, input_length),
        generator=generator,
        dtype=torch.long,
    ).to("cuda:0")
    return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}


def generate_once(
    model: torch.nn.Module,
    inputs: dict[str, torch.Tensor],
    new_tokens: int,
) -> tuple[torch.Tensor, TokenLatencyStreamer]:
    eos_token_id = getattr(model.config, "eos_token_id", None)
    pad_token_id = getattr(model.config, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = eos_token_id

    decoder = getattr(model, "model", None)
    if hasattr(decoder, "last_forward_num_tokens"):
        # AutoAWQ keeps cache tensors and positions on the fused decoder across generate() calls.
        decoder.last_forward_num_tokens = 0
        for block in decoder.blocks:
            block.attn.start_pos = 0

    streamer = TokenLatencyStreamer()
    with torch.inference_mode():
        streamer.start_ns = time.perf_counter_ns()
        output = model.generate(
            **inputs,
            min_new_tokens=new_tokens,
            max_new_tokens=new_tokens,
            do_sample=False,
            num_beams=1,
            use_cache=True,
            pad_token_id=pad_token_id,
            streamer=streamer,
        )
    if output.shape != (inputs["input_ids"].shape[0], inputs["input_ids"].shape[1] + new_tokens):
        raise RuntimeError(f"Expected exactly {new_tokens} new tokens per input, got output shape {output.shape}.")
    return output, streamer


def verify_decode_kernel(model: torch.nn.Module, model_kind: str, seed: int) -> None:
    expected, probe_length = {
        "aqlm": ("Code2x8MatVec", 8),
        "awq": ("gemm_forward_4bit_cuda_", 1),
        "crvq": ("_matvec", 8),
    }[model_kind]
    if model_kind == "awq":
        from awq.modules.linear.gemm import awq_ext

        if not callable(getattr(awq_ext, "gemm_forward_cuda", None)):
            raise RuntimeError("AWQ-GEMM/CUDA requires a loadable awq_ext.gemm_forward_cuda.")

    # A one-token AWQ prompt and its decode both exercise the single-token GEMM
    # without requiring a larger fused KV cache than the workload itself.
    probe_inputs = make_inputs(model, batch_size=1, input_length=probe_length, seed=seed + 1)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as profiler:
        probe_output, _ = generate_once(model, probe_inputs, new_tokens=2)
        synchronize()
    del probe_output, probe_inputs

    observed = sorted({
        event.name for event in profiler.events() if event.device_type == DeviceType.CUDA
    })
    if model_kind == "awq":
        found = any(name.removeprefix("void ").startswith(expected) for name in observed)
    else:
        found = any(name == expected or f"{expected}<" in name for name in observed)
    if not found:
        raise RuntimeError(f"{model_kind.upper()} decode expected CUDA kernel {expected}. Observed: {observed}")


def load_model(
    model_path: Path, model_kind: str, attn_implementation: str, *, max_seq_len: int, awq_fuse_layers: bool = True
) -> torch.nn.Module:
    if model_kind == "awq":
        from awq import AutoAWQForCausalLM

        model = AutoAWQForCausalLM.from_quantized(
            str(model_path),
            max_seq_len=max_seq_len,
            batch_size=1,
            fuse_layers=awq_fuse_layers,
            torch_dtype=torch.float16,
            device_map={"": 0},
            attn_implementation=attn_implementation,
            trust_remote_code=False,
            local_files_only=True,
        ).model
        if awq_fuse_layers:
            from awq.modules.fused import attn as awq_fused_attention
            from awq.modules.fused.model import LlamaLikeModel
            from flash_attn import flash_attn_with_kvcache

            if not isinstance(model.model, LlamaLikeModel):
                raise RuntimeError("NE AWQ benchmark requires AutoAWQ's fused Qwen2 decoder.")

            # AutoAWQ 0.2.9 already writes the current token to the cache. Read
            # that cache directly so the FlashAttention call does not append it again.
            def read_awq_kv_cache(*, k, v, **kwargs):
                return flash_attn_with_kvcache(**kwargs)

            awq_fused_attention.flash_attn_with_kvcache = read_awq_kv_cache
        return model.eval()

    return AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map={"": 0},
        attn_implementation=attn_implementation,
        local_files_only=True,
        low_cpu_mem_usage=True,
        trust_remote_code=model_kind == "crvq",
    ).eval()


def write_result(path: Path, result: dict[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(f"Result written to: {path}")


def main() -> None:
    args = parse_args()
    model_path = validate_args(args)
    default_label = args.model_kind
    if args.model_kind == "awq":
        default_label = "AWQ-4bit-Opt" if args.awq_fuse_layers else "AWQ-4bit"

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Run the benchmark on a GPU node.")
    torch.cuda.set_device(0)
    torch.manual_seed(args.seed)

    monitor = NvidiaSmiProcessMonitor(args.nvml_poll_interval)
    latency_samples: list[dict[str, int | float]] = []

    try:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        load_started = time.perf_counter()
        model = load_model(
            model_path,
            args.model_kind,
            args.attn_implementation,
            max_seq_len=args.input_length + args.new_tokens,
            awq_fuse_layers=args.awq_fuse_layers,
        )
        synchronize()
        load_seconds = time.perf_counter() - load_started
        load_peak_allocated = torch.cuda.max_memory_allocated()
        load_peak_reserved = torch.cuda.max_memory_reserved()

        if args.model_kind in ("awq", "aqlm", "crvq"):
            verify_decode_kernel(model, args.model_kind, args.seed)

        inputs = make_inputs(model, args.batch_size, args.input_length, args.seed)

        for _ in range(args.warmup_iters):
            output, _ = generate_once(model, inputs, args.new_tokens)
            synchronize()
            del output

        gc.collect()
        torch.cuda.empty_cache()
        synchronize()
        steady_allocated = torch.cuda.memory_allocated()
        steady_reserved = torch.cuda.memory_reserved()
        torch.cuda.reset_peak_memory_stats()

        # Start process memory sampling after loading, kernel checks, and warmup.
        monitor.start()
        time.sleep(args.nvml_poll_interval)
        benchmark_started = time.perf_counter()
        for request_index in range(1, args.benchmark_iters + 1):
            output, streamer = generate_once(model, inputs, args.new_tokens)
            synchronize()
            latency_samples.append({"request_index": request_index, **streamer.sample(args.new_tokens)})
            del output
        benchmark_seconds = time.perf_counter() - benchmark_started

        inference_peak_allocated = torch.cuda.max_memory_allocated()
        inference_peak_reserved = torch.cuda.max_memory_reserved()
        total_generated_tokens = args.batch_size * args.new_tokens * args.benchmark_iters

        result = {
            "model_kind": args.model_kind,
            "model_label": args.model_label or default_label,
            "model_path": str(model_path),
            "source_model_path": args.source_model_path,
            "environment": {
                "python_executable": os.sys.executable,
                "torch": torch.__version__,
                "transformers": package_version("transformers"),
                "aqlm": package_version("aqlm"),
                "autoawq": package_version("autoawq"),
                "flash_attn": package_version("flash-attn"),
                "triton": package_version("triton"),
                "cuda_runtime": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0),
            },
            "configuration": {
                "batch_size": args.batch_size,
                "input_length": args.input_length,
                "new_tokens": args.new_tokens,
                "warmup_iters": args.warmup_iters,
                "benchmark_iters": args.benchmark_iters,
                "dtype": args.dtype,
                "attn_implementation": args.attn_implementation,
                "use_cache": True,
                "do_sample": False,
                "num_beams": 1,
                "seed": args.seed,
                "stream_output": True,
            },
            "timing": {
                "load_seconds": round(load_seconds, 4),
                "benchmark_seconds": round(benchmark_seconds, 4),
                "generated_tokens_per_second": round(total_generated_tokens / benchmark_seconds, 4),
            },
            "gpu_memory": {
                "load_peak_allocated_mib": mib(load_peak_allocated),
                "load_peak_reserved_mib": mib(load_peak_reserved),
                "steady_allocated_mib": mib(steady_allocated),
                "steady_reserved_mib": mib(steady_reserved),
                "inference_peak_allocated_mib": mib(inference_peak_allocated),
                "inference_peak_reserved_mib": mib(inference_peak_reserved),
                "inference_incremental_allocated_mib": mib(inference_peak_allocated - steady_allocated),
                "inference_peak_allocated_gib": gib(inference_peak_allocated),
                "nvidia_smi_inference_peak_mib": monitor.peak_mib,
                "nvidia_smi_error": monitor.error,
            },
        }
    finally:
        monitor.stop()

    # Capture the final monitor sample after it has stopped.
    result["gpu_memory"]["nvidia_smi_inference_peak_mib"] = monitor.peak_mib
    result["gpu_memory"]["nvidia_smi_error"] = monitor.error

    latency_result = {
        key: result[key] for key in ("model_kind", "model_label", "model_path", "environment", "configuration")
    }
    if args.model_kind == "awq":
        result["awq_fuse_layers"] = args.awq_fuse_layers
        latency_result["awq_fuse_layers"] = args.awq_fuse_layers
    latency_result["timing_scope"] = "hf_generate_start_to_output_token_on_host"
    latency_result["e2e_throughput_tokens_per_second"] = result["timing"]["generated_tokens_per_second"]
    latency_result["requests"] = latency_samples
    write_result(args.latency_output_json, latency_result)

    print(json.dumps(result, indent=2, ensure_ascii=False))
    write_result(args.memory_output_json, result)


if __name__ == "__main__":
    main()
