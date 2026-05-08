#!/usr/bin/env python3
"""Lightweight inference benchmark for 340M recurrent variants.

Measures prefill latency, autoregressive decode latency, throughput, and the
persistent recurrent KV/state cache size. The benchmark intentionally uses
random token ids so it does not depend on datasets or prompt files.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM


DTYPES = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


@dataclass
class BenchResult:
    label: str
    path: str
    prompt_len: int
    decode_len: int
    batch_size: int
    prefill_ms: float
    decode_ms: float
    total_ms: float
    prefill_toks_s: float
    decode_toks_s: float
    total_toks_s: float
    kv_state_bf16_mib: float
    kv_state_fp32_mib: float
    peak_extra_mib: float | None


def parse_model_spec(spec: str) -> tuple[str, str]:
    if "=" not in spec:
        path = spec
        return Path(path).name, path
    label, path = spec.split("=", 1)
    return label.strip(), path.strip()


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def median(values: list[float]) -> float:
    return float(statistics.median(values))


def get_attr(obj: Any, names: tuple[str, ...], default: Any = None) -> Any:
    for name in names:
        if hasattr(obj, name):
            value = getattr(obj, name)
            if value is not None:
                return value
    return default


def recurrent_state_elements(model: torch.nn.Module) -> int:
    layers = list(getattr(model.model, "layers"))
    if not layers:
        return 0

    total = 0
    for layer in layers:
        attn = layer.attn
        heads = int(get_attr(attn, ("num_heads", "num_kv_heads")))
        key_dim = int(get_attr(attn, ("head_k_dim", "head_dim")))
        value_dim = int(get_attr(attn, ("head_v_dim",), key_dim))
        total += heads * key_dim * value_dim

        if bool(get_attr(attn, ("use_osgm",), False)):
            total += heads * key_dim
    return total


def fallback_state_elements(config: Any) -> int:
    layers = int(config.num_hidden_layers)
    heads = int(config.num_heads)
    model_type = str(getattr(config, "model_type", ""))
    expand_v = float(getattr(config, "expand_v", 1.0))

    if hasattr(config, "head_dim") and config.head_dim is not None:
        key_dim = int(config.head_dim)
        value_dim = int(key_dim * expand_v)
    else:
        expand_k = float(getattr(config, "expand_k", 1.0))
        key_dim = int(config.hidden_size * expand_k / heads)
        value_dim = int(config.hidden_size * expand_v / heads)

    total = layers * heads * key_dim * value_dim
    if bool(getattr(config, "use_osgm", False)) or model_type.startswith("os_"):
        total += layers * heads * key_dim
    return total


def make_inputs(config: Any, batch_size: int, prompt_len: int, device: torch.device) -> torch.Tensor:
    vocab_size = int(getattr(config, "vocab_size", 32000))
    # Avoid special low ids where possible, while still staying inside vocab.
    low = min(16, max(0, vocab_size - 1))
    return torch.randint(low=low, high=vocab_size, size=(batch_size, prompt_len), device=device)


@torch.inference_mode()
def run_one(
    label: str,
    path: str,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> BenchResult:
    try:
        import fla  # noqa: F401
    except Exception:
        # Some converted checkpoints register their classes through `fla`.
        # If transformers can load the model without it, continue.
        pass

    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
    ).to(device)
    model.eval()
    model.config.use_cache = True

    state_elements = recurrent_state_elements(model)
    if state_elements == 0:
        state_elements = fallback_state_elements(model.config)
    kv_state_bf16_mib = state_elements * 2 / 1024**2
    kv_state_fp32_mib = state_elements * 4 / 1024**2

    input_ids = make_inputs(model.config, args.batch_size, args.prompt_len, device)

    def step() -> tuple[float, float]:
        sync(device)
        t0 = time.perf_counter()
        out = model(input_ids=input_ids, use_cache=True)
        sync(device)
        t1 = time.perf_counter()

        past = out.past_key_values
        next_ids = out.logits[:, -1:].argmax(dim=-1)
        for _ in range(args.decode_len):
            out = model(input_ids=next_ids, past_key_values=past, use_cache=True)
            past = out.past_key_values
            next_ids = out.logits[:, -1:].argmax(dim=-1)
        sync(device)
        t2 = time.perf_counter()
        return (t1 - t0) * 1000.0, (t2 - t1) * 1000.0

    for _ in range(args.warmup):
        step()

    if device.type == "cuda":
        base_alloc = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
    else:
        base_alloc = 0

    prefill_ms_values = []
    decode_ms_values = []
    for _ in range(args.repeats):
        p_ms, d_ms = step()
        prefill_ms_values.append(p_ms)
        decode_ms_values.append(d_ms)

    peak_extra_mib = None
    if device.type == "cuda":
        peak_extra_mib = max(0, torch.cuda.max_memory_allocated(device) - base_alloc) / 1024**2

    prefill_ms = median(prefill_ms_values)
    decode_ms = median(decode_ms_values)
    total_ms = prefill_ms + decode_ms
    prefill_tokens = args.batch_size * args.prompt_len
    decode_tokens = args.batch_size * args.decode_len
    total_tokens = prefill_tokens + decode_tokens

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return BenchResult(
        label=label,
        path=path,
        prompt_len=args.prompt_len,
        decode_len=args.decode_len,
        batch_size=args.batch_size,
        prefill_ms=prefill_ms,
        decode_ms=decode_ms,
        total_ms=total_ms,
        prefill_toks_s=prefill_tokens / (prefill_ms / 1000.0),
        decode_toks_s=decode_tokens / (decode_ms / 1000.0),
        total_toks_s=total_tokens / (total_ms / 1000.0),
        kv_state_bf16_mib=kv_state_bf16_mib,
        kv_state_fp32_mib=kv_state_fp32_mib,
        peak_extra_mib=peak_extra_mib,
    )


def write_outputs(results: list[BenchResult], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = [r.__dict__ for r in results]
    if output.suffix == ".json":
        output.write_text(json.dumps(rows, indent=2) + "\n")
        return

    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", required=True, help="label=/path/to/checkpoint")
    parser.add_argument("--output", default="experiments/osla_340M/reports/inference_bench_340m.csv")
    parser.add_argument("--prompt-len", type=int, default=2048)
    parser.add_argument("--decode-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="bf16")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = DTYPES[args.dtype]
    results = []
    for spec in args.model:
        label, path = parse_model_spec(spec)
        print(f"Benchmarking {label}: {path}", flush=True)
        results.append(run_one(label, path, args, device, dtype))
        write_outputs(results, Path(args.output))

    for row in results:
        print(
            f"{row.label}: total={row.total_toks_s:.1f} tok/s, "
            f"prefill={row.prefill_ms:.1f} ms, decode={row.decode_ms:.1f} ms, "
            f"KV={row.kv_state_bf16_mib:.3f} MiB bf16 / {row.kv_state_fp32_mib:.3f} MiB fp32",
            flush=True,
        )


if __name__ == "__main__":
    main()
