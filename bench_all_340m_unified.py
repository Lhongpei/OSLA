"""Unified single-GPU benchmark for all 340M variants.

Runs 9 variants back-to-back on the same GPU with identical protocol.
Each variant is loaded fresh, warmed up twice, then timed 5 times.
Output: a CSV summary + a markdown table on stdout.
"""
from __future__ import annotations
import argparse, csv, statistics, time, json, os
from pathlib import Path
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM
import fla  # noqa


@dataclass
class Result:
    label: str
    family: str
    path: str
    prefill_ms: float
    decode_ms: float
    total_ms: float
    tokens_sec: float
    kv_mib_bf16: float


def state_mib(model, dtype_bytes=2):
    layers = list(getattr(model.model, "layers"))
    total = 0
    for layer in layers:
        attn = layer.attn
        heads = int(getattr(attn, "num_v_heads", getattr(attn, "num_heads", 0)))
        kdim = int(getattr(attn, "head_k_dim", getattr(attn, "head_dim", 0)))
        vdim = int(getattr(attn, "head_v_dim", kdim))
        total += heads * kdim * vdim
        if bool(getattr(attn, "use_osgm", False)):
            total += heads * kdim
    return total * dtype_bytes / 1024**2


@torch.inference_mode()
def bench_one(label, family, path, prompt_len, decode_len, warmup, repeats, device):
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device).eval()
    model.config.use_cache = True
    kv_mib = state_mib(model, dtype_bytes=2)
    ids = torch.randint(16, 32000, (1, prompt_len), device=device)

    def step():
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model(input_ids=ids, use_cache=True)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        past = out.past_key_values
        nxt = out.logits[:, -1:].argmax(-1)
        for _ in range(decode_len):
            out = model(input_ids=nxt, past_key_values=past, use_cache=True)
            past = out.past_key_values
            nxt = out.logits[:, -1:].argmax(-1)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        return (t1 - t0) * 1000.0, (t2 - t1) * 1000.0

    for _ in range(warmup):
        step()
    pre, dec = [], []
    for _ in range(repeats):
        p, d = step()
        pre.append(p); dec.append(d)
    pre_ms = statistics.median(pre)
    dec_ms = statistics.median(dec)
    tot_tok = prompt_len + decode_len
    tot_ms = pre_ms + dec_ms
    tps = tot_tok / (tot_ms / 1000)

    del model
    torch.cuda.empty_cache()
    return Result(label, family, path, pre_ms, dec_ms, tot_ms, tps, kv_mib)


def main():
    ap = argparse.ArgumentParser()
    default_exp_dir = Path(
        os.environ.get("OSLA_ROOT", str(Path(__file__).resolve().parent))
    ) / "experiments/osla_340M/exp"
    ap.add_argument("--exp-dir", default=str(default_exp_dir))
    ap.add_argument("--prompt-len", type=int, default=2048)
    ap.add_argument("--decode-len", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--output", default="/tmp/bench_all_340m.csv")
    args = ap.parse_args()

    device = torch.device("cuda")

    # Order: matches paper table ordering. Each entry: (label, family, dirname).
    variants = [
        ("DeltaNet",    "DeltaNet",        "deltanet-340M-baseline-4gpu-fair"),
        ("OS-DN",       "DeltaNet",        "deltanet-340M-osla-osgm-chunk-run2"),
        ("OS-DN-APF",   "DeltaNet",        "deltanet-340M-osla-osgm-dd-decay"),
        ("GDN",         "Gated DeltaNet",  "gated-deltanet-340M-baseline-v2"),
        ("OS-GDN",      "Gated DeltaNet",  "os-gdn-post-gate-regret-eta0p003-dmin0p6667-dmax1p5-no-dd-340m-8gpu-65k-fair-20260506"),
        ("OS-GDN-APF",  "Gated DeltaNet",  "os-gdn-post-gate-regret-eta0p003-dmin0p6667-dmax1p5-apf-340m-8gpu-65k-fair-20260506"),
        ("KDA",         "KDA",             "kda-340M-baseline"),
        ("OS-KDA",      "KDA",             "os-kda-340M-detach-phase1-no-dd-eta0p01-dmax1p2-bwdpatch2-full-20260505-185514"),
        ("OS-KDA-APF",  "KDA",             "os-kda-340M-dd-eta0p003-dmin0p667-dmax1p5-4gpu-full-20260506"),
    ]

    results = []
    for label, fam, sub in variants:
        path = os.path.join(args.exp_dir, sub)
        if not os.path.isfile(os.path.join(path, "model.safetensors")):
            print(f"[skip] {label}: no checkpoint at {path}", flush=True)
            continue
        print(f"[run] {label}: {path}", flush=True)
        try:
            r = bench_one(label, fam, path, args.prompt_len, args.decode_len,
                          args.warmup, args.repeats, device)
            print(f"  {label}: tps={r.tokens_sec:.1f}  prefill={r.prefill_ms:.1f}ms"
                  f"  decode={r.decode_ms:.1f}ms  KV={r.kv_mib_bf16:.3f}MiB", flush=True)
            results.append(r)
            # Persist after each model in case a later one crashes.
            with open(args.output, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["label", "family", "path",
                    "prefill_ms", "decode_ms", "total_ms", "tokens_sec", "kv_mib_bf16"])
                w.writeheader()
                for x in results:
                    w.writerow(x.__dict__)
        except Exception as e:
            print(f"  [error] {label}: {type(e).__name__}: {e}", flush=True)

    # Markdown summary
    print("\n=== SUMMARY ===")
    print("| Variant | Family | tokens/sec | prefill ms | decode ms | total ms | KV MiB bf16 |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for r in results:
        print(f"| {r.label} | {r.family} | {r.tokens_sec:.1f} | {r.prefill_ms:.1f} "
              f"| {r.decode_ms:.1f} | {r.total_ms:.1f} | {r.kv_mib_bf16:.3f} |")


if __name__ == "__main__":
    main()
