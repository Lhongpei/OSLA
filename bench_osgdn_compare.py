"""Side-by-side benchmark: OS-GDN with fused vs forced-chunk decode.

Same machine, same GPU, same checkpoint — only the dispatch differs.
Confirms the speedup is attributable to the new fused_recurrent kernel,
not cross-machine variance.
"""
from __future__ import annotations
import argparse, statistics, time, os
import torch
from transformers import AutoModelForCausalLM
import fla  # noqa


def _force_chunk_decode(model):
    """Patch every OS-GDN layer's _osgm_supports_fused_recurrent to return False,
    so decode falls through to chunk (the pre-fix behaviour)."""
    n_patched = 0
    for layer in model.model.layers:
        attn = layer.attn
        if hasattr(attn, "_osgm_supports_fused_recurrent"):
            attn._osgm_supports_fused_recurrent = lambda: False
            n_patched += 1
    return n_patched


def bench(label, model, prompt_len, decode_len, repeats, warmup, device):
    ids = torch.randint(16, 32000, (1, prompt_len), device=device)
    @torch.inference_mode()
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
    tps = tot_tok / ((pre_ms + dec_ms) / 1000)
    print(f"{label}: total={tps:.1f} tok/s  prefill={pre_ms:.1f}ms  decode={dec_ms:.1f}ms")
    return tps, pre_ms, dec_ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--prompt-len", type=int, default=2048)
    ap.add_argument("--decode-len", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()
    dev = torch.device("cuda")
    print(f"Loading {args.ckpt}")
    m = AutoModelForCausalLM.from_pretrained(
        args.ckpt, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(dev).eval()
    m.config.use_cache = True

    # Run 1: fused (default after our fix)
    bench("[FUSED] OS-GDN", m, args.prompt_len, args.decode_len,
          args.repeats, args.warmup, dev)

    # Run 2: forced chunk (pre-fix behaviour)
    n = _force_chunk_decode(m)
    print(f"  patched {n} layers to force chunk decode")
    bench("[CHUNK] OS-GDN", m, args.prompt_len, args.decode_len,
          args.repeats, args.warmup, dev)


if __name__ == "__main__":
    main()
