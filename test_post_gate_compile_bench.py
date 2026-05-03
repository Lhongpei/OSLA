"""Micro-bench: per-iter speedup from torch.compile on _post_gate_chunk_step.

Shapes match a 340M layer (head_dim=128, num_heads=6, expand_v=2). To keep the
benchmark cheap on a contended box, we run ONE compile case (decay_mode=none)
at T scaled small.

Run twice to compare:
  OSLA_DISABLE_POST_GATE_COMPILE=1 python test_post_gate_compile_bench.py
  python test_post_gate_compile_bench.py
"""
import os
import time

import torch

from fla.ops.os_gated_delta_rule.post_gate_regret import (
    post_gate_regret_recurrence,
    _DISABLE_COMPILE,
)


COMPILE = "OFF" if _DISABLE_COMPILE else "ON"
print(f"compile={COMPILE} (set OSLA_DISABLE_POST_GATE_COMPILE=1 to disable)")

device = "cuda"
dtype = torch.bfloat16

# 340M-equivalent shapes; T scaled to keep total runtime sane on busy box.
B, H, K, V = 2, 6, 128, 256
T = 1024            # 16 chunks of 64
N_WARMUP = 2
N_ITER = 8


def one_iter(seed):
    torch.manual_seed(seed)
    q = torch.randn(B, T, H, K, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(B, T, H, K, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(B, T, H, V, device=device, dtype=dtype, requires_grad=True)
    g = torch.zeros(B, T, H, device=device, dtype=torch.float32, requires_grad=True)
    beta = torch.zeros(B, T, H, device=device, dtype=dtype, requires_grad=True)
    with torch.no_grad():
        k.mul_(0.2)
        g.uniform_(-1.5, 0.0)
        beta.uniform_(0.2, 0.8)
    o, _ = post_gate_regret_recurrence(
        q, k, v, g, beta,
        eta=1.0, output_final_state=False,
        decay_mode="none",
        chunk_size=64,
    )
    o.float().square().sum().backward()


print(f"warmup: {N_WARMUP} iter (first compiles, may take >30s on busy GPU)…")
torch.cuda.synchronize()
t_warm0 = time.time()
for i in range(N_WARMUP):
    one_iter(seed=i)
torch.cuda.synchronize()
warm_ms = (time.time() - t_warm0) * 1000
print(f"  warmup total: {warm_ms:.0f} ms")

torch.cuda.synchronize()
t0 = time.time()
for i in range(N_ITER):
    one_iter(seed=i + 100)
torch.cuda.synchronize()
per_iter_ms = (time.time() - t0) / N_ITER * 1000
print(f"per fwd+bwd ({N_ITER} iter avg, T={T}): {per_iter_ms:.1f} ms")

print("DONE")
