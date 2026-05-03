"""Single-layer real-sized smoke test for the post-gate-regret path.

Uses 340M-equivalent shapes (head_dim=128, num_heads=6, expand_v=2, T=2048)
to verify forward+backward fits and runs at acceptable speed.
"""
import time

import torch

from fla.layers.gated_deltanet import GatedDeltaNet


torch.manual_seed(11)
device = "cuda"
dtype = torch.bfloat16

# 340M-config-equivalent layer.
B, T, H = 2, 2048, 6
hidden_size = 1024
head_dim = 128
expand_v = 2

layer = GatedDeltaNet(
    hidden_size=hidden_size,
    head_dim=head_dim,
    num_heads=H,
    num_v_heads=H,
    expand_v=expand_v,
    use_short_conv=True,
    use_gate=True,
    layer_idx=0,
    use_osgm=True,
    osgm_eta=1.0,
    osgm_use_denominator=False,
    osgm_d_min=0.0,
    osgm_d_max=1e9,
    osgm_decay_mode="data_dependent",
    osgm_post_gate_regret=True,
    osgm_post_gate_regret_chunk_size=64,
).to(device).to(dtype)

x = torch.randn(B, T, hidden_size, device=device, dtype=dtype, requires_grad=True)

# Warmup
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
o, _, _ = layer(x)
loss = o.float().square().sum()
loss.backward()

torch.cuda.synchronize()
print(f"warmup: loss={loss.item():.4e}, finite={torch.isfinite(loss).item()}")
print(f"  peak mem: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

# Speed test: 5 forward+backward
torch.cuda.synchronize()
t0 = time.time()
for _ in range(5):
    layer.zero_grad()
    if x.grad is not None: x.grad = None
    o, _, _ = layer(x)
    loss = o.float().square().sum()
    loss.backward()
torch.cuda.synchronize()
dt = (time.time() - t0) / 5
print(f"avg fwd+bwd: {dt*1000:.1f} ms")
print(f"  peak mem after speed test: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

# Compare: same shape but with osgm_post_gate_regret=False (uses chunk kernel)
print("\n--- Comparison: existing chunk kernel (gate_aware=True) ---")
del layer, o, loss
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

layer2 = GatedDeltaNet(
    hidden_size=hidden_size,
    head_dim=head_dim,
    num_heads=H,
    num_v_heads=H,
    expand_v=expand_v,
    use_short_conv=True,
    use_gate=True,
    layer_idx=0,
    use_osgm=True,
    osgm_eta=1.0,
    osgm_use_denominator=False,
    osgm_d_min=0.0,
    osgm_d_max=1e9,
    osgm_decay_mode="data_dependent",
    gate_aware_hypergradient=True,
).to(device).to(dtype)

torch.cuda.synchronize()
t0 = time.time()
for _ in range(5):
    layer2.zero_grad()
    if x.grad is not None: x.grad = None
    o, _, _ = layer2(x)
    loss = o.float().square().sum()
    loss.backward()
torch.cuda.synchronize()
dt2 = (time.time() - t0) / 5
print(f"avg fwd+bwd (gate_aware): {dt2*1000:.1f} ms")
print(f"  peak mem: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
print(f"\nslowdown ratio: {dt/dt2:.2f}x")
