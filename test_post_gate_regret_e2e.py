"""End-to-end smoke test: instantiate OSGDNForCausalLM with osgm_post_gate_regret=True
and verify forward+backward runs without errors and gradients propagate.
"""
import torch

from fla.models.os_gated_deltanet import OSGDNConfig, OSGDNForCausalLM


torch.manual_seed(7)
device = "cuda"

# Tiny config so it fits and runs fast.
config = OSGDNConfig(
    hidden_size=64,
    num_heads=2,
    num_v_heads=2,
    num_hidden_layers=2,
    head_dim=32,
    expand_v=2,
    vocab_size=512,
    max_position_embeddings=128,
    use_short_conv=True,
    use_gate=True,
    use_osgm=True,
    osgm_eta=1.0,
    osgm_use_denominator=False,
    osgm_d_min=0.0,
    osgm_d_max=1e9,
    osgm_decay_mode="data_dependent",
    osgm_post_gate_regret=True,                  # NEW
    osgm_post_gate_regret_chunk_size=32,         # smaller chunk for tiny T
    fuse_norm=False,
    fuse_swiglu=False,
    fuse_cross_entropy=False,
    fuse_linear_cross_entropy=False,
)

model = OSGDNForCausalLM(config).to(device).bfloat16()
model.train()

# Print which a_proj got the post-gate-regret config
layer0 = model.model.layers[0].attn
print(f"Layer attn type: {type(layer0).__name__}")
print(f"  use_osgm = {layer0.use_osgm}")
print(f"  osgm_decay_mode = {layer0.osgm_decay_mode}")
print(f"  osgm_post_gate_regret = {layer0.osgm_post_gate_regret}")
print(f"  osgm_post_gate_regret_chunk_size = {layer0.osgm_post_gate_regret_chunk_size}")
print(f"  initial_scale[0,:4] = {layer0.initial_scale.flatten()[:4].tolist()}")  # should be ~1
print(f"  osgm_a_proj.bias[:4] = {layer0.osgm_a_proj.bias[:4].tolist()}  (should be 6.9)")

B, T = 2, 64
input_ids = torch.randint(0, config.vocab_size, (B, T), device=device)
labels = torch.randint(0, config.vocab_size, (B, T), device=device)

print("\nRunning forward+backward...")
out = model(input_ids=input_ids, labels=labels)
print(f"  loss = {out.loss.item():.4f}")
print(f"  loss finite = {torch.isfinite(out.loss).item()}")

out.loss.backward()
total_grad_norm = 0.0
zero_grads = 0
n_params = 0
for name, p in model.named_parameters():
    if p.grad is None:
        continue
    n_params += 1
    g_norm = p.grad.float().norm().item()
    total_grad_norm += g_norm * g_norm
    if g_norm < 1e-12:
        zero_grads += 1
total_grad_norm = total_grad_norm ** 0.5
print(f"  grad norm = {total_grad_norm:.4f}")
print(f"  zero-grad params: {zero_grads}/{n_params}")

# Check key OSGM-specific params got non-zero grads
critical_params = ['initial_scale', 'osgm_a_proj.weight', 'osgm_a_proj.bias',
                   'a_proj.weight', 'A_log', 'dt_bias']
for name, p in model.named_parameters():
    for cp in critical_params:
        if cp in name and p.grad is not None:
            g = p.grad.float().norm().item()
            print(f"  ✓ {name}: grad_norm={g:.4e}")
            break

print("\nE2E TEST PASSED")
