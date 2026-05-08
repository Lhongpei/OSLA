"""Sanity checks for the beta-aware post-gate OS-GDN reference."""
import torch

from fla.ops.gated_delta_rule.naive import naive_recurrent_gated_delta_rule
from fla.ops.os_gated_delta_rule.naive import (
    naive_recurrent_os_gated_delta_rule_post_gate_regret,
)

torch.manual_seed(42)

B, T, H, K, V = 2, 16, 4, 8, 16
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float32

q = torch.randn(B, T, H, K, device=device, dtype=dtype)
k = torch.randn(B, T, H, K, device=device, dtype=dtype) * 0.1
v = torch.randn(B, T, H, V, device=device, dtype=dtype)
beta = torch.sigmoid(torch.randn(B, T, H, device=device, dtype=dtype))
g_log = -torch.rand(B, T, H, device=device, dtype=dtype)
initial_d = torch.ones(B, H, K, device=device, dtype=dtype)

# 1) With d fixed at one, the repaired path should reduce exactly to vanilla
# GatedDeltaNet. eta=0 freezes d, so every write key is raw k.
o_os, S_os, d_os = naive_recurrent_os_gated_delta_rule_post_gate_regret(
    q, k, v, g_log, beta,
    eta=0.0, initial_d=initial_d,
    output_final_state=True, output_final_d=True,
    decay_mode="none",
)
o_gdn, S_gdn = naive_recurrent_gated_delta_rule(
    q, k, v, beta=beta, g=g_log,
    output_final_state=True,
)
diff_o = (o_os - o_gdn).abs().max().item()
diff_S = (S_os - S_gdn).abs().max().item()
diff_d = (d_os - initial_d).abs().max().item()
print("[check 1: eta=0, d=1 -> vanilla GDN]")
print(f"  max |o_os - o_gdn| = {diff_o:.3e}")
print(f"  max |S_os - S_gdn| = {diff_S:.3e}")
print(f"  max |d_final - d0| = {diff_d:.3e}")
assert diff_o < 1e-5 and diff_S < 1e-5 and diff_d < 1e-6

# 2) First-token d update must be beta-aware:
# d_next = d + eta * beta * (1 - beta * <d,k^2>) * k^2.
eta = 0.7
_, _, d_after_one = naive_recurrent_os_gated_delta_rule_post_gate_regret(
    q[:, :1], k[:, :1], v[:, :1], g_log[:, :1], beta[:, :1],
    eta=eta, initial_d=initial_d, output_final_d=True, decay_mode="none",
)
k_sq = k[:, 0] * k[:, 0]
inner = (initial_d * k_sq).sum(-1, keepdim=True)
beta0 = beta[:, 0, :, None]
expected = initial_d + eta * beta0 * (1.0 - beta0 * inner) * k_sq
diff_update = (d_after_one - expected).abs().max().item()
print("[check 2: beta-aware first d update]")
print(f"  max |d_after_one - expected| = {diff_update:.3e}")
assert diff_update < 1e-6

# 3) Random gated inputs should stay finite when d is allowed to evolve.
o_g, S_g, d_g = naive_recurrent_os_gated_delta_rule_post_gate_regret(
    q, k, v, g_log, beta,
    eta=1.0, initial_d=initial_d,
    output_final_state=True, output_final_d=True,
    decay_mode="none",
)
print(f"[check 3: finite] o={torch.isfinite(o_g).all().item()} "
      f"S={torch.isfinite(S_g).all().item()} d={torch.isfinite(d_g).all().item()} "
      f"o.std={o_g.std().item():.4f}")
assert torch.isfinite(o_g).all() and torch.isfinite(S_g).all() and torch.isfinite(d_g).all()

print("\nALL CHECKS PASSED")
