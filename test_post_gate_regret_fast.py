"""Validate the fast beta-aware post-gate OS-GDN chunk path.

This compares the Triton phase1 + chunk-state implementation against the
pure-PyTorch post_gate_regret_recurrence reference for the production
``decay_mode='data_dependent'`` setting.
"""

import torch
import torch.nn.functional as F

from fla.ops.os_gated_delta_rule.chunk import chunk_os_gated_delta_rule
from fla.ops.os_gated_delta_rule.post_gate_regret import post_gate_regret_recurrence


torch.manual_seed(2026)
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float32

B, T, H, K, V = 2, 64, 3, 16, 32


def _leaf_normalized(shape):
    x = torch.randn(*shape, device=device, dtype=dtype)
    return F.normalize(x, p=2, dim=-1, eps=1e-6).detach().requires_grad_(True)


def make_inputs():
    q = _leaf_normalized((B, T, H, K))
    k = _leaf_normalized((B, T, H, K))
    v = (torch.randn(B, T, H, V, device=device, dtype=dtype) * 0.1).requires_grad_(True)
    g = (-torch.rand(B, T, H, device=device, dtype=dtype) * 0.2).requires_grad_(True)
    beta = torch.sigmoid(torch.randn(B, T, H, device=device, dtype=dtype)).detach().requires_grad_(True)
    g_decay = (-torch.rand(B, T, H, device=device, dtype=dtype) * 0.1).requires_grad_(True)
    return q, k, v, g, beta, g_decay


def clone_inputs(xs):
    return [x.detach().clone().requires_grad_(True) for x in xs]


q1, k1, v1, g1, b1, gd1 = make_inputs()
q2, k2, v2, g2, b2, gd2 = clone_inputs((q1, k1, v1, g1, b1, gd1))
initial_d = torch.ones(B, H, K, device=device, dtype=dtype)

# Reference: autograd through the readable recurrence.
o_ref, (S_ref, d_ref) = post_gate_regret_recurrence(
    q1, k1, v1, g1, b1,
    eta=1.0,
    initial_state=(None, initial_d),
    output_final_state=True,
    decay_mode="data_dependent",
    g_decay=gd1,
    d_min=0.0,
    d_max=1e9,
    chunk_size=64,
    use_checkpoint=False,
    use_compile=False,
)
loss_ref = o_ref.square().sum() + 0.1 * S_ref.square().sum() + 0.1 * d_ref.square().sum()
loss_ref.backward()

# Fast path: beta-aware phase1 Triton kernel plus OS-GDN chunk kernels.
o_fast, (S_fast, d_fast) = chunk_os_gated_delta_rule(
    q2, k2, v2, g2, b2,
    eta=1.0,
    initial_state=(None, initial_d),
    output_final_state=True,
    use_qk_l2norm_in_kernel=False,
    use_denominator=False,
    d_min=0.0,
    d_max=1e9,
    decay_mode="data_dependent",
    g_decay=gd2,
    post_gate_regret_beta_aware=True,
)
loss_fast = o_fast.square().sum() + 0.1 * S_fast.square().sum() + 0.1 * d_fast.square().sum()
loss_fast.backward()

print("forward diffs")
print(f"  o={((o_ref - o_fast).abs().max().item()):.3e}")
print(f"  S={((S_ref - S_fast).abs().max().item()):.3e}")
print(f"  d={((d_ref - d_fast).abs().max().item()):.3e}")

max_grad = 0.0
for name, a, b in (
    ("q", q1, q2),
    ("k", k1, k2),
    ("v", v1, v2),
    ("g", g1, g2),
    ("beta", b1, b2),
    ("g_decay", gd1, gd2),
):
    diff = (a.grad - b.grad).abs().max().item()
    max_grad = max(max_grad, diff)
    print(f"  grad {name}={diff:.3e}")

assert (o_ref - o_fast).abs().max().item() < 2e-4
assert (S_ref - S_fast).abs().max().item() < 2e-4
# The WY/chunk phase1 computes chunk boundary states through a triangular solve,
# so the final d state is slightly less bitwise-identical than the per-token
# state/output path while remaining within fp32 kernel tolerance.
assert (d_ref - d_fast).abs().max().item() < 3e-4
assert max_grad < 5e-4
print("FAST POST-GATE REGRET CHECK PASSED")
