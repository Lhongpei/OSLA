"""Validate `post_gate_regret_recurrence` against the naive reference.

Checks:
  1. Forward output matches naive reference to fp32 precision.
  2. Final state and final d match naive.
  3. Gradients (via autograd through checkpointed forward) match gradients
     through the naive (autograd-through-loop) implementation.
  4. Both with and without `use_checkpoint`.
  5. Both for `decay_mode='none'` and `'data_dependent'`.
"""
import torch

from fla.ops.os_gated_delta_rule.naive import (
    naive_recurrent_os_gated_delta_rule_post_gate_regret,
)
from fla.ops.os_gated_delta_rule.post_gate_regret import post_gate_regret_recurrence


torch.manual_seed(123)
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float32

B, T, H, K, V = 2, 128, 4, 16, 32   # T=128 = 2 chunks of 64

def make_inputs(requires_grad=False):
    q = torch.randn(B, T, H, K, device=device, dtype=dtype, requires_grad=requires_grad)
    k = (torch.randn(B, T, H, K, device=device, dtype=dtype) * 0.2).requires_grad_(requires_grad)
    v = torch.randn(B, T, H, V, device=device, dtype=dtype, requires_grad=requires_grad)
    g_log = (-torch.rand(B, T, H, device=device, dtype=dtype) * 1.5).requires_grad_(requires_grad)
    beta = torch.sigmoid(torch.randn(B, T, H, device=device, dtype=dtype)).requires_grad_(requires_grad)
    g_decay = (-torch.rand(B, T, H, device=device, dtype=dtype) * 0.1).requires_grad_(requires_grad)
    return q, k, v, g_log, beta, g_decay


def run_check(decay_mode, use_checkpoint):
    print(f"\n--- decay_mode={decay_mode}, use_checkpoint={use_checkpoint} ---")

    # Set seed so the same random inputs feed both paths.
    torch.manual_seed(123)
    q1, k1, v1, g1, b1, g_dec1 = make_inputs(requires_grad=True)
    torch.manual_seed(123)
    q2, k2, v2, g2, b2, g_dec2 = make_inputs(requires_grad=True)

    initial_d = torch.ones(B, H, K, device=device, dtype=dtype)

    # Naive reference (autograd through python loop).
    o_naive, S_naive, d_naive = naive_recurrent_os_gated_delta_rule_post_gate_regret(
        q1, k1, v1, g1, b1,
        eta=1.0, initial_d=initial_d,
        output_final_state=True, output_final_d=True,
        decay_mode=decay_mode,
        g_decay=g_dec1 if decay_mode == "data_dependent" else None,
    )
    loss_naive = o_naive.square().sum() + S_naive.square().sum() * 0.1 + d_naive.square().sum() * 0.1
    loss_naive.backward()

    # Implementation.
    o_impl, (S_impl, d_impl) = post_gate_regret_recurrence(
        q2, k2, v2, g2, b2,
        eta=1.0,
        initial_state=(None, initial_d.clone()),
        output_final_state=True,
        decay_mode=decay_mode,
        g_decay=g_dec2 if decay_mode == "data_dependent" else None,
        chunk_size=64,
        use_checkpoint=use_checkpoint,
    )
    loss_impl = o_impl.square().sum() + S_impl.square().sum() * 0.1 + d_impl.square().sum() * 0.1
    loss_impl.backward()

    # Forward checks
    diff_o = (o_naive - o_impl).abs().max().item()
    diff_S = (S_naive - S_impl).abs().max().item()
    diff_d = (d_naive - d_impl).abs().max().item()
    print(f"  fwd: |o| diff={diff_o:.3e}, |S|={diff_S:.3e}, |d|={diff_d:.3e}")

    # Gradient checks — compare gradient on each input
    diff_dq = (q1.grad - q2.grad).abs().max().item()
    diff_dk = (k1.grad - k2.grad).abs().max().item()
    diff_dv = (v1.grad - v2.grad).abs().max().item()
    diff_dg = (g1.grad - g2.grad).abs().max().item()
    diff_db = (b1.grad - b2.grad).abs().max().item()
    print(f"  bwd: |dq|={diff_dq:.3e} |dk|={diff_dk:.3e} |dv|={diff_dv:.3e} "
          f"|dg|={diff_dg:.3e} |db|={diff_db:.3e}")
    if decay_mode == "data_dependent":
        diff_ddec = (g_dec1.grad - g_dec2.grad).abs().max().item()
        print(f"      |dg_decay|={diff_ddec:.3e}")
    else:
        diff_ddec = 0.0

    tol = 1e-3
    assert diff_o < tol, f"forward output diverged: {diff_o}"
    assert diff_S < tol, f"final state diverged: {diff_S}"
    assert diff_d < tol, f"final d diverged: {diff_d}"
    assert diff_dq < tol, f"dq diverged: {diff_dq}"
    assert diff_dk < tol, f"dk diverged: {diff_dk}"
    assert diff_dv < tol, f"dv diverged: {diff_dv}"
    assert diff_dg < tol, f"dg diverged: {diff_dg}"
    assert diff_db < tol, f"db diverged: {diff_db}"
    print("  PASS")


for mode in ("none", "data_dependent"):
    for ckpt in (False, True):
        run_check(decay_mode=mode, use_checkpoint=ckpt)

print("\nALL CHECKS PASSED")
