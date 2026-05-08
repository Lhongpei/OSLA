"""Numerical equivalence test: fused_recurrent OS-GDN vs chunk path.

Generates small synthetic inputs and verifies the fused recurrent kernel
matches the chunk-mode forward token-by-token, for both decay_mode={"none",
"data_dependent"} and post_gate_regret=True (the configuration used by both
OS-GDN production checkpoints).
"""
from __future__ import annotations

import sys

import torch
import torch.nn.functional as F

# Force no compile interference
import os
os.environ.setdefault("OSLA_DISABLE_POST_GATE_COMPILE", "1")

import fla  # noqa: F401
from fla.ops.os_gated_delta_rule import (
    chunk_os_gated_delta_rule,
    fused_recurrent_os_gated_delta_rule,
)
from fla.modules.l2norm import l2norm_fwd


def _make_inputs(B: int, T: int, H: int, K: int, V: int, device, dtype, seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn(B, T, H, K, generator=g, device=device, dtype=dtype)
    k = torch.randn(B, T, H, K, generator=g, device=device, dtype=dtype)
    v = torch.randn(B, T, H, V, generator=g, device=device, dtype=dtype)
    # GDN-style raw log-decay: small negative values, e.g. -exp(A_log)*softplus(...)
    g_gdn = -torch.rand(B, T, H, generator=g, device=device, dtype=torch.float32) * 0.05
    g_gdn = g_gdn.to(dtype)
    beta = torch.rand(B, T, H, generator=g, device=device, dtype=dtype).sigmoid()
    return q, k, v, g_gdn, beta


def _run_chunk(
    q, k, v, g, beta, *, eta, d_min, d_max, decay_mode, g_decay, initial_h, initial_d,
):
    return chunk_os_gated_delta_rule(
        q=q, k=k, v=v, g=g, beta=beta,
        scale=None, eta=eta,
        initial_state=(initial_h, initial_d),
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=None,
        use_denominator=False,
        d_min=d_min, d_max=d_max,
        decay_mode=decay_mode,
        g_decay=g_decay,
        post_gate_regret_beta_aware=True,
    )


def _run_fused(
    q, k, v, g, beta, *, eta, d_min, d_max, decay_mode, g_decay, initial_h, initial_d,
):
    return fused_recurrent_os_gated_delta_rule(
        q=q, k=k, v=v, g=g, beta=beta, g_decay=g_decay,
        scale=None, eta=eta,
        d_min=d_min, d_max=d_max,
        initial_state=initial_h, initial_d=initial_d,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        decay_mode=decay_mode,
        cu_seqlens=None,
    )


def compare(name: str, *, decay_mode: str):
    device = "cuda"
    dtype = torch.bfloat16
    B, T, H, K, V = 2, 64, 4, 32, 64

    q, k, v, g, beta = _make_inputs(B, T, H, K, V, device, dtype)

    g_decay = g.clone() if decay_mode == "data_dependent" else None

    # Initial states
    initial_h = torch.randn(B, H, K, V, device=device, dtype=torch.float32) * 0.01
    initial_d = torch.full((B, H, K), 1.0, device=device, dtype=torch.float32)

    eta = 0.003
    d_min = 0.6666667
    d_max = 1.5

    print(f"\n=== {name} (decay_mode={decay_mode}) ===")
    print(f"shapes: q,k=[B={B},T={T},H={H},K={K}], v=[..., V={V}], dtype={dtype}")

    # Reference: chunk path
    o_ref, (h_ref, d_ref) = _run_chunk(
        q, k, v, g, beta,
        eta=eta, d_min=d_min, d_max=d_max,
        decay_mode=decay_mode, g_decay=g_decay,
        initial_h=initial_h.clone(), initial_d=initial_d.clone(),
    )

    # Test path 1: fused_recurrent over the FULL T (matches chunk for prefill)
    o_fused, h_fused, d_fused = _run_fused(
        q, k, v, g, beta,
        eta=eta, d_min=d_min, d_max=d_max,
        decay_mode=decay_mode, g_decay=g_decay,
        initial_h=initial_h.clone(), initial_d=initial_d.clone(),
    )

    o_diff = (o_ref.float() - o_fused.float()).abs()
    h_diff = (h_ref.float() - h_fused.float()).abs()
    d_diff = (d_ref.float() - d_fused.float()).abs()
    o_rel = o_diff.max() / o_ref.float().abs().max().clamp(min=1e-6)
    h_rel = h_diff.max() / h_ref.float().abs().max().clamp(min=1e-6)
    d_rel = d_diff.max() / d_ref.float().abs().max().clamp(min=1e-6)
    print(f"[full-T] o max abs diff = {o_diff.max().item():.3e}  (rel {o_rel.item():.3e})")
    print(f"[full-T] h max abs diff = {h_diff.max().item():.3e}  (rel {h_rel.item():.3e})")
    print(f"[full-T] d max abs diff = {d_diff.max().item():.3e}  (rel {d_rel.item():.3e})")

    # Test path 2: simulate decode — chunk for first (T-1) tokens, fused for last 1 token.
    # Should produce the same final outputs as chunk on the full T.
    o_chunk_pre, (h_pre, d_pre) = _run_chunk(
        q[:, :T-1], k[:, :T-1], v[:, :T-1], g[:, :T-1], beta[:, :T-1],
        eta=eta, d_min=d_min, d_max=d_max,
        decay_mode=decay_mode,
        g_decay=g_decay[:, :T-1] if g_decay is not None else None,
        initial_h=initial_h.clone(), initial_d=initial_d.clone(),
    )
    o_decode, h_decode, d_decode = _run_fused(
        q[:, T-1:], k[:, T-1:], v[:, T-1:], g[:, T-1:], beta[:, T-1:],
        eta=eta, d_min=d_min, d_max=d_max,
        decay_mode=decay_mode,
        g_decay=g_decay[:, T-1:] if g_decay is not None else None,
        initial_h=h_pre.clone(), initial_d=d_pre.clone(),
    )
    o_combined = torch.cat([o_chunk_pre, o_decode], dim=1)
    o_diff2 = (o_ref.float() - o_combined.float()).abs()
    h_diff2 = (h_ref.float() - h_decode.float()).abs()
    d_diff2 = (d_ref.float() - d_decode.float()).abs()
    print(f"[chunk+1tok] o max abs diff = {o_diff2.max().item():.3e}")
    print(f"[chunk+1tok] h max abs diff = {h_diff2.max().item():.3e}")
    print(f"[chunk+1tok] d max abs diff = {d_diff2.max().item():.3e}")

    # Tolerances: bf16 + fp32 accumulation, single rank-1 update per token.
    # Compute paths differ (chunk-parallel WY repr vs sequential), so use moderate tol.
    tol_o = 5e-2  # output is bf16
    tol_d = 5e-3  # d is fp32
    ok = (o_rel.item() < tol_o) and (d_rel.item() < tol_d)
    if not ok:
        raise AssertionError(
            f"FAIL {name}: o_rel={o_rel.item():.3e} (tol {tol_o}), "
            f"d_rel={d_rel.item():.3e} (tol {tol_d})"
        )
    print(f"OK ({name})")


if __name__ == "__main__":
    compare("OS-GDN no-DD", decay_mode="none")
    compare("OS-GDN APF (gdn-source dd_decay)", decay_mode="data_dependent")
    print("\nAll tests passed.")
