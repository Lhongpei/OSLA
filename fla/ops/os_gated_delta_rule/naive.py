# -*- coding: utf-8 -*-
"""
Naive PyTorch reference for OSGM + GatedDeltaNet (per-head scalar gate on state).

Semantics:
  - OSGM "phase 1": compute d_t online from k (same as os_delta_rule).
    d_0 = initial_d (or zeros)
    stored d[t] = d_t (pre-update, used for state absorption at token t)
    d_{t+1} = clip(d_t + eta * grad * k_t^2,  [d_min, d_max])
      grad = (1 - <d_t, k_t^2>)                     if not use_denominator
           = (1 - <d_t, k_t^2>) / (<k_t^2> + 1e-5)  if use_denominator

  - State recurrence (GDN-style per-head scalar forget gate + OSGM preconditioned key):
    kw_t = k_t * d_t                              (absorbed key, per-token)
    S_t  = exp(g_t) * S_{t-1}
    u_t  = beta_t * (v_t - S_t · kw_t)            (corrected value)
    S_t  = S_t + kw_t · u_t^T                     (state absorption)
    o_t  = (q_t · scale) · S_t

  Here `g_t` is the log-decay per head (usually <=0), applied as a scalar
  on the state. Matches how GatedDeltaNet uses g (see `chunk_gated_delta_rule`).

This file is training-agnostic ground truth. It's deliberately written in
a straightforward loop so that torch autograd gives us reference gradients
for unit testing kernels.
"""
from __future__ import annotations

import torch


def osgm_phase1_naive(
    k: torch.Tensor,
    eta: float = 1.0,
    use_denominator: bool = False,
    d_min: float | None = None,
    d_max: float | None = None,
    initial_d: torch.Tensor | None = None,
    output_final_state: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Compute d_t trajectory from k. Returns (d_traj, d_final).

    Args:
        k: [B, T, H, K]
        initial_d: [B, H, K] or None (zero)

    Returns:
        d_traj: [B, T, H, K], where d_traj[:, t] is d BEFORE processing token t
        d_final: [B, H, K] or None (post-update of last token)
    """
    B, T, H, K = k.shape
    k_f = k.float()
    d_curr = (
        initial_d.float().clone() if initial_d is not None
        else k_f.new_zeros(B, H, K)
    )
    d_traj = k_f.new_zeros(B, T, H, K)

    for t in range(T):
        d_traj[:, t] = d_curr
        k_t = k_f[:, t]
        k_sq = k_t * k_t
        term_A = 1.0 - (d_curr * k_sq).sum(-1, keepdim=True)  # [B,H,1]
        if use_denominator:
            grad_d = term_A / (k_sq.sum(-1, keepdim=True) + 1e-5)
        else:
            grad_d = term_A
        d_next = d_curr + eta * grad_d * k_sq
        if d_min is not None and d_max is not None:
            d_next = d_next.clamp(d_min, d_max)
        d_curr = d_next

    d_final = d_curr if output_final_state else None
    return d_traj, d_final


def naive_recurrent_os_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_log: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    eta: float = 1.0,
    use_denominator: bool = False,
    d_min: float | None = None,
    d_max: float | None = None,
    initial_state: torch.Tensor | None = None,
    initial_d: torch.Tensor | None = None,
    output_final_state: bool = False,
    output_final_d: bool = False,
    # --- d-decay options ---
    decay_mode: str = "none",        # "none" | "learnable" | "constant" | "data_dependent"
    gamma_log: torch.Tensor | None = None,   # [H],        for learnable/constant
    g_decay: torch.Tensor | None = None,     # [B, T, H],  for data_dependent (already logsigmoid'd)
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Reference OSGM + GDN-gate delta-rule recurrence.

    Args:
        q, k: [B, T, H, K]
        v:    [B, T, H, V]
        g_log:[B, T, H]   (per-head log-decay, typically <= 0)
        beta: [B, T, H]
        initial_state: [B, H, K, V] (state S)
        initial_d:     [B, H, K]    (OSGM d_0)
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    if scale is None:
        scale = K ** -0.5

    q_f, k_f, v_f, g_f, b_f = (x.float() for x in (q, k, v, g_log, beta))

    S = k_f.new_zeros(B, H, K, V)
    if initial_state is not None:
        S = S + initial_state.float()

    d_curr = (
        initial_d.float().clone() if initial_d is not None
        else k_f.new_zeros(B, H, K)
    )

    o = v_f.new_zeros(B, T, H, V)

    for t in range(T):
        q_t = q_f[:, t]     # [B, H, K]
        k_t = k_f[:, t]
        v_t = v_f[:, t]     # [B, H, V]
        g_t = g_f[:, t]     # [B, H]
        b_t = b_f[:, t]     # [B, H]

        # --- OSGM phase: use pre-update d_curr at this token; then step d ---
        k_sq = k_t * k_t
        term_A = 1.0 - (d_curr * k_sq).sum(-1, keepdim=True)
        if use_denominator:
            grad_d = term_A / (k_sq.sum(-1, keepdim=True) + 1e-5)
        else:
            grad_d = term_A

        if decay_mode == "none":
            d_base = d_curr
        elif decay_mode in ("learnable", "constant"):
            # γ = sigmoid(gamma_log), per-head scalar applied multiplicatively
            assert gamma_log is not None
            gamma = torch.sigmoid(gamma_log.float())[None, :, None]  # [1,H,1]
            d_base = gamma * d_curr
        elif decay_mode == "data_dependent":
            # γ_t = exp(g_decay_t) where g_decay is typically logsigmoid(a_proj(h))
            assert g_decay is not None
            gamma_t = g_decay[:, t].float().exp()[..., None]          # [B,H,1]
            d_base = gamma_t * d_curr
        else:
            raise ValueError(f"unknown decay_mode={decay_mode}")

        d_next = d_base + eta * grad_d * k_sq
        if d_min is not None and d_max is not None:
            d_next = d_next.clamp(d_min, d_max)

        kw_t = k_t * d_curr           # [B, H, K]

        # --- GDN state forget gate: S <- exp(g_t) * S ---
        gate = g_t.exp()[..., None, None]  # [B, H, 1, 1]
        S = S * gate

        # --- Delta rule update using kw_t as the absorbed key ---
        # Projected value under current state: S^T kw_t (sum over K axis = dim -2 here)
        v_proj = (S * kw_t[..., None]).sum(-2)         # [B, H, V]
        u_t = b_t[..., None] * (v_t - v_proj)          # [B, H, V]
        S = S + kw_t[..., None] * u_t[..., None, :]    # [B, H, K, V]

        # --- Output ---
        q_scaled = q_t * scale
        o[:, t] = (S * q_scaled[..., None]).sum(-2)    # [B, H, V]

        d_curr = d_next

    final_state = S if output_final_state else None
    final_d = d_curr if output_final_d else None
    return o.to(v.dtype), final_state, final_d
