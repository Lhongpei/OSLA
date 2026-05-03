# -*- coding: utf-8 -*-
"""
OSGM + GatedDeltaNet with the *post-gate regret* hypergradient (OS_GDN_REPORT §3.4).

Background (why this exists):
  Plain OSGM-on-DeltaNet uses ``grad_d = 1 − ⟨d, k²⟩`` because the per-step
  loss change ``f(S_t) − f(S_{t−1})`` is a clean function of d alone. With
  GDN's state-forget gate ``α = exp(g_gdn)``, the residual after the rank-1
  write picks up an additive ``(1 − α)·v`` term that d cannot influence —
  so the un-modified hypergradient is biased and OSGM's regret bound's
  comparator class becomes too weak (see §3.2–3.3 of the report).

Fix derived in §3.4: replace the regret reference with f(α·S_{t−1}) instead
of f(S_{t−1}), giving

    grad_d = ⟨ẽ, e'⟩ / ‖e'‖² − ⟨d, k²⟩

where  e' = v − k^T·S_{t−1}     (pre-gate residual)
       ẽ  = v − α·k^T·S_{t−1}    (post-gate residual)

When α≡1, this collapses to ``1 − ⟨d, k²⟩`` exactly, so un-gated training
is unaffected (DeltaNet/OSDN tests still pass).

Implementation notes
--------------------
This file ships a **pure-pytorch** chunked recurrence with
``torch.utils.checkpoint`` at chunk boundaries. It is slow vs the chunk-mode
triton kernel but:
  - is numerically simple and reviewable,
  - has correct autograd (no manual backward derivation),
  - serves as both reference and smoke-test path.

If the smoke test shows the gap closes, port to triton (one fused recurrent
kernel modeled on ``fla.ops.os_delta_rule.fused_recurrent_osgm.py``).
"""
from __future__ import annotations

import os
from typing import Optional, Tuple

import torch
from torch.utils.checkpoint import checkpoint


_DEFAULT_CHUNK = 64

# torch.compile of `_post_gate_chunk_step` removes the per-token kernel-launch
# overhead that dominates wall-time at production scale (21 layers × T=4096
# yields ~86k host iterations per fwd; without compile MFU collapses to <0.1%).
# Disable via OSLA_DISABLE_POST_GATE_COMPILE=1 if a torch/inductor regression
# breaks compilation on a given host.
_DISABLE_COMPILE = os.environ.get("OSLA_DISABLE_POST_GATE_COMPILE", "0") == "1"


def _post_gate_chunk_step(
    q_c: torch.Tensor,                 # [B, C, H, K]
    k_c: torch.Tensor,
    v_c: torch.Tensor,                 # [B, C, H, V]
    g_c: torch.Tensor,                 # [B, C, H]
    beta_c: torch.Tensor,              # [B, C, H]
    g_decay_c: Optional[torch.Tensor], # [B, C, H] or None
    S: torch.Tensor,                   # [B, H, K, V] fp32
    d: torch.Tensor,                   # [B, H, K]    fp32
    eta: float,
    decay_mode: str,
    gamma_log: Optional[torch.Tensor], # [H] for learnable/constant
    d_min: Optional[float],
    d_max: Optional[float],
    scale: float,
):
    """Run the post-gate-regret recurrence over one chunk.

    Returns:
        o_c: [B, C, H, V] (in q's dtype)
        S_next: [B, H, K, V] fp32
        d_next: [B, H, K]    fp32
    """
    B, C, H, K = q_c.shape
    V = v_c.shape[-1]

    # Float promotion (state path stays fp32 to avoid bf16 underflow on the
    # rank-1 K×V update).
    q_c_f = q_c.float()
    k_c_f = k_c.float()
    v_c_f = v_c.float()
    g_c_f = g_c.float()
    beta_c_f = beta_c.float()
    g_decay_c_f = g_decay_c.float() if g_decay_c is not None else None

    # Pre-compute per-token alphas once.
    alpha_c = g_c_f.exp()                                  # [B, C, H]

    o_buf = q_c.new_empty(B, C, H, V)
    for t in range(C):
        q_t = q_c_f[:, t]              # [B, H, K]
        k_t = k_c_f[:, t]
        v_t = v_c_f[:, t]              # [B, H, V]
        b_t = beta_c_f[:, t]           # [B, H]
        a_t = alpha_c[:, t]            # [B, H]

        # Pre-gate residual e' = v − k^T·S
        proj = (S * k_t[..., None]).sum(-2)               # [B, H, V]
        e_prime = v_t - proj                               # [B, H, V]
        # Post-gate residual ẽ = (1-α)·v + α·e'
        a_v = a_t[..., None]                               # [B, H, 1]
        e_tilde = (1.0 - a_v) * v_t + a_v * e_prime

        # Hypergradient
        e_norm_sq = (e_prime * e_prime).sum(-1, keepdim=True) + 1e-8   # [B,H,1]
        e_dot = (e_tilde * e_prime).sum(-1, keepdim=True)              # [B,H,1]
        k_sq = k_t * k_t
        inner_d_k = (d * k_sq).sum(-1, keepdim=True)                   # [B,H,1]
        grad_d = e_dot / e_norm_sq - inner_d_k                          # [B,H,1]

        # OSGM-side decay
        if decay_mode == "none":
            d_base = d
        elif decay_mode in ("learnable", "constant"):
            assert gamma_log is not None
            gamma = torch.sigmoid(gamma_log.float())[None, :, None]    # [1,H,1]
            d_base = gamma * d
        elif decay_mode == "data_dependent":
            assert g_decay_c_f is not None
            gamma_t = g_decay_c_f[:, t].exp()[..., None]                # [B,H,1]
            d_base = gamma_t * d
        else:
            raise ValueError(f"unknown decay_mode={decay_mode!r}")

        d_next = d_base + eta * grad_d * k_sq
        if d_min is not None and d_max is not None:
            d_next = d_next.clamp(d_min, d_max)

        # State update: S ← α·S + β·kw·e'^T (rank-1 outer)
        kw_t = k_t * d                                                  # uses d at step t
        S = a_t[..., None, None] * S
        S = S + kw_t[..., None] * (b_t[..., None] * e_prime)[..., None, :]

        # Output
        q_scaled = q_t * scale
        o_t = (S * q_scaled[..., None]).sum(-2)                         # [B, H, V]
        o_buf[:, t] = o_t.to(q_c.dtype)

        d = d_next

    return o_buf, S, d


# Compiled variant. dynamo unrolls the inner ``for t in range(C)`` (C is constant
# at 64), specializes on decay_mode/g_decay_c None-ness, and fuses the small
# per-token ops into one graph. ``fullgraph=False`` is required because
# ``decay_mode`` is a Python string with branching; dynamo graph-breaks at the
# string compare and emits a separate trace per mode. ``dynamic=False`` keeps
# shapes static so we don't pay re-compilation cost mid-training (B, T, H, K, V
# are fixed across a run).
if _DISABLE_COMPILE or not hasattr(torch, "compile"):
    _compiled_chunk_step = _post_gate_chunk_step
else:
    _compiled_chunk_step = torch.compile(
        _post_gate_chunk_step, fullgraph=False, dynamic=False
    )


def post_gate_regret_recurrence(
    q: torch.Tensor,                   # [B, T, H, K]
    k: torch.Tensor,
    v: torch.Tensor,                   # [B, T, H, V]
    g: torch.Tensor,                   # [B, T, H]   raw GDN log-decay (NOT cumsum'd)
    beta: torch.Tensor,                # [B, T, H]
    scale: Optional[float] = None,
    eta: float = 1.0,
    initial_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    output_final_state: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
    d_min: Optional[float] = None,
    d_max: Optional[float] = None,
    decay_mode: str = "none",
    gamma_log: Optional[torch.Tensor] = None,
    g_decay: Optional[torch.Tensor] = None,
    chunk_size: int = _DEFAULT_CHUNK,
    use_checkpoint: Optional[bool] = None,
    use_compile: Optional[bool] = None,
):
    """Post-gate-regret OS-GDN forward (autograd-friendly via checkpointing).

    Args:
        q, k, v, g, beta: standard OS-GDN inputs.
        scale: query scale (default K^-0.5).
        eta: OSGM step size.
        initial_state: (h0, d0) tuple; h0 is [N, H, K, V], d0 is [N, H, K].
        output_final_state: return (h_final, d_final).
        cu_seqlens: NOT supported (pad to full T for now).
        d_min/d_max: optional projection box for d.
        decay_mode/gamma_log/g_decay: same as ``chunk_os_gated_delta_rule``.
        chunk_size: granularity for activation checkpointing. Default 64.
        use_checkpoint: if None, defaults to (training-mode and grad-required).

    Returns:
        (o, (h_final, d_final)) if output_final_state else (o, None).
    """
    if cu_seqlens is not None:
        raise NotImplementedError(
            "post_gate_regret_recurrence does not yet support varlen (cu_seqlens). "
            "Pad inputs to a fixed T."
        )
    if decay_mode not in ("none", "learnable", "constant", "data_dependent"):
        raise ValueError(f"unknown decay_mode={decay_mode!r}")

    B, T, H, K = q.shape
    V = v.shape[-1]
    if scale is None:
        scale = K ** -0.5
    if T % chunk_size != 0:
        raise ValueError(
            f"T ({T}) must be divisible by chunk_size ({chunk_size}). "
            f"Either pad upstream or set a divisor chunk_size."
        )

    # Initial state.
    h0, d0 = (None, None)
    if initial_state is not None:
        h0, d0 = initial_state
    if h0 is None:
        h0 = q.new_zeros(B, H, K, V, dtype=torch.float32)
    else:
        h0 = h0.float()
    if d0 is None:
        d0 = q.new_zeros(B, H, K, dtype=torch.float32)
    else:
        d0 = d0.float()

    if use_checkpoint is None:
        use_checkpoint = q.requires_grad and torch.is_grad_enabled()
    if use_compile is None:
        use_compile = q.is_cuda and not _DISABLE_COMPILE
    chunk_fn = _compiled_chunk_step if use_compile else _post_gate_chunk_step

    n_chunks = T // chunk_size
    o_chunks = []
    S, d = h0, d0
    for c in range(n_chunks):
        s, e = c * chunk_size, (c + 1) * chunk_size
        q_c, k_c, v_c, g_c, b_c = q[:, s:e], k[:, s:e], v[:, s:e], g[:, s:e], beta[:, s:e]
        g_decay_c = g_decay[:, s:e] if g_decay is not None else None

        if use_checkpoint:
            o_c, S, d = checkpoint(
                chunk_fn,
                q_c, k_c, v_c, g_c, b_c, g_decay_c, S, d,
                eta, decay_mode, gamma_log, d_min, d_max, scale,
                use_reentrant=False,
            )
        else:
            o_c, S, d = chunk_fn(
                q_c, k_c, v_c, g_c, b_c, g_decay_c, S, d,
                eta, decay_mode, gamma_log, d_min, d_max, scale,
            )
        o_chunks.append(o_c)

    o = torch.cat(o_chunks, dim=1)
    if output_final_state:
        return o, (S, d)
    return o, None
