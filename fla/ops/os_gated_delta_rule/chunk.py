# -*- coding: utf-8 -*-
"""
Chunk-mode OSGM + GatedDeltaNet (OS-GDN) orchestration.

Combines:
  - OSGM phase1 (compute per-token d from k), reused from
    `fla.ops.os_delta_rule.chunk_osgm_phase{,_decay,_dd_decay}`.
  - Gated kkt + solve-tril, reused from
    `fla.ops.os_delta_rule.chunk_scaled_dot_qkw_fwd` (already supports g).
  - Gated w/u recompute, reused from `fla.ops.gated_delta_rule.wy_fast`.
  - Common gated state recurrence + output kernels (already support g).

Supported d-decay modes (selected via `decay_mode` argument):
  - `"none"`           (default): plain OSGM, d_{t+1} = d_t + η·grad·k²
  - `"learnable"`     : d_{t+1} = σ(gamma_log)·d_t + η·grad·k², γ learnable
                        per-head.
  - `"constant"`      : same as learnable but γ is a non-learnable buffer.
  - `"data_dependent"`: d_{t+1} = exp(g_decay_t)·d_t + η·grad·k², where
                        g_decay is a per-token per-head signal typically
                        from `logsigmoid(osgm_a_proj(h))`.

Not yet supported:
  - `"ema"` / `"ema_norm"` variants (os_delta_rule has these; port as needed).
  - `use_gate_in_kernel=True` (caller pre-computes the GDN gate with
    `fused_gdn_gate` before calling us).
  - `cp_context` context-parallel training.

Bwd implementation note:
    Forward of the wy representation is kernel-accelerated. For backward,
    the wy part is re-executed in pure torch with autograd tracking so that
    gradients w.r.t. k, d, g, beta, v from both the wy path AND the state
    path are computed jointly by autograd. This avoids needing a fused
    d-aware gated wy bwd kernel (which would otherwise be a risky rewrite
    — see the os_kda bwd debugging session). The wy re-execution is
    chunk-local (BT=64) so it's cheap compared to state bwd.
"""
from __future__ import annotations

from typing import Optional

import torch

from fla.modules.l2norm import l2norm_bwd, l2norm_fwd
from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_bwd_dhu, chunk_gated_delta_rule_fwd_h
from fla.ops.common.chunk_o import chunk_bwd_dqkwg, chunk_bwd_dv_local, chunk_fwd_o
from fla.ops.gated_delta_rule.wy_fast import recompute_w_u_fwd as gdn_recompute_w_u_fwd
from fla.ops.os_delta_rule.chunk_osgm_phase import (
    compute_osgm_phase1_bwd as compute_osgm_phase1_bwd_rec,
    compute_osgm_phase1_fwd as compute_osgm_phase1_fwd_rec,
)
from fla.ops.os_delta_rule.chunk_osgm_phase_decay import (
    compute_osgm_decay_phase1_bwd,
    compute_osgm_decay_phase1_fwd,
)
from fla.ops.os_delta_rule.chunk_osgm_phase_dd_decay import (
    compute_osgm_dd_decay_phase1_bwd,
    compute_osgm_dd_decay_phase1_fwd,
)
from fla.ops.os_delta_rule.chunk_scaled_dot_qkw import chunk_scaled_dot_qkw_fwd
from fla.ops.utils import chunk_local_cumsum
from fla.ops.utils.solve_tril import solve_tril
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard


_CHUNK_SIZE = 64


# ------------------------------------------------------------------
# Pure-torch wy (used only on the backward path, under enable_grad)
# ------------------------------------------------------------------

def _pure_torch_chunk_local_cumsum(g: torch.Tensor, chunk_size: int = _CHUNK_SIZE) -> torch.Tensor:
    """Chunk-local cumsum along the time axis, in pure torch.

    Matches `fla.ops.utils.chunk_local_cumsum` semantics: within each chunk
    of length `chunk_size`, compute inclusive cumulative sum. Chunk boundaries
    reset. Used on bwd so that autograd flows back to the raw g.

    Args:
        g: [B, T, H]
    Returns:
        g_cum: [B, T, H], fp32
    """
    B, T, H = g.shape
    assert T % chunk_size == 0, (
        f"_pure_torch_chunk_local_cumsum requires T ({T}) divisible by "
        f"chunk_size ({chunk_size}); caller must pad upstream."
    )
    g_f = g.float()
    g_c = g_f.view(B, T // chunk_size, chunk_size, H)
    g_c = torch.cumsum(g_c, dim=2)
    return g_c.view(B, T, H)


def _pure_torch_wy(
    k: torch.Tensor,
    kw: torch.Tensor,
    g_cum: torch.Tensor,
    beta: torch.Tensor,
    v: torch.Tensor,
    chunk_size: int = _CHUNK_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure-torch chunk-local wy representation for OS-GDN.

    Builds A = (I + X)^{-1}, chunk-locally, where
        X_ij = β_i · exp(g_i - g_j) · k_i · kw_j^T   for j < i  (strict lower tri)
    then
        w = A @ (k · β · exp(g_cum))
        u = A @ (v · β)

    Matches the kernel fwd path (chunk_scaled_dot_qkw_fwd + solve_tril +
    gdn_recompute_w_u_fwd) mathematically.

    This is only called under `torch.enable_grad()` in the backward of the
    autograd.Function below. It is NOT used on the forward pass (that goes
    through triton kernels).
    """
    B, T, H, K = k.shape
    V = v.shape[-1]
    BT = chunk_size
    assert T % BT == 0

    nc = T // BT

    def chunkify(x: torch.Tensor) -> torch.Tensor:
        # [B, T, H, *] -> [B, nc, H, BT, *]   (move H in front of BT)
        return x.view(B, nc, BT, H, -1).transpose(2, 3)

    k_c = chunkify(k).float()                                    # [B,nc,H,BT,K]
    kw_c = chunkify(kw).float()                                  # [B,nc,H,BT,K]
    v_c = chunkify(v).float()                                    # [B,nc,H,BT,V]
    g_c = g_cum.float().view(B, nc, BT, H).transpose(2, 3)       # [B,nc,H,BT]
    beta_c = beta.float().view(B, nc, BT, H).transpose(2, 3)     # [B,nc,H,BT]

    # X_ij = β_i · exp(g_i - g_j) · <k_i, kw_j>  for j < i
    # Compute gdiff first, apply the mask to the EXPONENT (not the final X)
    # so that unused upper-tri entries of `exp(gdiff)` don't overflow when
    # g_cum has a large dynamic range within the chunk (e.g. when the GDN
    # gate initialization gives per-token decay of order 1, cumulative decay
    # over BT=64 can reach hundreds in magnitude and upper-tri positions
    # would produce exp(+large) → inf → NaN through autograd.backward).
    mask = torch.tril(
        torch.ones(BT, BT, device=k.device, dtype=torch.bool),
        diagonal=-1,
    )
    gdiff = g_c.unsqueeze(-1) - g_c.unsqueeze(-2)                # [B,nc,H,BT,BT]
    gdiff = torch.where(mask, gdiff, torch.zeros_like(gdiff))    # kill unused entries
    kkt = torch.einsum('bnhik,bnhjk->bnhij', k_c, kw_c)          # [B,nc,H,BT,BT]
    X = kkt * torch.exp(gdiff) * beta_c.unsqueeze(-1)
    X = torch.where(mask, X, torch.zeros_like(X))  # strictly lower tri

    # A = (I + X)^{-1}   via triangular solve.
    I = torch.eye(BT, device=k.device, dtype=X.dtype).expand_as(X).contiguous()
    A = torch.linalg.solve_triangular(I + X, I, upper=False)     # [B,nc,H,BT,BT]

    # w = A @ (k · β · exp(g_cum))
    kbg_c = k_c * (beta_c * torch.exp(g_c)).unsqueeze(-1)        # [B,nc,H,BT,K]
    vb_c = v_c * beta_c.unsqueeze(-1)                            # [B,nc,H,BT,V]

    w_c = torch.einsum('bnhij,bnhjk->bnhik', A, kbg_c)           # [B,nc,H,BT,K]
    u_c = torch.einsum('bnhij,bnhjv->bnhiv', A, vb_c)            # [B,nc,H,BT,V]

    # [B,nc,H,BT,*] -> [B,T,H,*]  (keep fp32 through the autograd graph to
    # avoid bf16 underflow in the linalg.solve_triangular-driven backward).
    w = w_c.transpose(2, 3).reshape(B, T, H, K)
    u = u_c.transpose(2, 3).reshape(B, T, H, V)
    return w, u


# ------------------------------------------------------------------
# Kernel-path forward + backward
# ------------------------------------------------------------------

def os_gated_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,        # [B, T, H], raw log-decay per head (NOT cumsum'd)
    beta: torch.Tensor,
    d: torch.Tensor,        # [B, T, H, K] from OSGM phase1
    scale: float,
    initial_state: Optional[torch.Tensor],
    output_final_state: bool,
    cu_seqlens: Optional[torch.LongTensor] = None,
):
    """Kernel-accelerated forward. Returns (o, A, g_cum, final_state).

    A and g_cum are saved for the backward pass.
    """
    kw = k * d
    g_cum = chunk_local_cumsum(g, chunk_size=_CHUNK_SIZE, scale=None, cu_seqlens=cu_seqlens)

    # A = (I + β·exp(g_i-g_j)·k·kw^T)^{-1}, chunk-locally.
    A = chunk_scaled_dot_qkw_fwd(
        q=k, k=kw, g=g_cum, beta=beta,
        cu_seqlens=cu_seqlens, chunk_size=_CHUNK_SIZE,
    )
    A = solve_tril(A=A, cu_seqlens=cu_seqlens, output_dtype=k.dtype)

    # Reuse GDN wy kernel for w, u. It computes
    #     w = A @ (k · β · exp(g_cum)),  u = A @ (v · β)
    # which matches the (d-absorbed-into-state) formulation.
    w, u = gdn_recompute_w_u_fwd(
        k=k, v=v, beta=beta, A=A, g=g_cum,
        cu_seqlens=cu_seqlens, use_exp2=False,
    )

    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=kw, w=w, u=u, g=g_cum,
        initial_state=initial_state, output_final_state=output_final_state,
        cu_seqlens=cu_seqlens, use_exp2=False,
    )
    o = chunk_fwd_o(
        q=q, k=kw, v=v_new, h=h, g=g_cum, scale=scale,
        cu_seqlens=cu_seqlens, use_exp2=False,
    )
    return o, A, g_cum, final_state


def os_gated_delta_rule_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_raw: torch.Tensor,
    beta: torch.Tensor,
    d: torch.Tensor,
    A: torch.Tensor,
    g_cum: torch.Tensor,
    scale: float,
    initial_state: Optional[torch.Tensor],
    do: torch.Tensor,
    dht: Optional[torch.Tensor],
    cu_seqlens: Optional[torch.LongTensor] = None,
):
    """Kernel-accelerated state bwd + pure-torch wy bwd via autograd.

    Returns: dq, dk (pre-phase1), dv, db, dg, dh0, dd_from_state
    """
    kw = k * d

    # Recompute w, u via GDN kernel (same call as fwd).
    w, u = gdn_recompute_w_u_fwd(
        k=k, v=v, beta=beta, A=A, g=g_cum,
        cu_seqlens=cu_seqlens, use_exp2=False,
    )

    # State fwd recomputation (for h, v_new).
    h, v_new, _ = chunk_gated_delta_rule_fwd_h(
        k=kw, w=w, u=u, g=g_cum,
        initial_state=initial_state, output_final_state=False,
        cu_seqlens=cu_seqlens, use_exp2=False,
    )

    # --- State-side bwd (kernels) ---
    dv_local = chunk_bwd_dv_local(
        q=q, k=kw, g=g_cum, do=do, scale=scale,
        cu_seqlens=cu_seqlens, use_exp2=False,
    )
    dh, dh0, du = chunk_gated_delta_rule_bwd_dhu(
        q=q, k=kw, w=w, g=g_cum, h0=initial_state, dht=dht, do=do, dv=dv_local,
        scale=scale, cu_seqlens=cu_seqlens, use_exp2=False,
    )
    dq, dkw_from_state, dw, dg_from_state = chunk_bwd_dqkwg(
        q=q, k=kw, v=v_new, w=w, g=g_cum, h=h, dv=du, do=do, dh=dh, scale=scale,
        cu_seqlens=cu_seqlens, use_exp2=False,
    )

    # --- wy-side bwd via pure-torch autograd (all fp32) ---
    # We rebuild the wy + kw computation in python with autograd tracking,
    # then feed (dw, du, dkw_from_state, dg_from_state) as grad_outputs.
    # Autograd splits them correctly across k, d, g, beta, v.
    #
    # The whole graph is kept in fp32 (by detaching fp32 clones as inputs) to
    # avoid bf16 underflow through `linalg.solve_triangular`. Grads are cast
    # back to input dtypes at the very end.
    k_g = k.detach().float().requires_grad_(True)
    d_g = d.detach().float().requires_grad_(True)
    g_raw_g = g_raw.detach().float().requires_grad_(True)
    beta_g = beta.detach().float().requires_grad_(True)
    v_g = v.detach().float().requires_grad_(True)

    with torch.enable_grad():
        g_cum_py = _pure_torch_chunk_local_cumsum(g_raw_g, _CHUNK_SIZE)
        kw_py = k_g * d_g
        w_py, u_py = _pure_torch_wy(k_g, kw_py, g_cum_py, beta_g, v_g, _CHUNK_SIZE)
        torch.autograd.backward(
            tensors=[w_py, u_py, kw_py, g_cum_py],
            grad_tensors=[
                dw.float(),
                du.float(),
                dkw_from_state.float(),
                dg_from_state.float(),
            ],
            inputs=[k_g, d_g, g_raw_g, beta_g, v_g],
        )

    dk = k_g.grad.to(k.dtype)
    dd_from_state = d_g.grad.to(d.dtype)
    dg = g_raw_g.grad.to(g_raw.dtype)
    db = beta_g.grad.to(beta.dtype)
    dv = v_g.grad.to(v.dtype)

    return dq, dk, dv, db, dg, dh0, dd_from_state


class ChunkOSGatedDeltaRuleFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        q, k, v, g, beta,
        scale, eta, initial_h, initial_d, output_final_state,
        use_qk_l2norm_in_kernel, cu_seqlens,
        use_denominator, d_min, d_max,
    ):
        q_rstd, k_rstd = None, None
        if use_qk_l2norm_in_kernel:
            q, q_rstd = l2norm_fwd(q)
            k, k_rstd = l2norm_fwd(k)

        d, final_d = compute_osgm_phase1_fwd_rec(
            k, eta, use_denominator, d_min, d_max, cu_seqlens,
            initial_d, output_final_state,
        )

        o, A, g_cum, final_h = os_gated_delta_rule_fwd(
            q=q, k=k, v=v, g=g, beta=beta, d=d, scale=scale,
            initial_state=initial_h, output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
        )

        ctx.save_for_backward(
            q, q_rstd, k, k_rstd, v, g, g_cum, d, beta, A, initial_h,
        )
        ctx.scale = scale
        ctx.eta = eta
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        ctx.cu_seqlens = cu_seqlens
        ctx.use_denominator = use_denominator
        ctx.d_min = d_min
        ctx.d_max = d_max
        ctx.has_initial_d = (initial_d is not None)

        return o.to(q.dtype), final_h, final_d

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do, dh_final, dd_final):
        (q, q_rstd, k, k_rstd, v, g_raw, g_cum, d, beta, A, initial_h) = ctx.saved_tensors

        dq, dk_state, dv, db, dg, dh0, dd_from_state = os_gated_delta_rule_bwd(
            q=q, k=k, v=v, g_raw=g_raw, beta=beta, d=d, A=A, g_cum=g_cum,
            scale=ctx.scale,
            initial_state=initial_h, do=do, dht=dh_final,
            cu_seqlens=ctx.cu_seqlens,
        )

        dk_phase1, dd_initial = compute_osgm_phase1_bwd_rec(
            k, d, dd_from_state, ctx.eta, ctx.use_denominator,
            ctx.d_min, ctx.d_max, ctx.cu_seqlens,
            dd_final=dd_final,
            output_initial_state_gradient=ctx.has_initial_d,
        )
        dk = dk_state + dk_phase1

        if ctx.use_qk_l2norm_in_kernel:
            dq = l2norm_bwd(q, q_rstd, dq)
            dk = l2norm_bwd(k, k_rstd, dk)

        return (
            dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype),
            dg.to(q.dtype) if dg is not None else None, db.to(beta.dtype),
            None, None, dh0, dd_initial, None, None, None, None, None, None,
        )


class ChunkOSGatedDeltaRuleDecayFunction(torch.autograd.Function):
    """OSGM with learnable/constant gamma decay on d: d_{t+1} = γ·d_t + η·grad·k²."""

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        q, k, v, g, beta, gamma_log,
        scale, eta, initial_h, initial_d, output_final_state,
        use_qk_l2norm_in_kernel, cu_seqlens,
        use_denominator, d_min, d_max,
    ):
        q_rstd, k_rstd = None, None
        if use_qk_l2norm_in_kernel:
            q, q_rstd = l2norm_fwd(q)
            k, k_rstd = l2norm_fwd(k)

        d, final_d = compute_osgm_decay_phase1_fwd(
            k, gamma_log, eta, use_denominator, d_min, d_max,
            cu_seqlens, initial_d, output_final_state,
        )

        o, A, g_cum, final_h = os_gated_delta_rule_fwd(
            q=q, k=k, v=v, g=g, beta=beta, d=d, scale=scale,
            initial_state=initial_h, output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
        )

        ctx.save_for_backward(
            q, q_rstd, k, k_rstd, v, g, g_cum, d, beta, A, gamma_log, initial_h,
        )
        ctx.scale = scale
        ctx.eta = eta
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        ctx.cu_seqlens = cu_seqlens
        ctx.use_denominator = use_denominator
        ctx.d_min = d_min
        ctx.d_max = d_max
        ctx.has_initial_d = (initial_d is not None)
        return o.to(q.dtype), final_h, final_d

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do, dh_final, dd_final):
        (q, q_rstd, k, k_rstd, v, g_raw, g_cum, d, beta, A,
         gamma_log, initial_h) = ctx.saved_tensors

        dq, dk_state, dv, db, dg, dh0, dd_from_state = os_gated_delta_rule_bwd(
            q=q, k=k, v=v, g_raw=g_raw, beta=beta, d=d, A=A, g_cum=g_cum,
            scale=ctx.scale,
            initial_state=initial_h, do=do, dht=dh_final,
            cu_seqlens=ctx.cu_seqlens,
        )
        dk_phase1, dd_initial, dgamma = compute_osgm_decay_phase1_bwd(
            k, gamma_log, d, dd_from_state, ctx.eta, ctx.use_denominator,
            ctx.d_min, ctx.d_max, ctx.cu_seqlens,
            dd_final=dd_final,
            output_initial_state_gradient=ctx.has_initial_d,
        )
        dk = dk_state + dk_phase1

        if ctx.use_qk_l2norm_in_kernel:
            dq = l2norm_bwd(q, q_rstd, dq)
            dk = l2norm_bwd(k, k_rstd, dk)

        return (
            dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype),
            dg.to(q.dtype) if dg is not None else None, db.to(beta.dtype),
            dgamma,                          # d/dgamma_log
            None, None, dh0, dd_initial,
            None, None, None, None, None, None,
        )


class ChunkOSGatedDeltaRuleDDDecayFunction(torch.autograd.Function):
    """OSGM with data-dependent (per-token per-head) decay on d.

    The decay γ_t is derived from a_proj(h) via logsigmoid, and passed in as
    `g_decay: [B, T, H]`. This is a separate gate from the GDN state forget
    gate `g` which acts on S_{t-1}.
    """

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        q, k, v, g, beta, g_decay,
        scale, eta, initial_h, initial_d, output_final_state,
        use_qk_l2norm_in_kernel, cu_seqlens,
        use_denominator, d_min, d_max,
    ):
        q_rstd, k_rstd = None, None
        if use_qk_l2norm_in_kernel:
            q, q_rstd = l2norm_fwd(q)
            k, k_rstd = l2norm_fwd(k)

        d, final_d = compute_osgm_dd_decay_phase1_fwd(
            k, g_decay, eta, use_denominator, d_min, d_max,
            cu_seqlens, initial_d, output_final_state,
        )

        o, A, g_cum, final_h = os_gated_delta_rule_fwd(
            q=q, k=k, v=v, g=g, beta=beta, d=d, scale=scale,
            initial_state=initial_h, output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
        )

        ctx.save_for_backward(
            q, q_rstd, k, k_rstd, v, g, g_cum, d, beta, A, g_decay, initial_h,
        )
        ctx.scale = scale
        ctx.eta = eta
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        ctx.cu_seqlens = cu_seqlens
        ctx.use_denominator = use_denominator
        ctx.d_min = d_min
        ctx.d_max = d_max
        ctx.has_initial_d = (initial_d is not None)
        return o.to(q.dtype), final_h, final_d

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do, dh_final, dd_final):
        (q, q_rstd, k, k_rstd, v, g_raw, g_cum, d, beta, A,
         g_decay, initial_h) = ctx.saved_tensors

        dq, dk_state, dv, db, dg, dh0, dd_from_state = os_gated_delta_rule_bwd(
            q=q, k=k, v=v, g_raw=g_raw, beta=beta, d=d, A=A, g_cum=g_cum,
            scale=ctx.scale,
            initial_state=initial_h, do=do, dht=dh_final,
            cu_seqlens=ctx.cu_seqlens,
        )
        dk_phase1, dd_initial, dg_decay = compute_osgm_dd_decay_phase1_bwd(
            k, g_decay, d, dd_from_state, ctx.eta, ctx.use_denominator,
            ctx.d_min, ctx.d_max, ctx.cu_seqlens,
            dd_final=dd_final,
            output_initial_state_gradient=ctx.has_initial_d,
        )
        dk = dk_state + dk_phase1

        if ctx.use_qk_l2norm_in_kernel:
            dq = l2norm_bwd(q, q_rstd, dq)
            dk = l2norm_bwd(k, k_rstd, dk)

        return (
            dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype),
            dg.to(q.dtype) if dg is not None else None, db.to(beta.dtype),
            dg_decay.to(g_decay.dtype) if dg_decay is not None else None,
            None, None, dh0, dd_initial,
            None, None, None, None, None, None,
        )


@torch.compiler.disable
def chunk_os_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,                                # [B, T, H] raw log-decay per head
    beta: torch.Tensor,
    scale: Optional[float] = None,
    eta: Optional[float] = None,
    initial_state=None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
    use_denominator: Optional[bool] = None,
    d_min: Optional[float] = None,
    d_max: Optional[float] = None,
    # --- d-decay dispatch ---
    decay_mode: str = "none",
    gamma_log: Optional[torch.Tensor] = None,      # for "learnable" / "constant"
    g_decay: Optional[torch.Tensor] = None,        # for "data_dependent"
    **kwargs,
):
    """OSGM + GatedDeltaNet chunk-mode entrypoint.

    Args:
        q, k: [B, T, H, K]
        v:    [B, T, H, V]
        g:    [B, T, H]   per-head log-decay (NOT cumsum'd; we do the cumsum)
        beta: [B, T, H]
        decay_mode: one of "none" / "learnable" / "constant" / "data_dependent".
            - "none": plain OSGM, d_{t+1} = d_t + η·grad·k²
            - "learnable": d_{t+1} = σ(gamma_log)·d_t + η·grad·k²  (γ is a
              learnable per-head parameter).
            - "constant": same as learnable but γ is a non-learnable buffer.
            - "data_dependent": d_{t+1} = σ(g_decay)·d_t + η·grad·k²  where
              g_decay is per-token per-head, typically
              `logsigmoid(osgm_a_proj(h))`.
        gamma_log: required for "learnable"/"constant", shape [num_heads].
        g_decay:   required for "data_dependent", shape [B, T, num_heads].
        initial_state: None, or a tuple (initial_h, initial_d), or (legacy)
                       a single tensor treated as initial_h.

    Returns:
        (o, (final_h, final_d)) if output_final_state else (o, None)
    """
    if decay_mode not in ("none", "learnable", "constant", "data_dependent"):
        raise ValueError(
            f"Unsupported decay_mode={decay_mode!r}. "
            f"Expected one of: none, learnable, constant, data_dependent."
        )

    if use_qk_l2norm_in_kernel:
        if eta is None:
            eta = 1.0
        if use_denominator is None:
            use_denominator = False
        if d_min is None:
            d_min = 0.0
        if d_max is None:
            d_max = 1e9
    else:
        if eta is None:
            eta = 0.1
        if use_denominator is None:
            use_denominator = True
        if d_min is None:
            d_min = 0.0
        if d_max is None:
            d_max = 1e9

    if scale is None:
        scale = k.shape[-1] ** -0.5

    initial_h, initial_d = None, None
    if initial_state is not None:
        if isinstance(initial_state, tuple):
            initial_h, initial_d = initial_state
        else:
            initial_h = initial_state

    if decay_mode == "none":
        o, final_h, final_d = ChunkOSGatedDeltaRuleFunction.apply(
            q, k, v, g, beta,
            scale, eta, initial_h, initial_d, output_final_state,
            use_qk_l2norm_in_kernel, cu_seqlens,
            use_denominator, d_min, d_max,
        )
    elif decay_mode in ("learnable", "constant"):
        if gamma_log is None:
            raise ValueError(f"decay_mode={decay_mode!r} requires `gamma_log` (shape [H])")
        o, final_h, final_d = ChunkOSGatedDeltaRuleDecayFunction.apply(
            q, k, v, g, beta, gamma_log,
            scale, eta, initial_h, initial_d, output_final_state,
            use_qk_l2norm_in_kernel, cu_seqlens,
            use_denominator, d_min, d_max,
        )
    else:  # data_dependent
        if g_decay is None:
            raise ValueError("decay_mode='data_dependent' requires `g_decay` (shape [B, T, H])")
        o, final_h, final_d = ChunkOSGatedDeltaRuleDDDecayFunction.apply(
            q, k, v, g, beta, g_decay,
            scale, eta, initial_h, initial_d, output_final_state,
            use_qk_l2norm_in_kernel, cu_seqlens,
            use_denominator, d_min, d_max,
        )

    if output_final_state:
        return o, (final_h, final_d)
    return o, None
