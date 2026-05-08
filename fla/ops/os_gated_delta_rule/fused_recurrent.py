# -*- coding: utf-8 -*-
"""Fused recurrent (single-token) kernel for OS-GDN.

Implements the post-gate-regret OSGM + GatedDeltaNet recurrence in a single
Triton kernel, matching the chunk-mode forward of
``fla.ops.os_gated_delta_rule.chunk_os_gated_delta_rule`` with
``post_gate_regret_beta_aware=True`` for ``decay_mode in {"none", "data_dependent"}``.

Per-token math (matches ``post_gate_regret._post_gate_chunk_step``):

    alpha_t = exp(g_t)                                     # GDN forget gate
    S_bar   = alpha_t * S_{t-1}                            # decay state
    e_t     = v_t - S_bar^T k_t                            # residual (read with raw k)
    inner   = <d_{t-1}, k_t**2>
    grad_d  = beta_t * (1 - beta_t * inner)                # post-gate-regret
    d_base  = d_{t-1} if decay_mode=="none" else exp(g_decay_t) * d_{t-1}
    d_t     = clamp(d_base + eta * grad_d * k_t**2, d_min, d_max)
    kw_t    = k_t * d_{t-1}                                # write with OLD d
    S_t     = S_bar + kw_t (outer) (beta_t * e_t)
    o_t     = scale * S_t^T q_t

Forward only: training uses the chunk path; this kernel exists for fast decode
(q_len <= 64).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from fla.utils import input_guard


# Decay-mode flags. These show up as DECAY_MODE constexpr inside the kernel.
# Triton (>=3.x) requires that any module-level constant referenced from a
# @triton.jit body be a tl.constexpr. We keep plain ints here for the Python
# wrapper; the kernel compares its DECAY_MODE constexpr against the same
# integer literals (0 = none, 1 = data_dependent).
_DECAY_NONE = 0
_DECAY_DD = 1


@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
    'USE_INITIAL_D': lambda args: args['d0'] is not None,
    'STORE_FINAL_D': lambda args: args['dt'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def fused_recurrent_os_gated_delta_rule_fwd_kernel(
    q,
    k,
    v,
    g,
    beta,
    g_decay,
    o,
    h0,
    ht,
    d0,
    dt,
    cu_seqlens,
    scale,
    eta,
    d_min,
    d_max,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    USE_PROJECTION: tl.constexpr,
    DECAY_MODE: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    USE_INITIAL_D: tl.constexpr,
    STORE_FINAL_D: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H

    if IS_VARLEN:
        bos = tl.load(cu_seqlens + i_n).to(tl.int64)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T_local = eos - bos
    else:
        bos = i_n * T
        T_local = T

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)

    p_q = q + (bos * H + i_h) * K + o_k
    p_k = k + (bos * H + i_h) * K + o_k
    p_v = v + (bos * H + i_h) * V + o_v
    p_o = o + (bos * H + i_h) * V + o_v

    p_g = g + bos * H + i_h
    p_beta = beta + bos * H + i_h
    if DECAY_MODE == 1:  # _DECAY_DD
        p_g_decay = g_decay + bos * H + i_h

    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    # Recurrent state h: [BK, BV]
    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = h0 + i_nh * K * V + o_k[:, None] * V + o_v[None, :]
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    # Diagonal preconditioner state d: [BK]
    b_d = tl.zeros([BK], dtype=tl.float32)
    if USE_INITIAL_D:
        p_d0 = d0 + i_nh * K + o_k
        b_d += tl.load(p_d0, mask=mask_k, other=0).to(tl.float32)

    for _ in tl.range(0, T_local):
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q / (tl.sqrt(tl.sum(b_q * b_q)) + 1e-6)
            b_k = b_k / (tl.sqrt(tl.sum(b_k * b_k)) + 1e-6)
        b_q = b_q * scale

        b_g = tl.load(p_g).to(tl.float32)
        b_beta = tl.load(p_beta).to(tl.float32)

        # Step 1: GDN forget gate (decay state)
        b_alpha = tl.exp(b_g)
        b_h *= b_alpha

        # Step 2: residual e = v - S_bar^T k  (read with raw k)
        b_v_minus = tl.sum(b_h * b_k[:, None], 0)
        b_e = b_v - b_v_minus

        # Step 3: post-gate-regret hypergradient
        b_k_sq = b_k * b_k
        b_inner = tl.sum(b_d * b_k_sq)
        b_grad_d = b_beta * (1.0 - b_beta * b_inner)

        # Step 4: compute d_next using OLD d
        if DECAY_MODE == 1:  # _DECAY_DD
            b_g_dec = tl.load(p_g_decay).to(tl.float32)
            b_d_base = tl.exp(b_g_dec) * b_d
        else:
            b_d_base = b_d
        b_d_next = b_d_base + eta * b_grad_d * b_k_sq
        if USE_PROJECTION:
            b_d_next = tl.maximum(b_d_next, d_min)
            b_d_next = tl.minimum(b_d_next, d_max)

        # Step 5: rank-1 write using OLD d (kw = k * d)
        b_kw = b_k * b_d
        b_h += b_kw[:, None] * (b_beta * b_e)[None, :]

        # Step 6: output
        b_o = tl.sum(b_h * b_q[:, None], 0)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        # Step 7: commit d update for next step
        b_d = b_d_next

        # Advance pointers
        p_q += H * K
        p_k += H * K
        p_v += H * V
        p_o += H * V
        p_g += H
        p_beta += H
        if DECAY_MODE == 1:  # _DECAY_DD
            p_g_decay += H

    if STORE_FINAL_STATE:
        p_ht = ht + i_nh * K * V + o_k[:, None] * V + o_v[None, :]
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)

    if STORE_FINAL_D:
        if i_v == 0:
            p_dt = dt + i_nh * K + o_k
            tl.store(p_dt, b_d.to(p_dt.dtype.element_ty), mask=mask_k)


def fused_recurrent_os_gated_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    g_decay: torch.Tensor | None,
    scale: float,
    eta: float,
    d_min: float | None,
    d_max: float | None,
    initial_state: torch.Tensor | None,
    initial_d: torch.Tensor | None,
    output_final_state: bool,
    use_qk_l2norm_in_kernel: bool,
    decay_mode: str,
    cu_seqlens: torch.LongTensor | None,
):
    B, T, H, K = k.shape
    V = v.shape[-1]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BK = triton.next_power_of_2(K)
    BV = min(8, triton.next_power_of_2(V))
    NV = triton.cdiv(V, BV)

    if decay_mode == "none":
        _decay = _DECAY_NONE
    elif decay_mode == "data_dependent":
        _decay = _DECAY_DD
        if g_decay is None:
            raise ValueError("decay_mode='data_dependent' requires g_decay")
    else:
        raise ValueError(
            f"fused_recurrent OS-GDN supports decay_mode in "
            f"{{'none', 'data_dependent'}}, got {decay_mode!r}"
        )

    use_projection = (d_min is not None) and (d_max is not None)
    if not use_projection:
        d_min = 0.0
        d_max = 0.0

    o = torch.empty_like(v)
    if output_final_state:
        final_state = q.new_empty(N, H, K, V, dtype=torch.float32)
        final_d = q.new_empty(N, H, K, dtype=torch.float32)
    else:
        final_state = None
        final_d = None

    grid = (NV, N * H)
    fused_recurrent_os_gated_delta_rule_fwd_kernel[grid](
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        g_decay=g_decay,
        o=o,
        h0=initial_state,
        ht=final_state,
        d0=initial_d,
        dt=final_d,
        cu_seqlens=cu_seqlens,
        scale=scale,
        eta=float(eta),
        d_min=float(d_min),
        d_max=float(d_max),
        T=T,
        H=H,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        USE_PROJECTION=use_projection,
        DECAY_MODE=_decay,
        num_warps=1,
        num_stages=3,
    )
    return o, final_state, final_d


class FusedRecurrentOSGatedDeltaRuleFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        g_decay: torch.Tensor | None,
        scale: float,
        eta: float,
        d_min: float | None,
        d_max: float | None,
        initial_state: torch.Tensor | None,
        initial_d: torch.Tensor | None,
        output_final_state: bool,
        use_qk_l2norm_in_kernel: bool,
        decay_mode: str,
        cu_seqlens: torch.LongTensor | None,
    ):
        return fused_recurrent_os_gated_delta_rule_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            g_decay=g_decay,
            scale=scale,
            eta=eta,
            d_min=d_min,
            d_max=d_max,
            initial_state=initial_state,
            initial_d=initial_d,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            decay_mode=decay_mode,
            cu_seqlens=cu_seqlens,
        )

    @staticmethod
    @input_guard
    def backward(ctx, do, dht, ddt):
        raise NotImplementedError(
            "Backward pass is not implemented; this kernel is for "
            "single-stream decode only. Training uses the chunk path."
        )


def fused_recurrent_os_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    g_decay: torch.Tensor | None = None,
    scale: float | None = None,
    eta: float = 0.003,
    d_min: float | None = None,
    d_max: float | None = None,
    initial_state: torch.Tensor | None = None,
    initial_d: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = True,
    decay_mode: str = "none",
    cu_seqlens: torch.LongTensor | None = None,
):
    r"""Single-token-amortized fused recurrent OS-GDN forward.

    Args:
        q, k: [B, T, H, K]
        v:    [B, T, H, V]
        g:    [B, T, H]   raw GDN log-decay (NOT cumsum'd)
        beta: [B, T, H]
        g_decay: [B, T, H] OSGM data-dependent decay log-signal (only required
            when decay_mode == "data_dependent"). When the OS-GDN config sets
            ``osgm_d_decay_source="gdn"`` the layer aliases this to ``g``.
        scale: query scale (default K^{-1/2}).
        eta:   OSGM step size.
        d_min, d_max: optional OSGM clamp bounds. Both must be set together.
        initial_state: [N, H, K, V] fp32 prior recurrent state, or None.
        initial_d:     [N, H, K]    fp32 prior d state, or None (treated as zeros;
            the layer should pass the learned ``initial_scale`` here on the
            first call).
        output_final_state: whether to return (final_state, final_d).
        use_qk_l2norm_in_kernel: l2-normalize q,k inside the kernel; matches the
            chunk path.
        decay_mode: "none" or "data_dependent".
        cu_seqlens: variable-length boundaries; B must equal 1 if provided.

    Returns:
        (o, final_state, final_d). final_state and final_d are None when
        output_final_state=False.
    """
    if cu_seqlens is not None and q.shape[0] != 1:
        raise ValueError(
            f"batch size must be 1 with cu_seqlens, got {q.shape[0]}",
        )
    if scale is None:
        scale = k.shape[-1] ** -0.5
    return FusedRecurrentOSGatedDeltaRuleFunction.apply(
        q, k, v, g, beta, g_decay,
        scale, eta, d_min, d_max,
        initial_state, initial_d,
        output_final_state, use_qk_l2norm_in_kernel, decay_mode,
        cu_seqlens,
    )
