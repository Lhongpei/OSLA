# -*- coding: utf-8 -*-
# Fused recurrent OSGM with data-dependent (per-token per-head) decay on preconditioner D.
# Used for inference (seq_len <= 64).

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from fla.modules.l2norm import l2norm_bwd, l2norm_fwd
from fla.utils import input_guard


@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None
})
@triton.jit(do_not_specialize=['T'])
def fused_recurrent_osgm_dd_decay_fwd_kernel(
    q, k, v, u, d_out, beta, g_ptr, o,
    h0, ht, scale_0, scale_t,
    cu_seqlens, scale, eta, d_min, d_max,
    T,
    B: tl.constexpr, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BK: tl.constexpr, BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr, STORE_FINAL_STATE: tl.constexpr,
    IS_BETA_HEADWISE: tl.constexpr, IS_VARLEN: tl.constexpr,
    USE_DENOMINATOR: tl.constexpr, USE_PROJECTION: tl.constexpr
):
    i_v, i_k, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_h = i_nh // H, i_nh % H

    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        all_t = T
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
        all_t = B * T

    p_q = q + (bos * H + i_h) * K + i_k * BK + tl.arange(0, BK)
    p_k = k + (bos * H + i_h) * K + i_k * BK + tl.arange(0, BK)
    p_v = v + (bos * H + i_h) * V + i_v * BV + tl.arange(0, BV)
    p_u = u + (bos * H + i_h) * V + i_v * BV + tl.arange(0, BV)
    p_d_out = d_out + (bos * H + i_h) * K + i_k * BK + tl.arange(0, BK)
    p_o = o + ((i_k * all_t + bos) * H + i_h) * V + i_v * BV + tl.arange(0, BV)

    # g pointer: [B, T, H] layout
    p_g = g_ptr + bos * H + i_h

    if IS_BETA_HEADWISE:
        p_beta = beta + (bos * H + i_h) * V + i_v * BV + tl.arange(0, BV)
    else:
        p_beta = beta + bos * H + i_h

    mask_k = (i_k * BK + tl.arange(0, BK)) < K
    mask_v = (i_v * BV + tl.arange(0, BV)) < V
    mask_h = mask_k[None, :] & mask_v[:, None]

    b_h = tl.zeros([BV, BK], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = h0 + i_n * H * K * V + i_h * K * V + (i_k * BK + tl.arange(0, BK)[None, :]) * V + (i_v * BV + tl.arange(0, BV)[:, None])
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    p_scale = scale_0 + i_n * H * K + i_h * K + i_k * BK + tl.arange(0, BK)
    b_scale = tl.load(p_scale, mask=mask_k, other=0).to(tl.float32)

    for _ in range(0, T):
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32) * scale

        tl.store(p_d_out, b_scale.to(p_d_out.dtype.element_ty), mask=mask_k)

        b_v_minus = tl.sum(b_h * b_k[None, :], axis=1)
        b_v -= b_v_minus
        tl.store(p_u, b_v.to(p_u.dtype.element_ty), mask=mask_v)

        if IS_BETA_HEADWISE:
            b_beta = tl.load(p_beta, mask=mask_v, other=0).to(tl.float32)
        else:
            b_beta = tl.load(p_beta).to(tl.float32)

        b_k_osgm = b_k * b_scale
        b_h += (b_v * b_beta)[:, None] * b_k_osgm[None, :]

        b_o = tl.sum(b_h * b_q[None, :], axis=1)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        # Load per-token decay
        b_g = tl.load(p_g).to(tl.float32)
        b_gamma = tl.exp(b_g)

        k_sq = b_k * b_k
        inner_prod = tl.sum(b_scale * k_sq)
        term_A = 1.0 - inner_prod

        if USE_DENOMINATOR:
            sum_k_sq = tl.sum(k_sq) + 1e-5
            grad_d = term_A / sum_k_sq
        else:
            grad_d = term_A

        d_next = b_gamma * b_scale + eta * grad_d * k_sq

        if USE_PROJECTION:
            d_next = tl.maximum(d_next, d_min)
            d_next = tl.minimum(d_next, d_max)

        b_scale = d_next

        p_q += H*K; p_k += H*K; p_o += H*V; p_v += H*V; p_u += H*V; p_d_out += H*K
        p_beta += H * (V if IS_BETA_HEADWISE else 1)
        p_g += H

    if STORE_FINAL_STATE:
        p_ht = ht + i_n * H * K * V + i_h * K * V + (i_k * BK + tl.arange(0, BK)[None, :]) * V + (i_v * BV + tl.arange(0, BV)[:, None])
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)
        p_scale_t = scale_t + i_n * H * K + i_h * K + i_k * BK + tl.arange(0, BK)
        tl.store(p_scale_t, b_scale.to(p_scale_t.dtype.element_ty), mask=mask_k)


def fused_recurrent_osgm_dd_decay_fwd(
    q, k, v, beta, g, scale, eta,
    initial_state, initial_scale, output_final_state,
    cu_seqlens=None, use_denominator=False, d_min=None, d_max=None,
):
    B, T, H, K, V = *k.shape, v.shape[-1]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BK, BV = triton.next_power_of_2(K), min(triton.next_power_of_2(V), 8)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
    assert NK == 1

    o = q.new_empty(NK, *v.shape)
    d_out = torch.empty_like(k)
    u = torch.empty_like(v)

    if output_final_state:
        final_state = q.new_empty(N, H, K, V, dtype=torch.float32)
        final_scale = q.new_empty(N, H, K, dtype=torch.float32)
    else:
        final_state = None
        final_scale = None

    USE_PROJECTION = (d_min is not None) and (d_max is not None)

    grid = (NV, NK, N * H)
    fused_recurrent_osgm_dd_decay_fwd_kernel[grid](
        q, k, v, u, d_out, beta, g, o,
        initial_state, final_state, initial_scale, final_scale,
        cu_seqlens, scale, eta,
        float(d_min) if d_min is not None else 0.0,
        float(d_max) if d_max is not None else 0.0,
        T=T, B=B, H=H, K=K, V=V, BK=BK, BV=BV,
        IS_BETA_HEADWISE=beta.ndim == v.ndim,
        num_warps=1, num_stages=1,
        USE_DENOMINATOR=use_denominator, USE_PROJECTION=USE_PROJECTION
    )
    o = o.squeeze(0)
    return o, u, d_out, final_state, final_scale


class FusedRecurrentDDDecayFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    def forward(
        ctx, q, k, v, beta, g, scale, eta,
        initial_state, initial_scale,
        output_final_state, use_qk_l2norm_in_kernel, cu_seqlens,
        use_denominator, d_min, d_max
    ):
        if use_qk_l2norm_in_kernel:
            q, q_rstd = l2norm_fwd(q)
            k, k_rstd = l2norm_fwd(k)
        else:
            q_rstd, k_rstd = None, None

        o, u, d_out, final_state, final_scale = fused_recurrent_osgm_dd_decay_fwd(
            q=q, k=k, v=v, beta=beta, g=g,
            scale=scale, eta=eta,
            initial_state=initial_state, initial_scale=initial_scale,
            output_final_state=output_final_state, cu_seqlens=cu_seqlens,
            use_denominator=use_denominator, d_min=d_min, d_max=d_max
        )

        # Forward-only kernel for inference (no backward needed).
        ctx.mark_non_differentiable(o)
        return o, final_state, final_scale


@torch.compiler.disable
def fused_recurrent_delta_rule_osgm_dd_decay(
    q, k, v, beta=None, g=None,
    scale=None, eta=None,
    initial_state=None, initial_scale=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
    cu_seqlens=None,
    use_denominator=None, d_min=None, d_max=None,
):
    if use_qk_l2norm_in_kernel:
        if eta is None: eta = 1.0
        if use_denominator is None: use_denominator = False
        if d_min is None: d_min = 0.0
        if d_max is None: d_max = 1e9
    else:
        if eta is None: eta = 0.1
        if use_denominator is None: use_denominator = True

    if scale is None: scale = k.shape[-1] ** -0.5
    if beta is None: beta = torch.ones_like(q[..., 0])

    initial_h = None
    if initial_state is not None:
        if isinstance(initial_state, tuple):
            initial_h, initial_scale = initial_state
        else:
            initial_h = initial_state

    if initial_scale is None:
        B, H, K = q.shape[0], q.shape[2], q.shape[3]
        N = B if cu_seqlens is None else len(cu_seqlens) - 1
        initial_scale = q.new_zeros(N, H, K, dtype=torch.float32)

    o, final_state, final_scale = FusedRecurrentDDDecayFunction.apply(
        q, k, v, beta, g, scale, eta,
        initial_h, initial_scale,
        output_final_state, use_qk_l2norm_in_kernel, cu_seqlens,
        use_denominator, d_min, d_max
    )
    if output_final_state:
        return o, (final_state, final_scale)
    return o, None
