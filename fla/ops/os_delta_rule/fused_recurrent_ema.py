# -*- coding: utf-8 -*-
# Fused recurrent EMA variant — forward only, used for inference (seq_len <= 64).
# Same as fused_recurrent_osgm.py but d is computed via EMA instead of SGD.

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from fla.modules.l2norm import l2norm_fwd
from fla.utils import input_guard

EPS: tl.constexpr = 1e-6


@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None
})
@triton.jit(do_not_specialize=['T'])
def fused_recurrent_ema_fwd_kernel(
    q, k, v, u, d_out, beta, o,
    h0, ht, ema_0, ema_t,
    cu_seqlens, scale, alpha,
    T,
    B: tl.constexpr, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BK: tl.constexpr, BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr, STORE_FINAL_STATE: tl.constexpr,
    IS_BETA_HEADWISE: tl.constexpr, IS_VARLEN: tl.constexpr,
    NORMALIZE: tl.constexpr
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

    # Load initial ema state
    p_ema = ema_0 + i_n * H * K + i_h * K + i_k * BK + tl.arange(0, BK)
    b_ema = tl.load(p_ema, mask=mask_k, other=1.0).to(tl.float32)

    one_minus_alpha = 1.0 - alpha

    for _ in range(0, T):
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32) * scale

        k_sq = b_k * b_k

        # Compute d from ema
        b_d_raw = 1.0 / (b_ema + EPS)
        if NORMALIZE:
            c = tl.sum(b_d_raw * k_sq) + EPS
            b_scale = b_d_raw / c
        else:
            b_scale = b_d_raw

        tl.store(p_d_out, b_scale.to(p_d_out.dtype.element_ty), mask=mask_k)

        # Delta rule: residual + state update
        b_v_minus = tl.sum(b_h * b_k[None, :], axis=1)
        b_v -= b_v_minus
        tl.store(p_u, b_v.to(p_u.dtype.element_ty), mask=mask_v)

        if IS_BETA_HEADWISE:
            b_beta = tl.load(p_beta, mask=mask_v, other=0).to(tl.float32)
        else:
            b_beta = tl.load(p_beta).to(tl.float32)

        b_k_scaled = b_k * b_scale
        b_h += (b_v * b_beta)[:, None] * b_k_scaled[None, :]

        b_o = tl.sum(b_h * b_q[None, :], axis=1)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        # Update ema
        b_ema = alpha * b_ema + one_minus_alpha * k_sq

        p_q += H*K; p_k += H*K; p_o += H*V; p_v += H*V; p_u += H*V; p_d_out += H*K
        p_beta += H * (V if IS_BETA_HEADWISE else 1)

    if STORE_FINAL_STATE:
        p_ht = ht + i_n * H * K * V + i_h * K * V + (i_k * BK + tl.arange(0, BK)[None, :]) * V + (i_v * BV + tl.arange(0, BV)[:, None])
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)
        p_ema_t = ema_t + i_n * H * K + i_h * K + i_k * BK + tl.arange(0, BK)
        tl.store(p_ema_t, b_ema.to(p_ema_t.dtype.element_ty), mask=mask_k)


def fused_recurrent_ema_fwd(
    q, k, v, beta, scale, alpha, normalize,
    initial_state, initial_ema, output_final_state,
    cu_seqlens=None,
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
        final_ema = q.new_empty(N, H, K, dtype=torch.float32)
    else:
        final_state = None
        final_ema = None

    grid = (NV, NK, N * H)
    fused_recurrent_ema_fwd_kernel[grid](
        q, k, v, u, d_out, beta, o,
        initial_state, final_state, initial_ema, final_ema,
        cu_seqlens, scale, alpha,
        T=T, B=B, H=H, K=K, V=V, BK=BK, BV=BV,
        IS_BETA_HEADWISE=beta.ndim == v.ndim,
        NORMALIZE=normalize,
        num_warps=1, num_stages=1,
    )
    o = o.squeeze(0)
    return o, final_state, final_ema


class FusedRecurrentEMAFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    def forward(
        ctx, q, k, v, beta, scale, alpha, normalize,
        initial_state, initial_ema,
        output_final_state, use_qk_l2norm_in_kernel, cu_seqlens,
    ):
        if use_qk_l2norm_in_kernel:
            q, _ = l2norm_fwd(q)
            k, _ = l2norm_fwd(k)

        o, final_state, final_ema = fused_recurrent_ema_fwd(
            q=q, k=k, v=v, beta=beta, scale=scale,
            alpha=alpha, normalize=normalize,
            initial_state=initial_state, initial_ema=initial_ema,
            output_final_state=output_final_state, cu_seqlens=cu_seqlens,
        )

        # Forward-only for inference — no backward through this path
        ctx.mark_non_differentiable(o)
        return o, final_state, final_ema


@torch.compiler.disable
def fused_recurrent_delta_rule_ema(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor = None,
    scale: float = None,
    alpha: float = 0.999,
    normalize: bool = False,
    initial_state: torch.Tensor = None,
    initial_ema: torch.Tensor = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

    if scale is None:
        scale = k.shape[-1] ** -0.5
    if beta is None:
        beta = torch.ones_like(q[..., 0])

    initial_h = None
    if initial_state is not None:
        if isinstance(initial_state, tuple):
            initial_h, initial_ema = initial_state
        else:
            initial_h = initial_state

    if initial_ema is None:
        B, H, K = q.shape[0], q.shape[2], q.shape[3]
        N = B if cu_seqlens is None else len(cu_seqlens) - 1
        # Initialize ema=1.0 so d≈1 and <d,k²>≈1 for L2-normed keys
        initial_ema = q.new_ones(N, H, K, dtype=torch.float32)

    o, final_state, final_ema = FusedRecurrentEMAFunction.apply(
        q, k, v, beta, scale, alpha, normalize,
        initial_h, initial_ema,
        output_final_state, use_qk_l2norm_in_kernel, cu_seqlens,
    )
    if output_final_state:
        return o, (final_state, final_ema)
    return o, None
