# -*- coding: utf-8 -*-
# Chunk-mode EMA variant: replaces OSGM phase1 with EMA-based d computation.
# Phase 2 (delta rule with scaled keys) is identical to OSGM.

from typing import Optional

import torch

from fla.modules.l2norm import l2norm_bwd, l2norm_fwd
from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_bwd_dhu, chunk_gated_delta_rule_fwd_h
from fla.ops.common.chunk_o import chunk_bwd_dv_local, chunk_bwd_dqkwg, chunk_fwd_o
from fla.ops.os_delta_rule.wy_fast_os import prepare_wy_repr_bwd, prepare_wy_repr_fwd, recompute_w_u_fwd
from fla.ops.os_delta_rule.chunk_osgm_phase import fused_osgm_bwd_mapping
from fla.ops.os_delta_rule.chunk_osgm_phase_ema import compute_ema_phase1_fwd
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard


def chunk_delta_rule_fwd(
    q, k, v, d, beta, scale, initial_state, output_final_state, cu_seqlens=None,
):
    kw = k * d
    w, u, A = prepare_wy_repr_fwd(k=k, v=v, beta=beta, d=d, cu_seqlens=cu_seqlens)
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=kw, w=w, u=u, g=None, initial_state=initial_state,
        output_final_state=output_final_state, cu_seqlens=cu_seqlens
    )
    o = chunk_fwd_o(q=q, k=kw, v=v_new, h=h, g=None, scale=scale, cu_seqlens=cu_seqlens)
    return o, A, final_state


def chunk_delta_rule_bwd(
    q, k, v, d, beta, A, scale, initial_state, do, dht, cu_seqlens=None,
):
    kw = k * d
    w, u = recompute_w_u_fwd(k=k, v=v, beta=beta, A=A, cu_seqlens=cu_seqlens)
    h, v_new, _ = chunk_gated_delta_rule_fwd_h(
        k=kw, w=w, u=u, g=None, initial_state=initial_state,
        output_final_state=False, cu_seqlens=cu_seqlens,
    )
    dv = chunk_bwd_dv_local(q=q, k=kw, do=do, g=None, scale=scale, cu_seqlens=cu_seqlens)
    dh, dh0, dv = chunk_gated_delta_rule_bwd_dhu(
        q=q, k=kw, w=w, g=None, h0=initial_state, dht=dht, do=do, dv=dv,
        scale=scale, cu_seqlens=cu_seqlens
    )
    dq, dkw, dw, _ = chunk_bwd_dqkwg(
        q=q, k=kw, v=v_new, h=h, w=w, dv=dv, do=do, dh=dh, g=None,
        scale=scale, cu_seqlens=cu_seqlens
    )
    dk_read, dv, db, dkw_from_A = prepare_wy_repr_bwd(
        k=k, v=v, beta=beta, A=A, d=d, dw=dw, du=dv, cu_seqlens=cu_seqlens
    )
    dk, dd = fused_osgm_bwd_mapping(dkw, dkw_from_A, k, d, dk_read)
    return dq, dk, dv, db, dh0, dd


class ChunkDeltaRuleEMAFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx, q, k, v, beta, scale, alpha, normalize,
        initial_h, initial_ema, output_final_state,
        use_qk_l2norm_in_kernel, cu_seqlens,
    ):
        if use_qk_l2norm_in_kernel:
            q, q_rstd = l2norm_fwd(q)
            k, k_rstd = l2norm_fwd(k)
        else:
            q_rstd, k_rstd = None, None

        # Phase 1: compute d from keys via EMA
        # d is treated as a non-differentiable running statistic (like Adam/BatchNorm).
        # Gradient through 1/(ema+eps)² is O(K²) per step and explodes over sequence length,
        # so we detach d and only let gradients flow through phase 2 (via k*d, q, v).
        d, _, final_ema = compute_ema_phase1_fwd(
            k, alpha, normalize, cu_seqlens, initial_ema, output_final_state
        )

        # Phase 2: standard delta rule with scaled keys
        o, A, final_h = chunk_delta_rule_fwd(
            q=q, k=k, v=v, d=d, beta=beta, scale=scale,
            initial_state=initial_h,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
        )

        ctx.save_for_backward(q, q_rstd, k, k_rstd, v, d, beta, A, initial_h)
        ctx.scale = scale
        ctx.cu_seqlens = cu_seqlens
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel

        return o.to(q.dtype), final_h, final_ema

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do, dh_final, d_final_ema):
        q, q_rstd, k, k_rstd, v, d, beta, A, initial_h = ctx.saved_tensors

        # Phase 2 backward only — d is detached (non-differentiable EMA statistic).
        # Gradients flow through k*d → k (with d as constant), q, v, beta.
        dq, dk_phase2, dv, db, dh0, dd = chunk_delta_rule_bwd(
            q=q, k=k, v=v, d=d, beta=beta, A=A, scale=ctx.scale,
            initial_state=initial_h, do=do, dht=dh_final, cu_seqlens=ctx.cu_seqlens
        )
        # dd is discarded — no backward through EMA phase 1.

        dk = dk_phase2

        if ctx.use_qk_l2norm_in_kernel:
            dq = l2norm_bwd(q, q_rstd, dq)
            dk = l2norm_bwd(k, k_rstd, dk)

        return (
            dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), db.to(beta.dtype),
            None, None, None,  # scale, alpha, normalize
            dh0, None,  # initial_h, initial_ema (no grad)
            None, None, None,  # output_final_state, use_qk_l2norm, cu_seqlens
        )


@torch.compiler.disable
def chunk_delta_rule_ema(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, beta: torch.Tensor,
    scale: float = None, alpha: float = 0.999, normalize: bool = False,
    initial_state=None, output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
):
    scale = k.shape[-1] ** -0.5 if scale is None else scale

    initial_h, initial_ema = None, None
    if initial_state is not None:
        if isinstance(initial_state, tuple):
            initial_h, initial_ema = initial_state
        else:
            initial_h = initial_state

    o, final_h, final_ema = ChunkDeltaRuleEMAFunction.apply(
        q, k, v, beta, scale, alpha, normalize,
        initial_h, initial_ema, output_final_state,
        use_qk_l2norm_in_kernel, cu_seqlens,
    )

    if output_final_state:
        return o, (final_h, final_ema)
    return o, None
