# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# Copyright (c) 2026, Hongpei Li

import warnings
from typing import Optional

import torch

from fla.modules.l2norm import l2norm_bwd, l2norm_fwd
from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_bwd_dhu, chunk_gated_delta_rule_fwd_h
from fla.ops.common.chunk_o import chunk_bwd_dqkwg, chunk_bwd_dv_local, chunk_fwd_o
from fla.ops.osla_delta_rule.wy_fast_osla import prepare_wy_repr_bwd, prepare_wy_repr_fwd, recompute_w_u_fwd
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard
from fla.ops.osla_delta_rule.chunk_osgm_phase import compute_osgm_phase1_fwd, compute_osgm_phase1_bwd, fused_osgm_bwd_mapping
def chunk_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    d: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    output_final_state: bool,
    cu_seqlens: Optional[torch.LongTensor] = None,
):
    kw = k * d 

    w, u, A = prepare_wy_repr_fwd(k=k, v=v, beta=beta, d=d, cu_seqlens=cu_seqlens)
    
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=kw, w=w, u=u, g=None, initial_state=initial_state,
        output_final_state=output_final_state, cu_seqlens=cu_seqlens
    )

    o = chunk_fwd_o(
        q=q, k=kw, v=v_new, h=h, g=None, scale=scale, cu_seqlens=cu_seqlens
    )
    return o, A, final_state


def chunk_delta_rule_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    d: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor,
    cu_seqlens: Optional[torch.LongTensor] = None,
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


class ChunkDeltaRuleFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx, q, k, v, beta, scale, eta, initial_state, output_final_state,
        use_qk_l2norm_in_kernel, cu_seqlens, use_denominator, d_min, d_max, 
    ):
        if use_qk_l2norm_in_kernel:
            q, q_rstd = l2norm_fwd(q)
            k, k_rstd = l2norm_fwd(k)
        else:
            q_rstd, k_rstd = None, None

        d = compute_osgm_phase1_fwd(k, eta, use_denominator, d_min, d_max)

        o, A, final_state = chunk_delta_rule_fwd(
            q=q, k=k, v=v, d=d, beta=beta, scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
        )
        
        ctx.save_for_backward(q, q_rstd, k, k_rstd, v, d, beta, A, initial_state)
        ctx.scale = scale; ctx.eta = eta; ctx.cu_seqlens = cu_seqlens
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        ctx.use_denominator = use_denominator; ctx.d_min = d_min; ctx.d_max = d_max
        return o.to(q.dtype), final_state

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do, dht):
        q, q_rstd, k, k_rstd, v, d, beta, A, initial_state = ctx.saved_tensors

        dq, dk_phase2, dv, db, dh0, dd = chunk_delta_rule_bwd(
            q=q, k=k, v=v, d=d, beta=beta, A=A, scale=ctx.scale,
            initial_state=initial_state, do=do, dht=dht, cu_seqlens=ctx.cu_seqlens
        )
        
        dk_phase1 = compute_osgm_phase1_bwd(
            k, d, dd, ctx.eta, ctx.use_denominator, ctx.d_min, ctx.d_max
        )
        dk = dk_phase2 + dk_phase1

        if ctx.use_qk_l2norm_in_kernel:
            dq = l2norm_bwd(q, q_rstd, dq)
            dk = l2norm_bwd(k, k_rstd, dk)
            
        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), db.to(beta.dtype), None, None, dh0, None, None, None, None, None, None


@torch.compiler.disable
def chunk_delta_rule_osgm(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, beta: torch.Tensor,
    scale: float = None, eta: float = None,
    initial_state: torch.Tensor = None, output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
    use_denominator: bool = None,
    d_min: float = None,
    d_max: float = None,
    head_first: bool = False,
):
    if use_qk_l2norm_in_kernel:
        if eta is None: eta = 1.0
        if use_denominator is None: use_denominator = False
        if d_min is None: d_min = 0.0
        if d_max is None: d_max = 2.0
    else:
        if eta is None: eta = 0.1 
        if use_denominator is None: use_denominator = True

    scale = k.shape[-1] ** -0.5 if scale is None else scale
    o, final_state = ChunkDeltaRuleFunction.apply(
        q, k, v, beta, scale, eta, initial_state, output_final_state,
        use_qk_l2norm_in_kernel, cu_seqlens,
        use_denominator, d_min, d_max
    )
    return o, final_state