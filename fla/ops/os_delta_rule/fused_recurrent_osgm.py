# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# Copyright (c) 2026, Hongpei Li

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
def fused_recurrent_delta_rule_fwd_kernel(
    q, k, v, u, d_out, beta, o,
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
        
        k_sq = b_k * b_k
        inner_prod = tl.sum(b_scale * k_sq)
        term_A = 1.0 - inner_prod
        
        if USE_DENOMINATOR:
            sum_k_sq = tl.sum(k_sq) + 1e-5
            grad_d = term_A / sum_k_sq
        else:
            grad_d = term_A
            
        d_next = b_scale + eta * grad_d * k_sq

        if USE_PROJECTION:
            d_next = tl.maximum(d_next, d_min)
            d_next = tl.minimum(d_next, d_max)

        b_scale = d_next

        p_q += H*K; p_k += H*K; p_o += H*V; p_v += H*V; p_u += H*V; p_d_out += H*K
        p_beta += H * (V if IS_BETA_HEADWISE else 1)

    if STORE_FINAL_STATE:
        p_ht = ht + i_n * H * K * V + i_h * K * V + (i_k * BK + tl.arange(0, BK)[None, :]) * V + (i_v * BV + tl.arange(0, BV)[:, None])
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)
        p_scale_t = scale_t + i_n * H * K + i_h * K + i_k * BK + tl.arange(0, BK)
        tl.store(p_scale_t, b_scale.to(p_scale_t.dtype.element_ty), mask=mask_k)


@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'USE_FINAL_STATE_GRADIENT': lambda args: args['dht'] is not None,
    'USE_FINAL_SCALE_GRADIENT': lambda args: args['d_scale_t'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None
})
@triton.jit(do_not_specialize=['T'])
def fused_recurrent_delta_rule_bwd_kernel(
    q, k, v, d_out, beta, h0, dh0, dht, do, dq, dk, dv, db,
    scale_0, d_scale_0, d_scale_t, cu_seqlens, scale, eta, d_min, d_max,
    B: tl.constexpr, T, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BK: tl.constexpr, BV: tl.constexpr, NK: tl.constexpr,
    IS_BETA_HEADWISE: tl.constexpr, USE_INITIAL_STATE: tl.constexpr,
    USE_FINAL_STATE_GRADIENT: tl.constexpr, USE_FINAL_SCALE_GRADIENT: tl.constexpr, IS_VARLEN: tl.constexpr,
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

    mask_k = i_k * BK + tl.arange(0, BK) < K
    mask_v = i_v * BV + tl.arange(0, BV) < V

    p_q = q + (bos * H + i_h) * K + i_k * BK + tl.arange(0, BK) + (T - 1) * H*K
    p_k = k + (bos * H + i_h) * K + i_k * BK + tl.arange(0, BK) + (T - 1) * H*K
    p_v = v + (bos * H + i_h) * V + i_v * BV + tl.arange(0, BV) + (T - 1) * H*V
    p_do = do + (bos * H + i_h) * V + i_v * BV + tl.arange(0, BV) + (T - 1) * H*V
    p_dk = dk + ((i_v * all_t + bos) * H + i_h) * K + i_k * BK + tl.arange(0, BK) + (T - 1) * H*K
    p_dv = dv + ((i_k * all_t + bos) * H + i_h) * V + i_v * BV + tl.arange(0, BV) + (T - 1) * H*V
    p_d_out = d_out + (bos * H + i_h) * K + i_k * BK + tl.arange(0, BK) + (T - 1) * H*K
    
    if IS_BETA_HEADWISE:
        p_beta = beta + (bos + T - 1) * H*V + i_h * V + i_v * BV + tl.arange(0, BV)
        p_dbeta = db + ((i_v * NK + i_k) * all_t + bos + T - 1) * H*V + i_h * V + tl.arange(0, BV)
    else:
        p_beta = beta + (bos + T - 1) * H + i_h
        p_dbeta = db + (i_v * all_t + bos + T - 1) * H + i_h

    b_dh = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_FINAL_STATE_GRADIENT:
        p_dht = dht + i_n * H * K * V + i_h * K * V + (i_k * BK + tl.arange(0, BK)[:, None]) * V + (i_v * BV + tl.arange(0, BV)[None, :])
        b_dh += tl.load(p_dht, mask=mask_k[:, None] & mask_v[None, :], other=0).to(tl.float32)

    # 💥 核心：将末尾的 d 梯度接收进来，不再是纯 0！
    b_dd = tl.zeros([BK], dtype=tl.float32)
    if USE_FINAL_SCALE_GRADIENT:
        p_dst = d_scale_t + i_n * H * K + i_h * K + i_k * BK + tl.arange(0, BK)
        b_dd += tl.load(p_dst, mask=mask_k, other=0).to(tl.float32)

    for _ in range(T):
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32) * scale
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_u = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_do = tl.load(p_do, mask=mask_v, other=0).to(tl.float32)
        b_d = tl.load(p_d_out, mask=mask_k, other=0).to(tl.float32)
        
        if IS_BETA_HEADWISE:
            b_beta = tl.load(p_beta, mask=mask_v, other=0).to(tl.float32)
        else:
            b_beta = tl.load(p_beta).to(tl.float32)
        
        b_dh += b_q[:, None] * b_do[None, :]
        
        b_p = b_k * b_d  
        b_beta_u = b_u * b_beta  
        
        b_d_beta_u = tl.sum(b_dh * b_p[:, None], axis=0)  
        b_du = b_d_beta_u * b_beta  
        b_db = b_d_beta_u * b_u if IS_BETA_HEADWISE else tl.sum(b_d_beta_u * b_u)  
        
        b_dp = tl.sum(b_dh * b_beta_u[None, :], axis=1)  
        b_dk_from_p = b_dp * b_d  

        k_sq = b_k * b_k
        term_A = 1.0 - tl.sum(b_d * k_sq)
        
        if USE_DENOMINATOR:
            sum_k_sq = tl.sum(k_sq) + 1e-5
            grad_d = term_A / sum_k_sq
        else:
            grad_d = term_A
            
        if USE_PROJECTION:
            d_next_pre = b_d + eta * grad_d * k_sq
            mask_proj = (d_next_pre >= d_min) & (d_next_pre <= d_max)
            b_dd = tl.where(mask_proj, b_dd, 0.0)

        term_B = tl.sum(b_dd * k_sq)

        if USE_DENOMINATOR:
            grad_w = (eta / sum_k_sq) * (-term_B * b_d + term_A * b_dd - (term_A * term_B / sum_k_sq))
            b_dd = b_dd - eta * (term_B / sum_k_sq) * k_sq + b_dp * b_k
        else:
            grad_w = eta * (term_A * b_dd - term_B * b_d)
            b_dd = b_dd - eta * term_B * k_sq + b_dp * b_k
        
        b_dk_from_osgm = 2.0 * b_k * grad_w

        b_dk = b_dk_from_p + b_dk_from_osgm  
        
        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), mask=mask_k)
        tl.store(p_dv, b_du.to(p_dv.dtype.element_ty), mask=mask_v)
        if IS_BETA_HEADWISE:
            tl.store(p_dbeta, b_db.to(p_dbeta.dtype.element_ty), mask=mask_v)
        else:
            tl.store(p_dbeta, b_db.to(p_dbeta.dtype.element_ty))

        b_dh -= b_k[:, None] * b_du[None, :]  

        p_q -= H*K; p_k -= H*K; p_v -= H*V; p_do -= H*V; p_dk -= H*K; p_dv -= H*V; p_d_out -= H*K
        p_dbeta -= H * (V if IS_BETA_HEADWISE else 1); p_beta -= H * (V if IS_BETA_HEADWISE else 1)

    if USE_INITIAL_STATE:
        p_dh0 = dh0 + i_n * H * K * V + i_h * K * V + (i_k * BK + tl.arange(0, BK)[:, None]) * V + (i_v * BV + tl.arange(0, BV)[None, :])
        tl.store(p_dh0, b_dh.to(p_dh0.dtype.element_ty), mask=mask_k[:, None] & mask_v[None, :])

    p_ds0 = d_scale_0 + (i_v * (B if not IS_VARLEN else 1) * H + i_n * H + i_h) * K + i_k * BK + tl.arange(0, BK)
    tl.store(p_ds0, b_dd.to(p_ds0.dtype.element_ty), mask=mask_k)

    tl.debug_barrier()
    
    b_h = tl.zeros([BV, BK], dtype=tl.float32)  
    
    p_k = k + (bos * H + i_h) * K + i_k * BK + tl.arange(0, BK)
    p_v = v + (bos * H + i_h) * V + i_v * BV + tl.arange(0, BV)
    p_do = do + (bos * H + i_h) * V + i_v * BV + tl.arange(0, BV)
    p_dq = dq + ((i_v * all_t + bos) * H + i_h) * K + i_k * BK + tl.arange(0, BK)
    p_dk = dk + ((i_v * all_t + bos) * H + i_h) * K + i_k * BK + tl.arange(0, BK)
    p_dv = dv + ((i_k * all_t + bos) * H + i_h) * V + i_v * BV + tl.arange(0, BV)
    p_d_out = d_out + (bos * H + i_h) * K + i_k * BK + tl.arange(0, BK)
    
    if IS_BETA_HEADWISE:
        p_beta = beta + (bos * H + i_h) * V + i_v * BV + tl.arange(0, BV)
    else:
        p_beta = beta + bos * H + i_h

    if USE_INITIAL_STATE:
        p_h0 = h0 + i_n * H * K * V + i_h * K * V + (i_k * BK + tl.arange(0, BK)[None, :]) * V + (i_v * BV + tl.arange(0, BV)[:, None])
        b_h += tl.load(p_h0, mask=mask_k[:, None] & mask_v[None, :], other=0).to(tl.float32)

    for _ in range(T):
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_u = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_do = tl.load(p_do, mask=mask_v, other=0).to(tl.float32)
        b_du = tl.load(p_dv, mask=mask_v, other=0).to(tl.float32)
        b_dk = tl.load(p_dk, mask=mask_k, other=0).to(tl.float32)
        b_d = tl.load(p_d_out, mask=mask_k, other=0).to(tl.float32)
        
        if IS_BETA_HEADWISE:
            b_beta = tl.load(p_beta, mask=mask_v, other=0).to(tl.float32)
        else:
            b_beta = tl.load(p_beta).to(tl.float32)

        b_dk_from_u = -tl.sum(b_h * b_du[:, None], axis=0)  
        b_dk += b_dk_from_u
        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), mask=mask_k)

        b_k_osgm = b_k * b_d
        b_h += (b_u * b_beta)[:, None] * b_k_osgm[None, :]

        b_dq = tl.sum(b_h * b_do[:, None], axis=0) * scale  
        tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), mask=mask_k)

        p_k += H*K; p_v += H*V; p_do += H*V; p_dq += H*K; p_dk += H*K; p_dv += H*V; p_d_out += H*K
        p_beta += H * (V if IS_BETA_HEADWISE else 1)


def fused_recurrent_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    eta: float,
    initial_state: torch.Tensor,
    initial_scale: torch.Tensor,
    output_final_state: bool,
    cu_seqlens: Optional[torch.LongTensor] = None,
    use_denominator: bool = False,
    d_min: float = None,
    d_max: float = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *k.shape, v.shape[-1]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BK, BV = triton.next_power_of_2(K), min(triton.next_power_of_2(V), 8)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
    assert NK == 1, "NK > 1 is not supported yet"
    num_stages = 1
    num_warps = 1

    o = q.new_empty(NK, *v.shape)
    d_out = torch.empty_like(k) 
    
    if output_final_state:
        final_state = q.new_empty(N, H, K, V, dtype=torch.float32)
        final_scale = q.new_empty(N, H, K, dtype=torch.float32)
    else:
        final_state = None
        final_scale = None

    USE_PROJECTION = (d_min is not None) and (d_max is not None)
    d_min_val = float(d_min) if d_min is not None else 0.0
    d_max_val = float(d_max) if d_max is not None else 0.0

    grid = (NV, NK, N * H)
    u = torch.empty_like(v)
    fused_recurrent_delta_rule_fwd_kernel[grid](
        q, k, v, u, d_out, beta, o,
        initial_state, final_state, initial_scale, final_scale,
        cu_seqlens, scale, eta, d_min_val, d_max_val,
        T=T, B=B, H=H, K=K, V=V, BK=BK, BV=BV,
        IS_BETA_HEADWISE=beta.ndim == v.ndim,
        num_warps=num_warps, num_stages=num_stages,
        USE_DENOMINATOR=use_denominator, USE_PROJECTION=USE_PROJECTION
    )
    o = o.squeeze(0)
    return o, u, d_out, final_state, final_scale


def fused_recurrent_delta_rule_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    d_out: torch.Tensor,
    beta: torch.Tensor,
    dht: torch.Tensor,
    d_scale_t: torch.Tensor,
    do: torch.Tensor,
    scale: float,
    eta: float,
    initial_state: torch.Tensor,
    initial_scale: torch.Tensor,
    cu_seqlens: Optional[torch.LongTensor] = None,
    use_denominator: bool = False,
    d_min: float = None,
    d_max: float = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *k.shape, v.shape[-1]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BK, BV = triton.next_power_of_2(K), min(triton.next_power_of_2(V), 32)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
    assert NK == 1, "NK > 1 is not supported yet"
    num_stages = 1
    num_warps = 2

    beta_vector = beta.ndim == v.ndim

    dq = q.new_empty(NV, *q.shape)
    dk = q.new_empty(NV, *k.shape)
    dv = q.new_empty(NK, *v.shape)
    if beta_vector:
        db = q.new_empty(NV, NK, B, T, H, V)
    else:
        db = q.new_empty(NV, B, T, H)
    grid = (NV, NK, N * H)

    if initial_state is not None and initial_state.requires_grad:
        dh0 = torch.empty_like(initial_state, dtype=torch.float32)
    else:
        dh0 = None

    d_scale_0 = q.new_empty(NV, N, H, K, dtype=torch.float32)

    USE_PROJECTION = (d_min is not None) and (d_max is not None)
    d_min_val = float(d_min) if d_min is not None else 0.0
    d_max_val = float(d_max) if d_max is not None else 0.0

    fused_recurrent_delta_rule_bwd_kernel[grid](
        q, k, v, d_out, beta, initial_state, dh0, dht, do, dq, dk, dv, db,
        initial_scale, d_scale_0, d_scale_t, cu_seqlens, scale, eta, d_min_val, d_max_val,
        T=T, B=B, H=H, K=K, V=V, BK=BK, BV=BV, NK=NK,
        IS_BETA_HEADWISE=beta_vector,
        num_warps=num_warps, num_stages=num_stages,
        USE_DENOMINATOR=use_denominator, USE_PROJECTION=USE_PROJECTION
    )
    dq = dq.sum(0)
    dk = dk.sum(0)
    dv = dv.sum(0)
    db = db.sum((0, 1)) if beta_vector else db.sum(0)
    d_scale_0 = d_scale_0.sum(0)

    return dq, dk, dv, db, dh0, d_scale_0


class FusedRecurrentFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    def forward(
        ctx, q, k, v, beta, scale, eta, initial_state, initial_scale,
        output_final_state, use_qk_l2norm_in_kernel, cu_seqlens,
        use_denominator, d_min, d_max
    ):
        if use_qk_l2norm_in_kernel:
            q, q_rstd = l2norm_fwd(q)
            k, k_rstd = l2norm_fwd(k)
        else:
            q_rstd, k_rstd = None, None
            
        o, u, d_out, final_state, final_scale = fused_recurrent_delta_rule_fwd(
            q=q, k=k, v=v, beta=beta, scale=scale, eta=eta,
            initial_state=initial_state, initial_scale=initial_scale,
            output_final_state=output_final_state, cu_seqlens=cu_seqlens,
            use_denominator=use_denominator, d_min=d_min, d_max=d_max
        )

        ctx.save_for_backward(q, q_rstd, k, k_rstd, u, d_out, beta, initial_state, initial_scale)
        ctx.scale = scale
        ctx.eta = eta
        ctx.cu_seqlens = cu_seqlens
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        ctx.use_denominator = use_denominator
        ctx.d_min = d_min
        ctx.d_max = d_max
        return o, final_state, final_scale

    @staticmethod
    @input_guard
    def backward(ctx, do, dht, d_final_scale):
        q, q_rstd, k, k_rstd, u, d_out, beta, initial_state, initial_scale = ctx.saved_tensors
        
        dq, dk, dv, db, dh0, d_scale_0 = fused_recurrent_delta_rule_bwd(
            q=q, k=k, v=u, d_out=d_out, beta=beta, dht=dht, d_scale_t=d_final_scale, do=do,
            scale=ctx.scale, eta=ctx.eta,
            initial_state=initial_state, initial_scale=initial_scale,
            cu_seqlens=ctx.cu_seqlens,
            use_denominator=ctx.use_denominator, d_min=ctx.d_min, d_max=ctx.d_max
        )

        if ctx.use_qk_l2norm_in_kernel:
            dq = l2norm_bwd(q, q_rstd, dq)
            dk = l2norm_bwd(k, k_rstd, dk)

        return dq.to(q), dk.to(k), dv.to(u), db.to(beta), None, None, dh0, d_scale_0, None, None, None, None, None, None


@torch.compiler.disable
def fused_recurrent_delta_rule_osgm(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor = None,
    scale: float = None,
    eta: float = None,
    initial_state: torch.Tensor = None,
    initial_scale: torch.Tensor = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
    use_denominator: bool = None,
    d_min: float = None,
    d_max: float = None,
)-> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    
    if use_qk_l2norm_in_kernel:
        if eta is None: eta = 1.0
        if use_denominator is None: use_denominator = False
        if d_min is None: d_min = 0.0
        if d_max is None: d_max = 2.0
    else:
        if eta is None: eta = 0.1 
        if use_denominator is None: use_denominator = True
    
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(f"Batch size must be 1 for varlen.")
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(f"Initial state dimension mismatch.")
            
    if scale is None: scale = k.shape[-1] ** -0.5
    else: assert scale > 0, "scale must be positive"
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
        
    o, final_state, final_scale = FusedRecurrentFunction.apply(
        q, k, v, beta, scale, eta,
        initial_h, initial_scale,
        output_final_state, use_qk_l2norm_in_kernel, cu_seqlens,
        use_denominator, d_min, d_max
    )
    if output_final_state:
        return o, (final_state, final_scale)
    return o, None