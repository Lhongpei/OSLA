# -*- coding: utf-8 -*-

import torch
import triton
import triton.language as tl
from fla.ops.os_delta_rule.chunk_scaled_dot_qkw import chunk_scaled_dot_qkw_fwd
from fla.ops.utils.solve_tril import solve_tril

@triton.jit
def osgm_phase1_wy_cross_chunk_scan_kernel(
    s_ptr, u_ptr, W_ptr, d_starts_ptr, cu_seqlens_ptr,
    initial_d_ptr, final_d_ptr, 
    d_min, d_max,
    stride_id_b, stride_id_h, stride_id_k,
    stride_kb, stride_kh, stride_kt,
    stride_wb, stride_wh, stride_wt, stride_wc,
    stride_db, stride_dh, stride_dn,
    T: tl.constexpr, H: tl.constexpr, K: tl.constexpr, BK: tl.constexpr, BT: tl.constexpr,
    IS_VARLEN: tl.constexpr, USE_PROJECTION: tl.constexpr, 
    USE_INITIAL_STATE: tl.constexpr, USE_FINAL_STATE: tl.constexpr
):
    i_nh = tl.program_id(0)
    i_n = i_nh // H
    i_h = i_nh % H

    if IS_VARLEN:
        bos = tl.load(cu_seqlens_ptr + i_n).to(tl.int32)
        eos = tl.load(cu_seqlens_ptr + i_n + 1).to(tl.int32)
        i_b = 0; seq_len = eos - bos
    else:
        bos = 0; eos = T
        i_b = i_n; seq_len = T
        
    seq_NT = tl.cdiv(seq_len, BT)
    offset_k = tl.arange(0, BK)
    offset_c = tl.arange(0, BT)
    mask_k = offset_k < K

    p_s = s_ptr + i_b * stride_kb + i_h * stride_kh + bos * stride_kt + offset_c[:, None] * stride_kt + offset_k[None, :]
    p_u = u_ptr + i_b * stride_kb + i_h * stride_kh + bos * stride_kt + offset_c[:, None] * stride_kt + offset_k[None, :]

    if USE_INITIAL_STATE:
        p_initial_d = initial_d_ptr + i_n * stride_id_b + i_h * stride_id_h + offset_k * stride_id_k
        d_curr = tl.load(p_initial_d, mask=mask_k, other=0.0).to(tl.float32)
    else:
        d_curr = tl.zeros([BK], dtype=tl.float32)

    for i_c in range(seq_NT):
        chunk_idx = (bos // BT) + i_c
        
        p_d_starts = d_starts_ptr + i_b * stride_db + i_h * stride_dh + chunk_idx * stride_dn + offset_k
        tl.store(p_d_starts, d_curr.to(p_d_starts.dtype.element_ty), mask=mask_k)

        mask_t = (i_c * BT + offset_c) < seq_len
        mask_st = mask_t[:, None] & mask_k[None, :]

        b_s = tl.load(p_s, mask=mask_st, other=0.0).to(tl.float32)
        b_u = tl.load(p_u, mask=mask_st, other=0.0).to(tl.float32)
        
        p_W = W_ptr + i_b * stride_wb + i_h * stride_wh + (bos + i_c * BT) * stride_wt + offset_c[:, None] * stride_wt + offset_c[None, :] * stride_wc
        b_W = tl.load(p_W, mask=mask_t[:, None], other=0.0).to(tl.float32)

        b_V_d = tl.sum(b_s * d_curr[None, :], axis=1) 
        b_diff = 1.0 - b_V_d 
        b_W_diff = tl.sum(b_W * b_diff[None, :], axis=1) 
        
        d_jump = tl.sum(b_u * b_W_diff[:, None], axis=0) 
        d_curr = d_curr + d_jump
        
        if USE_PROJECTION:
            d_curr = tl.maximum(d_curr, d_min)
            d_curr = tl.minimum(d_curr, d_max)

        p_s += BT * stride_kt
        p_u += BT * stride_kt

    if USE_FINAL_STATE:
        p_final_d = final_d_ptr + i_n * H * K + i_h * K + offset_k
        tl.store(p_final_d, d_curr.to(p_final_d.dtype.element_ty), mask=mask_k)


@triton.jit
def osgm_phase1_wy_intra_chunk_kernel(
    k_ptr, d_out_ptr, d_starts_ptr, cu_seqlens_ptr,
    eta, d_min, d_max,
    stride_kb, stride_kh, stride_kt,
    stride_db, stride_dh, stride_dn,
    T: tl.constexpr, H: tl.constexpr, K: tl.constexpr, BK: tl.constexpr, BT: tl.constexpr,
    IS_VARLEN: tl.constexpr, USE_DENOMINATOR: tl.constexpr, USE_PROJECTION: tl.constexpr
):
    i_c, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    
    if IS_VARLEN:
        bos = tl.load(cu_seqlens_ptr + i_n).to(tl.int32)
        eos = tl.load(cu_seqlens_ptr + i_n + 1).to(tl.int32)
        i_b = 0; seq_len = eos - bos
    else:
        bos = 0; eos = T
        i_b = i_n; seq_len = T

    if i_c * BT >= seq_len: return

    offset_k = tl.arange(0, BK)
    mask_k = offset_k < K

    chunk_idx = (bos // BT) + i_c
    p_d_start = d_starts_ptr + i_b * stride_db + i_h * stride_dh + chunk_idx * stride_dn + offset_k
    d_curr = tl.load(p_d_start, mask=mask_k, other=0.0).to(tl.float32)

    for t in range(BT):
        t_idx = i_c * BT + t
        
        if t_idx < seq_len:
            p_k = k_ptr + i_b * stride_kb + i_h * stride_kh + (bos + t_idx) * stride_kt + offset_k
            p_d_out = d_out_ptr + i_b * stride_kb + i_h * stride_kh + (bos + t_idx) * stride_kt + offset_k
            
            b_k = tl.load(p_k, mask=mask_k, other=0.0).to(tl.float32)
            tl.store(p_d_out, d_curr.to(p_d_out.dtype.element_ty), mask=mask_k)

            b_s = b_k * b_k
            term_A = 1.0 - tl.sum(d_curr * b_s)
            
            if USE_DENOMINATOR:
                grad_d = term_A / (tl.sum(b_s) + 1e-5)
            else:
                grad_d = term_A

            d_next = d_curr + eta * grad_d * b_s

            if USE_PROJECTION:
                d_next = tl.maximum(d_next, d_min)
                d_next = tl.minimum(d_next, d_max)
            d_curr = d_next


@triton.jit
def osgm_phase1_bwd_pass1_local_kernel(
    k_ptr, d_out_ptr, grad_d_out_ptr, lambda_local_ptr, cu_seqlens_ptr,
    eta, d_min, d_max,
    stride_kb, stride_kh, stride_kt, stride_db, stride_dh, stride_dn,
    T: tl.constexpr, H: tl.constexpr, K: tl.constexpr, BK: tl.constexpr, BT: tl.constexpr,
    IS_VARLEN: tl.constexpr, USE_DENOMINATOR: tl.constexpr, USE_PROJECTION: tl.constexpr
):
    i_c, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    
    if IS_VARLEN:
        bos = tl.load(cu_seqlens_ptr + i_n).to(tl.int32)
        eos = tl.load(cu_seqlens_ptr + i_n + 1).to(tl.int32)
        i_b = 0; seq_len = eos - bos
    else:
        bos = 0; eos = T
        i_b = i_n; seq_len = T
        
    if i_c * BT >= seq_len: return

    offset_k = tl.arange(0, BK)
    mask_k = offset_k < K

    lambda_curr = tl.zeros([BK], dtype=tl.float32)

    for t in range(BT - 1, -1, -1):
        t_idx = i_c * BT + t
        if t_idx < seq_len:
            p_k = k_ptr + i_b * stride_kb + i_h * stride_kh + (bos + t_idx) * stride_kt + offset_k
            p_g_d = grad_d_out_ptr + i_b * stride_kb + i_h * stride_kh + (bos + t_idx) * stride_kt + offset_k
            p_d_prev = d_out_ptr + i_b * stride_kb + i_h * stride_kh + (bos + t_idx) * stride_kt + offset_k
            
            b_k = tl.load(p_k, mask=mask_k, other=0.0).to(tl.float32)
            b_g_d_out = tl.load(p_g_d, mask=mask_k, other=0.0).to(tl.float32)
            b_d_prev = tl.load(p_d_prev, mask=mask_k, other=0.0).to(tl.float32)
            
            b_s = b_k * b_k
            term_A = 1.0 - tl.sum(b_s * b_d_prev)
            if USE_DENOMINATOR:
                sum_s = tl.sum(b_s) + 1e-5
                grad_d = term_A / sum_s
                b_u = (eta / sum_s) * b_s
            else:
                grad_d = term_A
                b_u = eta * b_s
                
            if USE_PROJECTION:
                d_next_pre = b_d_prev + eta * grad_d * b_s
                mask_proj = (d_next_pre >= d_min) & (d_next_pre <= d_max)
                lambda_curr = tl.where(mask_proj, lambda_curr, 0.0)
            
            v_lambda = tl.sum(lambda_curr * b_s)
            lambda_prev = lambda_curr - b_u * v_lambda
            lambda_curr = lambda_prev + b_g_d_out

    chunk_idx = (bos // BT) + i_c
    p_lambda_local = lambda_local_ptr + i_b * stride_db + i_h * stride_dh + chunk_idx * stride_dn + offset_k
    tl.store(p_lambda_local, lambda_curr.to(p_lambda_local.dtype.element_ty), mask=mask_k)


@triton.jit
def osgm_phase1_bwd_pass2_scan_kernel(
    k_ptr, lambda_local_ptr, W_ptr, g_d_next_ptr,
    grad_final_d_ptr, grad_initial_d_ptr, cu_seqlens_ptr,
    eta, stride_kb, stride_kh, stride_kt,
    stride_wb, stride_wh, stride_wt, stride_wc,
    stride_db, stride_dh, stride_dn,
    T: tl.constexpr, H: tl.constexpr, K: tl.constexpr, BK: tl.constexpr, BT: tl.constexpr,
    IS_VARLEN: tl.constexpr, USE_DENOMINATOR: tl.constexpr,
    USE_FINAL_STATE_GRADIENT: tl.constexpr, STORE_INITIAL_STATE_GRADIENT: tl.constexpr
):
    i_nh = tl.program_id(0)
    i_n = i_nh // H
    i_h = i_nh % H

    if IS_VARLEN:
        bos = tl.load(cu_seqlens_ptr + i_n).to(tl.int32)
        eos = tl.load(cu_seqlens_ptr + i_n + 1).to(tl.int32)
        i_b = 0; seq_len = eos - bos
    else:
        bos = 0; eos = T
        i_b = i_n; seq_len = T
        
    seq_NT = tl.cdiv(seq_len, BT)
    offset_k = tl.arange(0, BK)
    offset_c = tl.arange(0, BT)
    mask_k = offset_k < K

    g_d_next = tl.zeros([BK], dtype=tl.float32)
    if USE_FINAL_STATE_GRADIENT:
        p_g_final = grad_final_d_ptr + i_n * H * K + i_h * K + offset_k
        g_d_next += tl.load(p_g_final, mask=mask_k, other=0.0).to(tl.float32)

    for i_c in range(seq_NT - 1, -1, -1):
        chunk_idx = (bos // BT) + i_c
        
        p_g_d_next = g_d_next_ptr + i_b * stride_db + i_h * stride_dh + chunk_idx * stride_dn + offset_k
        tl.store(p_g_d_next, g_d_next.to(p_g_d_next.dtype.element_ty), mask=mask_k)

        p_lambda = lambda_local_ptr + i_b * stride_db + i_h * stride_dh + chunk_idx * stride_dn + offset_k
        b_lambda_local = tl.load(p_lambda, mask=mask_k, other=0.0).to(tl.float32)

        p_W = W_ptr + i_b * stride_wb + i_h * stride_wh + (bos + i_c * BT) * stride_wt + offset_c[:, None] * stride_wt + offset_c[None, :] * stride_wc
        mask_t = (i_c * BT + offset_c) < seq_len
        b_W = tl.load(p_W, mask=mask_t[:, None], other=0.0).to(tl.float32)
        
        p_k = k_ptr + i_b * stride_kb + i_h * stride_kh + (bos + i_c * BT) * stride_kt + offset_c[:, None] * stride_kt + offset_k[None, :]
        mask_st = mask_t[:, None] & mask_k[None, :]
        b_k = tl.load(p_k, mask=mask_st, other=0.0).to(tl.float32)
        
        b_s = b_k * b_k
        if USE_DENOMINATOR:
            b_u = (eta / (tl.sum(b_s, axis=1) + 1e-5)[:, None]) * b_s
        else:
            b_u = eta * b_s

        U_lambda = tl.sum(b_u * g_d_next[None, :], axis=1) 
        WT_U_lambda = tl.sum(b_W * U_lambda[:, None], axis=0) 
        V_WT_U_lambda = tl.sum(b_s * WT_U_lambda[:, None], axis=0) 
        
        g_d_next = g_d_next - V_WT_U_lambda + b_lambda_local

    if STORE_INITIAL_STATE_GRADIENT:
        p_g_init = grad_initial_d_ptr + i_n * H * K + i_h * K + offset_k
        tl.store(p_g_init, g_d_next.to(p_g_init.dtype.element_ty), mask=mask_k)

@triton.jit
def osgm_phase1_bwd_pass3_final_kernel(
    k_ptr, d_out_ptr, grad_d_out_ptr, g_d_next_ptr, grad_k_ptr, cu_seqlens_ptr,
    eta, d_min, d_max,
    stride_kb, stride_kh, stride_kt, stride_db, stride_dh, stride_dn,
    T: tl.constexpr, H: tl.constexpr, K: tl.constexpr, BK: tl.constexpr, BT: tl.constexpr,
    IS_VARLEN: tl.constexpr, USE_DENOMINATOR: tl.constexpr, USE_PROJECTION: tl.constexpr
):
    i_c, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    
    if IS_VARLEN:
        bos = tl.load(cu_seqlens_ptr + i_n).to(tl.int32)
        eos = tl.load(cu_seqlens_ptr + i_n + 1).to(tl.int32)
        i_b = 0; seq_len = eos - bos
    else:
        bos = 0; eos = T
        i_b = i_n; seq_len = T
        
    if i_c * BT >= seq_len: return

    offset_k = tl.arange(0, BK)
    mask_k = offset_k < K

    chunk_idx = (bos // BT) + i_c

    p_g_d_next = g_d_next_ptr + i_b * stride_db + i_h * stride_dh + chunk_idx * stride_dn + offset_k
    lambda_curr = tl.load(p_g_d_next, mask=mask_k, other=0.0).to(tl.float32)

    for t in range(BT - 1, -1, -1):
        t_idx = i_c * BT + t
        if t_idx < seq_len:
            p_k = k_ptr + i_b * stride_kb + i_h * stride_kh + (bos + t_idx) * stride_kt + offset_k
            p_g_d = grad_d_out_ptr + i_b * stride_kb + i_h * stride_kh + (bos + t_idx) * stride_kt + offset_k
            p_g_k = grad_k_ptr + i_b * stride_kb + i_h * stride_kh + (bos + t_idx) * stride_kt + offset_k
            
            b_k = tl.load(p_k, mask=mask_k, other=0.0).to(tl.float32)
            b_g_d_out = tl.load(p_g_d, mask=mask_k, other=0.0).to(tl.float32)
            
            p_d_prev = d_out_ptr + i_b * stride_kb + i_h * stride_kh + (bos + t_idx) * stride_kt + offset_k
            b_d_prev = tl.load(p_d_prev, mask=mask_k, other=0.0).to(tl.float32)
            
            b_s = b_k * b_k
            term_A = 1.0 - tl.sum(b_s * b_d_prev)
            if USE_DENOMINATOR:
                sum_s = tl.sum(b_s) + 1e-5
                eta_t = eta / sum_s
                grad_d = term_A / sum_s
            else:
                eta_t = eta
                grad_d = term_A

            if USE_PROJECTION:
                d_next_pre = b_d_prev + eta * grad_d * b_s
                mask_proj = (d_next_pre >= d_min) & (d_next_pre <= d_max)
                lambda_curr = tl.where(mask_proj, lambda_curr, 0.0)

            b_u = eta_t * b_s
            c_t = term_A
            
            v_lambda = tl.sum(lambda_curr * b_s)
            g_c = eta_t * v_lambda
            g_s = eta_t * c_t * lambda_curr - g_c * b_d_prev
            
            if USE_DENOMINATOR:
                g_eta = c_t * v_lambda
                g_s_from_eta = - (g_eta / sum_s) * eta_t 
                g_s += g_s_from_eta
                
            g_k = 2.0 * b_k * g_s
            tl.store(p_g_k, g_k.to(p_g_k.dtype.element_ty), mask=mask_k)

            lambda_prev = lambda_curr - b_u * v_lambda
            lambda_curr = lambda_prev + b_g_d_out


def compute_osgm_phase1_fwd_wy(
    k: torch.Tensor, eta: float, use_denominator: bool, d_min: float, d_max: float, 
    cu_seqlens: torch.Tensor = None, initial_d: torch.Tensor = None, output_final_state: bool = False
):
    B, T, H, K = k.shape
    BK = triton.next_power_of_2(K)
    N = len(cu_seqlens) - 1 if cu_seqlens is not None else B

    s = k * k
    if use_denominator:
        s_sum = s.sum(dim=-1, keepdim=True) + 1e-5
        u = (eta / s_sum) * s
    else:
        u = eta * s
    beta_dummy = torch.ones_like(k[..., 0])

    with torch.no_grad():
        A = chunk_scaled_dot_qkw_fwd(q=s, k=u, beta=beta_dummy, cu_seqlens=cu_seqlens, chunk_size=64)
        W = solve_tril(A=A, cu_seqlens=cu_seqlens, output_dtype=k.dtype)

    BT = W.shape[-1]
    NT = triton.cdiv(T, BT)

    if W.dim() == 4:
        if W.shape[1] == T or (cu_seqlens is not None and W.shape[1] == k.shape[1]):
            W = W.transpose(1, 2)
        W_bh = W.contiguous()
    else:
        W_bh = W.contiguous()

    d_starts_bh = torch.empty(B, H, NT, K, device=k.device, dtype=k.dtype)
    d_out = torch.empty_like(k)
    final_d = torch.empty(N, H, K, device=k.device, dtype=k.dtype) if output_final_state else None

    s_bh = s.transpose(1, 2).contiguous()
    u_bh = u.transpose(1, 2).contiguous()
    k_bh = k.transpose(1, 2).contiguous()
    d_out_bh = d_out.transpose(1, 2).contiguous()

    if initial_d is not None:
        initial_d = initial_d.contiguous()
        stride_id_b, stride_id_h, stride_id_k = initial_d.stride(0), initial_d.stride(1), initial_d.stride(2)
    else:
        stride_id_b, stride_id_h, stride_id_k = 0, 0, 0

    osgm_phase1_wy_cross_chunk_scan_kernel[(N * H,)](
        s_bh, u_bh, W_bh, d_starts_bh, cu_seqlens,
        initial_d, final_d, 
        d_min or 0.0, d_max or 0.0,
        stride_id_b, stride_id_h, stride_id_k,
        s_bh.stride(0), s_bh.stride(1), s_bh.stride(2),
        W_bh.stride(0), W_bh.stride(1), W_bh.stride(2), W_bh.stride(3),
        d_starts_bh.stride(0), d_starts_bh.stride(1), d_starts_bh.stride(2),
        T, H, K, BK, BT,
        IS_VARLEN=(cu_seqlens is not None),
        USE_PROJECTION=(d_min is not None and d_max is not None),
        USE_INITIAL_STATE=(initial_d is not None),
        USE_FINAL_STATE=output_final_state
    )

    osgm_phase1_wy_intra_chunk_kernel[(NT, N * H)](
        k_bh, d_out_bh, d_starts_bh, cu_seqlens,
        eta, d_min or 0.0, d_max or 0.0,
        k_bh.stride(0), k_bh.stride(1), k_bh.stride(2),
        d_starts_bh.stride(0), d_starts_bh.stride(1), d_starts_bh.stride(2),
        T, H, K, BK, BT,
        IS_VARLEN=(cu_seqlens is not None),
        USE_DENOMINATOR=use_denominator,
        USE_PROJECTION=(d_min is not None and d_max is not None)
    )

    d_out = d_out_bh.transpose(1, 2).contiguous()
    return d_out, final_d, W_bh, d_starts_bh


def compute_osgm_phase1_bwd_wy(
    k, d_out, dd, eta, use_denominator, d_min, d_max, cu_seqlens, 
    dd_final, has_initial_d, 
    W_bh, d_starts_bh
):
    B, T, H, K = k.shape
    BK = triton.next_power_of_2(K)
    N = len(cu_seqlens) - 1 if cu_seqlens is not None else B

    BT = W_bh.shape[-1]
    NT = d_starts_bh.shape[2]

    k_bh = k.transpose(1, 2).contiguous()
    d_out_bh = d_out.transpose(1, 2).contiguous()
    dd_bh = dd.transpose(1, 2).contiguous()

    lambda_local_bh = torch.empty(B, H, NT, K, device=k.device, dtype=k.dtype)
    g_d_next_bh = torch.empty(B, H, NT, K, device=k.device, dtype=k.dtype)
    
    osgm_phase1_bwd_pass1_local_kernel[(NT, N * H)](
        k_bh, d_out_bh, dd_bh, lambda_local_bh, cu_seqlens, 
        eta, d_min or 0.0, d_max or 0.0,
        k_bh.stride(0), k_bh.stride(1), k_bh.stride(2),
        d_starts_bh.stride(0), d_starts_bh.stride(1), d_starts_bh.stride(2),
        T, H, K, BK, BT,
        IS_VARLEN=(cu_seqlens is not None), USE_DENOMINATOR=use_denominator,
        USE_PROJECTION=(d_min is not None and d_max is not None)
    )

    grad_initial_d = torch.empty(N, H, K, device=k.device, dtype=k.dtype) if has_initial_d else None

    osgm_phase1_bwd_pass2_scan_kernel[(N * H,)](
        k_bh, lambda_local_bh, W_bh, g_d_next_bh,
        dd_final, grad_initial_d, cu_seqlens,
        eta, k_bh.stride(0), k_bh.stride(1), k_bh.stride(2),
        W_bh.stride(0), W_bh.stride(1), W_bh.stride(2), W_bh.stride(3),
        g_d_next_bh.stride(0), g_d_next_bh.stride(1), g_d_next_bh.stride(2),
        T, H, K, BK, BT,
        IS_VARLEN=(cu_seqlens is not None), USE_DENOMINATOR=use_denominator,
        USE_FINAL_STATE_GRADIENT=(dd_final is not None),
        STORE_INITIAL_STATE_GRADIENT=has_initial_d
    )

    grad_k_bh = torch.empty_like(k_bh)

    osgm_phase1_bwd_pass3_final_kernel[(NT, N * H)](
        k_bh, d_out_bh, dd_bh, g_d_next_bh, grad_k_bh, cu_seqlens, 
        eta, d_min or 0.0, d_max or 0.0,
        k_bh.stride(0), k_bh.stride(1), k_bh.stride(2),
        g_d_next_bh.stride(0), g_d_next_bh.stride(1), g_d_next_bh.stride(2),
        T, H, K, BK, BT,
        IS_VARLEN=(cu_seqlens is not None), USE_DENOMINATOR=use_denominator,
        USE_PROJECTION=(d_min is not None and d_max is not None)
    )

    grad_k = grad_k_bh.transpose(1, 2).contiguous()
    return grad_k, grad_initial_d