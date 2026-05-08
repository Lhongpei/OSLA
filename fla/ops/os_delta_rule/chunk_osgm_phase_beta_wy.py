# -*- coding: utf-8 -*-
"""Chunk/WY OSGM phase1 for beta-aware no-decay post-gate regret.

This implements the exact recurrence

    d_{t+1} = d_t + eta * beta_t * (1 - beta_t * <d_t, k_t^2>) * k_t^2

as

    d_{t+1} = d_t + u_t * (1 - <s_t, d_t>)
    s_t = beta_t * k_t^2
    u_t = eta * beta_t * k_t^2

The chunk-level algebra is the same as the existing plain OSGM WY path, but
the backward must expose gradients for both k and beta.
"""

import torch
import triton
import triton.language as tl

from fla.ops.os_delta_rule.chunk_osgm_phase_wy import osgm_phase1_wy_cross_chunk_scan_kernel
from fla.ops.os_delta_rule.chunk_scaled_dot_qkw import chunk_scaled_dot_qkw_fwd
from fla.ops.utils.solve_tril import solve_tril


@triton.jit
def osgm_beta_wy_intra_chunk_kernel(
    s_ptr, u_ptr, d_out_ptr, d_starts_ptr, cu_seqlens_ptr,
    d_min, d_max,
    stride_s_b, stride_s_h, stride_s_t,
    stride_d_b, stride_d_h, stride_d_t,
    stride_db, stride_dh, stride_dn,
    T: tl.constexpr, H: tl.constexpr, K: tl.constexpr, BK: tl.constexpr, BT: tl.constexpr,
    IS_VARLEN: tl.constexpr, USE_PROJECTION: tl.constexpr,
):
    i_c, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H

    if IS_VARLEN:
        bos = tl.load(cu_seqlens_ptr + i_n).to(tl.int32)
        eos = tl.load(cu_seqlens_ptr + i_n + 1).to(tl.int32)
        i_b = 0
        seq_len = eos - bos
    else:
        bos = 0
        i_b = i_n
        seq_len = T

    if i_c * BT >= seq_len:
        return

    offs = tl.arange(0, BK)
    mask_k = offs < K
    chunk_idx = (bos // BT) + i_c
    p_d_start = d_starts_ptr + i_b * stride_db + i_h * stride_dh + chunk_idx * stride_dn + offs
    d_curr = tl.load(p_d_start, mask=mask_k, other=0.0).to(tl.float32)

    for t in range(BT):
        t_idx = i_c * BT + t
        if t_idx < seq_len:
            p_s = s_ptr + i_b * stride_s_b + i_h * stride_s_h + (bos + t_idx) * stride_s_t + offs
            p_u = u_ptr + i_b * stride_s_b + i_h * stride_s_h + (bos + t_idx) * stride_s_t + offs
            p_d = d_out_ptr + i_b * stride_d_b + i_h * stride_d_h + (bos + t_idx) * stride_d_t + offs

            b_s = tl.load(p_s, mask=mask_k, other=0.0).to(tl.float32)
            b_u = tl.load(p_u, mask=mask_k, other=0.0).to(tl.float32)
            tl.store(p_d, d_curr.to(p_d.dtype.element_ty), mask=mask_k)

            c = 1.0 - tl.sum(b_s * d_curr)
            d_next = d_curr + b_u * c
            if USE_PROJECTION:
                d_next = tl.maximum(d_next, d_min)
                d_next = tl.minimum(d_next, d_max)
            d_curr = d_next


@triton.jit
def osgm_beta_bwd_pass1_local_kernel(
    s_ptr, u_ptr, d_out_ptr, grad_d_out_ptr, lambda_local_ptr, cu_seqlens_ptr,
    d_min, d_max,
    stride_s_b, stride_s_h, stride_s_t,
    stride_d_b, stride_d_h, stride_d_t,
    stride_db, stride_dh, stride_dn,
    T: tl.constexpr, H: tl.constexpr, K: tl.constexpr, BK: tl.constexpr, BT: tl.constexpr,
    IS_VARLEN: tl.constexpr, USE_PROJECTION: tl.constexpr,
):
    i_c, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H

    if IS_VARLEN:
        bos = tl.load(cu_seqlens_ptr + i_n).to(tl.int32)
        eos = tl.load(cu_seqlens_ptr + i_n + 1).to(tl.int32)
        i_b = 0
        seq_len = eos - bos
    else:
        bos = 0
        i_b = i_n
        seq_len = T

    if i_c * BT >= seq_len:
        return

    offs = tl.arange(0, BK)
    mask_k = offs < K
    lambda_curr = tl.zeros([BK], dtype=tl.float32)

    for t in range(BT - 1, -1, -1):
        t_idx = i_c * BT + t
        if t_idx < seq_len:
            p_s = s_ptr + i_b * stride_s_b + i_h * stride_s_h + (bos + t_idx) * stride_s_t + offs
            p_u = u_ptr + i_b * stride_s_b + i_h * stride_s_h + (bos + t_idx) * stride_s_t + offs
            p_d = d_out_ptr + i_b * stride_d_b + i_h * stride_d_h + (bos + t_idx) * stride_d_t + offs
            p_gd = grad_d_out_ptr + i_b * stride_d_b + i_h * stride_d_h + (bos + t_idx) * stride_d_t + offs

            b_s = tl.load(p_s, mask=mask_k, other=0.0).to(tl.float32)
            b_u = tl.load(p_u, mask=mask_k, other=0.0).to(tl.float32)
            b_d = tl.load(p_d, mask=mask_k, other=0.0).to(tl.float32)
            b_gd = tl.load(p_gd, mask=mask_k, other=0.0).to(tl.float32)

            if USE_PROJECTION:
                c = 1.0 - tl.sum(b_s * b_d)
                d_next_pre = b_d + b_u * c
                mask_proj = (d_next_pre >= d_min) & (d_next_pre <= d_max)
                lambda_curr = tl.where(mask_proj, lambda_curr, 0.0)

            u_lambda = tl.sum(lambda_curr * b_u)
            lambda_curr = lambda_curr - b_s * u_lambda + b_gd

    chunk_idx = (bos // BT) + i_c
    p_lambda = lambda_local_ptr + i_b * stride_db + i_h * stride_dh + chunk_idx * stride_dn + offs
    tl.store(p_lambda, lambda_curr.to(p_lambda.dtype.element_ty), mask=mask_k)


@triton.jit
def osgm_beta_bwd_pass2_scan_kernel(
    s_ptr, u_ptr, lambda_local_ptr, W_ptr, g_d_next_ptr,
    grad_final_d_ptr, grad_initial_d_ptr, cu_seqlens_ptr,
    stride_s_b, stride_s_h, stride_s_t,
    stride_wb, stride_wh, stride_wt, stride_wc,
    stride_db, stride_dh, stride_dn,
    T: tl.constexpr, H: tl.constexpr, K: tl.constexpr, BK: tl.constexpr, BT: tl.constexpr,
    IS_VARLEN: tl.constexpr, USE_FINAL_STATE_GRADIENT: tl.constexpr,
    STORE_INITIAL_STATE_GRADIENT: tl.constexpr,
):
    i_nh = tl.program_id(0)
    i_n, i_h = i_nh // H, i_nh % H

    if IS_VARLEN:
        bos = tl.load(cu_seqlens_ptr + i_n).to(tl.int32)
        eos = tl.load(cu_seqlens_ptr + i_n + 1).to(tl.int32)
        i_b = 0
        seq_len = eos - bos
    else:
        bos = 0
        i_b = i_n
        seq_len = T

    seq_nt = tl.cdiv(seq_len, BT)
    offs = tl.arange(0, BK)
    offs_t = tl.arange(0, BT)
    mask_k = offs < K

    g_next = tl.zeros([BK], dtype=tl.float32)
    if USE_FINAL_STATE_GRADIENT:
        p_gf = grad_final_d_ptr + i_n * H * K + i_h * K + offs
        g_next += tl.load(p_gf, mask=mask_k, other=0.0).to(tl.float32)

    for i_c in range(seq_nt - 1, -1, -1):
        chunk_idx = (bos // BT) + i_c
        p_g_next = g_d_next_ptr + i_b * stride_db + i_h * stride_dh + chunk_idx * stride_dn + offs
        tl.store(p_g_next, g_next.to(p_g_next.dtype.element_ty), mask=mask_k)

        p_lambda = lambda_local_ptr + i_b * stride_db + i_h * stride_dh + chunk_idx * stride_dn + offs
        b_lambda = tl.load(p_lambda, mask=mask_k, other=0.0).to(tl.float32)

        mask_t = (i_c * BT + offs_t) < seq_len
        p_W = W_ptr + i_b * stride_wb + i_h * stride_wh + (bos + i_c * BT) * stride_wt + offs_t[:, None] * stride_wt + offs_t[None, :] * stride_wc
        b_W = tl.load(p_W, mask=mask_t[:, None], other=0.0).to(tl.float32)

        p_s = s_ptr + i_b * stride_s_b + i_h * stride_s_h + (bos + i_c * BT) * stride_s_t + offs_t[:, None] * stride_s_t + offs[None, :]
        p_u = u_ptr + i_b * stride_s_b + i_h * stride_s_h + (bos + i_c * BT) * stride_s_t + offs_t[:, None] * stride_s_t + offs[None, :]
        mask_st = mask_t[:, None] & mask_k[None, :]
        b_s = tl.load(p_s, mask=mask_st, other=0.0).to(tl.float32)
        b_u = tl.load(p_u, mask=mask_st, other=0.0).to(tl.float32)

        U_lambda = tl.sum(b_u * g_next[None, :], axis=1)
        WT_U_lambda = tl.sum(b_W * U_lambda[:, None], axis=0)
        V_WT_U_lambda = tl.sum(b_s * WT_U_lambda[:, None], axis=0)
        g_next = g_next - V_WT_U_lambda + b_lambda

    if STORE_INITIAL_STATE_GRADIENT:
        p_gi = grad_initial_d_ptr + i_n * H * K + i_h * K + offs
        tl.store(p_gi, g_next.to(p_gi.dtype.element_ty), mask=mask_k)


@triton.jit
def osgm_beta_bwd_pass3_final_kernel(
    k_ptr, beta_ptr, s_ptr, u_ptr, d_out_ptr, grad_d_out_ptr, g_d_next_ptr,
    grad_k_ptr, grad_beta_ptr, cu_seqlens_ptr,
    eta, d_min, d_max,
    stride_kb, stride_kt, stride_kh,
    stride_bb, stride_bt, stride_bh,
    stride_s_b, stride_s_h, stride_s_t,
    stride_d_b, stride_d_h, stride_d_t,
    stride_db, stride_dh, stride_dn,
    T: tl.constexpr, H: tl.constexpr, K: tl.constexpr, BK: tl.constexpr, BT: tl.constexpr,
    IS_VARLEN: tl.constexpr, USE_PROJECTION: tl.constexpr,
):
    i_c, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H

    if IS_VARLEN:
        bos = tl.load(cu_seqlens_ptr + i_n).to(tl.int32)
        eos = tl.load(cu_seqlens_ptr + i_n + 1).to(tl.int32)
        i_b = 0
        seq_len = eos - bos
    else:
        bos = 0
        i_b = i_n
        seq_len = T

    if i_c * BT >= seq_len:
        return

    offs = tl.arange(0, BK)
    mask_k = offs < K
    chunk_idx = (bos // BT) + i_c
    p_g_next = g_d_next_ptr + i_b * stride_db + i_h * stride_dh + chunk_idx * stride_dn + offs
    lambda_curr = tl.load(p_g_next, mask=mask_k, other=0.0).to(tl.float32)

    for t in range(BT - 1, -1, -1):
        t_idx = i_c * BT + t
        if t_idx < seq_len:
            p_s = s_ptr + i_b * stride_s_b + i_h * stride_s_h + (bos + t_idx) * stride_s_t + offs
            p_u = u_ptr + i_b * stride_s_b + i_h * stride_s_h + (bos + t_idx) * stride_s_t + offs
            p_d = d_out_ptr + i_b * stride_d_b + i_h * stride_d_h + (bos + t_idx) * stride_d_t + offs
            p_gd = grad_d_out_ptr + i_b * stride_d_b + i_h * stride_d_h + (bos + t_idx) * stride_d_t + offs
            p_k = k_ptr + i_b * stride_kb + (bos + t_idx) * stride_kt + i_h * stride_kh + offs
            p_b = beta_ptr + i_b * stride_bb + (bos + t_idx) * stride_bt + i_h * stride_bh
            p_gk = grad_k_ptr + i_b * stride_kb + (bos + t_idx) * stride_kt + i_h * stride_kh + offs
            p_gb = grad_beta_ptr + i_b * stride_bb + (bos + t_idx) * stride_bt + i_h * stride_bh

            b_s = tl.load(p_s, mask=mask_k, other=0.0).to(tl.float32)
            b_u = tl.load(p_u, mask=mask_k, other=0.0).to(tl.float32)
            b_d = tl.load(p_d, mask=mask_k, other=0.0).to(tl.float32)
            b_gd = tl.load(p_gd, mask=mask_k, other=0.0).to(tl.float32)
            b_k = tl.load(p_k, mask=mask_k, other=0.0).to(tl.float32)
            b_beta = tl.load(p_b).to(tl.float32)

            c = 1.0 - tl.sum(b_s * b_d)
            if USE_PROJECTION:
                d_next_pre = b_d + b_u * c
                mask_proj = (d_next_pre >= d_min) & (d_next_pre <= d_max)
                lambda_curr = tl.where(mask_proj, lambda_curr, 0.0)

            u_lambda = tl.sum(lambda_curr * b_u)
            grad_u = lambda_curr * c
            grad_s = -u_lambda * b_d

            k_sq = b_k * b_k
            common = grad_s + eta * grad_u
            grad_k_sq = b_beta * common
            grad_beta = tl.sum(k_sq * common)
            grad_k = 2.0 * b_k * grad_k_sq

            tl.store(p_gk, grad_k.to(p_gk.dtype.element_ty), mask=mask_k)
            tl.store(p_gb, grad_beta.to(p_gb.dtype.element_ty))

            lambda_curr = lambda_curr - b_s * u_lambda + b_gd


def _make_s_u(k: torch.Tensor, beta: torch.Tensor, eta: float):
    k_sq = k.float() * k.float()
    beta_e = beta.float().unsqueeze(-1)
    s = beta_e * k_sq
    u = eta * s
    return s.to(k.dtype), u.to(k.dtype)


def compute_osgm_beta_phase1_fwd_wy(
    k: torch.Tensor,
    beta: torch.Tensor,
    eta: float,
    use_denominator: bool,
    d_min: float,
    d_max: float,
    cu_seqlens: torch.Tensor = None,
    initial_d: torch.Tensor = None,
    output_final_state: bool = False,
):
    if use_denominator:
        raise NotImplementedError("beta WY phase1 currently supports use_denominator=False only.")

    B, T, H, K = k.shape
    BK = triton.next_power_of_2(K)
    N = len(cu_seqlens) - 1 if cu_seqlens is not None else B
    s, u = _make_s_u(k, beta, eta)
    beta_dummy = torch.ones_like(beta)

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

    s_bh = s.transpose(1, 2).contiguous()
    u_bh = u.transpose(1, 2).contiguous()
    d_out_bh = torch.empty(B, H, T, K, device=k.device, dtype=k.dtype)
    d_starts_bh = torch.empty(B, H, NT, K, device=k.device, dtype=k.dtype)
    final_d = torch.empty(N, H, K, device=k.device, dtype=k.dtype) if output_final_state else None

    if initial_d is not None:
        initial_d = initial_d.contiguous()
        stride_id_b, stride_id_h, stride_id_k = initial_d.stride()
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
        USE_FINAL_STATE=output_final_state,
    )

    osgm_beta_wy_intra_chunk_kernel[(NT, N * H)](
        s_bh, u_bh, d_out_bh, d_starts_bh, cu_seqlens,
        d_min or 0.0, d_max or 0.0,
        s_bh.stride(0), s_bh.stride(1), s_bh.stride(2),
        d_out_bh.stride(0), d_out_bh.stride(1), d_out_bh.stride(2),
        d_starts_bh.stride(0), d_starts_bh.stride(1), d_starts_bh.stride(2),
        T, H, K, BK, BT,
        IS_VARLEN=(cu_seqlens is not None),
        USE_PROJECTION=(d_min is not None and d_max is not None),
    )

    d_out = d_out_bh.transpose(1, 2).contiguous()
    return d_out, final_d, W_bh, d_starts_bh


def compute_osgm_beta_phase1_bwd_wy(
    k: torch.Tensor,
    beta: torch.Tensor,
    d_out: torch.Tensor,
    dd_in: torch.Tensor,
    eta: float,
    use_denominator: bool,
    d_min: float,
    d_max: float,
    cu_seqlens: torch.Tensor = None,
    dd_final: torch.Tensor = None,
    output_initial_state_gradient: bool = False,
    W_bh: torch.Tensor = None,
    d_starts_bh: torch.Tensor = None,
):
    if use_denominator:
        raise NotImplementedError("beta WY phase1 currently supports use_denominator=False only.")
    if W_bh is None or d_starts_bh is None:
        raise ValueError("W_bh and d_starts_bh from forward are required.")

    B, T, H, K = k.shape
    BK = triton.next_power_of_2(K)
    N = len(cu_seqlens) - 1 if cu_seqlens is not None else B
    BT = W_bh.shape[-1]
    NT = d_starts_bh.shape[2]

    s, u = _make_s_u(k, beta, eta)
    s_bh = s.transpose(1, 2).contiguous()
    u_bh = u.transpose(1, 2).contiguous()
    d_bh = d_out.transpose(1, 2).contiguous()
    dd_bh = dd_in.transpose(1, 2).contiguous()

    lambda_local_bh = torch.empty(B, H, NT, K, device=k.device, dtype=k.dtype)
    g_d_next_bh = torch.empty(B, H, NT, K, device=k.device, dtype=k.dtype)

    osgm_beta_bwd_pass1_local_kernel[(NT, N * H)](
        s_bh, u_bh, d_bh, dd_bh, lambda_local_bh, cu_seqlens,
        d_min or 0.0, d_max or 0.0,
        s_bh.stride(0), s_bh.stride(1), s_bh.stride(2),
        d_bh.stride(0), d_bh.stride(1), d_bh.stride(2),
        lambda_local_bh.stride(0), lambda_local_bh.stride(1), lambda_local_bh.stride(2),
        T, H, K, BK, BT,
        IS_VARLEN=(cu_seqlens is not None),
        USE_PROJECTION=(d_min is not None and d_max is not None),
    )

    dd_initial = torch.empty(N, H, K, device=k.device, dtype=k.dtype) if output_initial_state_gradient else None

    osgm_beta_bwd_pass2_scan_kernel[(N * H,)](
        s_bh, u_bh, lambda_local_bh, W_bh, g_d_next_bh,
        dd_final, dd_initial, cu_seqlens,
        s_bh.stride(0), s_bh.stride(1), s_bh.stride(2),
        W_bh.stride(0), W_bh.stride(1), W_bh.stride(2), W_bh.stride(3),
        g_d_next_bh.stride(0), g_d_next_bh.stride(1), g_d_next_bh.stride(2),
        T, H, K, BK, BT,
        IS_VARLEN=(cu_seqlens is not None),
        USE_FINAL_STATE_GRADIENT=(dd_final is not None),
        STORE_INITIAL_STATE_GRADIENT=output_initial_state_gradient,
    )

    dk = torch.empty_like(k)
    dbeta = torch.empty_like(beta)
    osgm_beta_bwd_pass3_final_kernel[(NT, N * H)](
        k, beta, s_bh, u_bh, d_bh, dd_bh, g_d_next_bh,
        dk, dbeta, cu_seqlens,
        eta, d_min or 0.0, d_max or 0.0,
        k.stride(0), k.stride(1), k.stride(2),
        beta.stride(0), beta.stride(1), beta.stride(2),
        s_bh.stride(0), s_bh.stride(1), s_bh.stride(2),
        d_bh.stride(0), d_bh.stride(1), d_bh.stride(2),
        g_d_next_bh.stride(0), g_d_next_bh.stride(1), g_d_next_bh.stride(2),
        T, H, K, BK, BT,
        IS_VARLEN=(cu_seqlens is not None),
        USE_PROJECTION=(d_min is not None and d_max is not None),
    )
    return dk, dd_initial, dbeta
