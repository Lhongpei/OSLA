# -*- coding: utf-8 -*-
"""Chunk/WY phase1 for beta-aware OSGM with data-dependent decay.

This is the parallel companion to ``chunk_osgm_phase_dd_decay_beta.py``.
It preserves the recurrence

    d_t is stored for the state update
    d_{t+1} = exp(g_t) * d_t
              + eta * beta_t * (1 - beta_t * <d_t, k_t^2>) * k_t^2

but computes chunk starts with a stable WY representation instead of one
program scanning all T tokens.  Within each chunk we still run the exact
64-token recurrence; the scan across chunks is only over T / 64 states.
"""

import torch
import triton
import triton.language as tl

from fla.ops.os_delta_rule.chunk_scaled_dot_qkw import chunk_scaled_dot_qkw_fwd
from fla.ops.utils.solve_tril import solve_tril


@triton.jit
def osgm_dd_decay_beta_wy_cross_chunk_scan_kernel(
    s_ptr, u_ptr, pre_ptr, W_ptr, d_starts_ptr, chunk_decay_ptr,
    initial_d_ptr, final_d_ptr,
    d_min, d_max,
    stride_s_b, stride_s_h, stride_s_t,
    stride_p_b, stride_p_h, stride_p_t,
    stride_wb, stride_wh, stride_wt, stride_wc,
    stride_db, stride_dh, stride_dn,
    stride_cb, stride_ch, stride_cn,
    T: tl.constexpr, H: tl.constexpr, K: tl.constexpr, BK: tl.constexpr, BT: tl.constexpr,
    USE_PROJECTION: tl.constexpr, USE_INITIAL_STATE: tl.constexpr, USE_FINAL_STATE: tl.constexpr,
):
    i_bh = tl.program_id(0)
    i_b, i_h = i_bh // H, i_bh % H

    offs_k = tl.arange(0, BK)
    offs_t = tl.arange(0, BT)
    mask_k = offs_k < K
    NT = tl.cdiv(T, BT)

    if USE_INITIAL_STATE:
        p_initial = initial_d_ptr + i_b * H * K + i_h * K + offs_k
        d_curr = tl.load(p_initial, mask=mask_k, other=0.0).to(tl.float32)
    else:
        d_curr = tl.zeros([BK], dtype=tl.float32)

    for i_c in range(NT):
        p_d_start = d_starts_ptr + i_b * stride_db + i_h * stride_dh + i_c * stride_dn + offs_k
        tl.store(p_d_start, d_curr.to(p_d_start.dtype.element_ty), mask=mask_k)

        p_s = s_ptr + i_b * stride_s_b + i_h * stride_s_h + i_c * BT * stride_s_t + offs_t[:, None] * stride_s_t + offs_k[None, :]
        p_u = u_ptr + i_b * stride_s_b + i_h * stride_s_h + i_c * BT * stride_s_t + offs_t[:, None] * stride_s_t + offs_k[None, :]
        p_pre = pre_ptr + i_b * stride_p_b + i_h * stride_p_h + i_c * BT * stride_p_t + offs_t * stride_p_t
        p_W = W_ptr + i_b * stride_wb + i_h * stride_wh + i_c * BT * stride_wt + offs_t[:, None] * stride_wt + offs_t[None, :] * stride_wc
        p_decay = chunk_decay_ptr + i_b * stride_cb + i_h * stride_ch + i_c * stride_cn

        b_s = tl.load(p_s, mask=mask_k[None, :], other=0.0).to(tl.float32)
        b_u = tl.load(p_u, mask=mask_k[None, :], other=0.0).to(tl.float32)
        b_pre = tl.load(p_pre).to(tl.float32)
        b_W = tl.load(p_W).to(tl.float32)
        b_decay = tl.load(p_decay).to(tl.float32)
        b_end = tl.log(b_decay)

        b_a = tl.exp(b_pre) * tl.sum(b_s * d_curr[None, :], axis=1)
        b_z = tl.sum(b_W * (1.0 - b_a)[None, :], axis=1)
        b_weight = tl.exp(b_end - b_pre)
        d_jump = tl.sum(b_u * (b_z * b_weight)[:, None], axis=0)
        d_next = b_decay * d_curr + d_jump

        if USE_PROJECTION:
            d_next = tl.maximum(d_next, d_min)
            d_next = tl.minimum(d_next, d_max)
        d_curr = d_next

    if USE_FINAL_STATE:
        p_final = final_d_ptr + i_b * H * K + i_h * K + offs_k
        tl.store(p_final, d_curr.to(p_final.dtype.element_ty), mask=mask_k)


@triton.jit
def osgm_dd_decay_beta_wy_intra_chunk_kernel(
    k_ptr, g_ptr, beta_ptr, d_out_ptr, d_starts_ptr,
    eta, d_min, d_max,
    stride_kb, stride_kt, stride_kh,
    stride_gb, stride_gt, stride_gh,
    stride_bb, stride_bt, stride_bh,
    stride_db, stride_dh, stride_dn,
    T: tl.constexpr, H: tl.constexpr, K: tl.constexpr, BK: tl.constexpr, BT: tl.constexpr,
    USE_DENOMINATOR: tl.constexpr, USE_PROJECTION: tl.constexpr,
):
    i_c, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H

    offs = tl.arange(0, BK)
    mask = offs < K
    p_d_start = d_starts_ptr + i_b * stride_db + i_h * stride_dh + i_c * stride_dn + offs
    d_curr = tl.load(p_d_start, mask=mask, other=0.0).to(tl.float32)

    p_k = k_ptr + i_b * stride_kb + i_c * BT * stride_kt + i_h * stride_kh + offs
    p_d = d_out_ptr + i_b * stride_kb + i_c * BT * stride_kt + i_h * stride_kh + offs
    p_g = g_ptr + i_b * stride_gb + i_c * BT * stride_gt + i_h * stride_gh
    p_beta = beta_ptr + i_b * stride_bb + i_c * BT * stride_bt + i_h * stride_bh

    for _ in range(BT):
        b_k = tl.load(p_k, mask=mask, other=0.0).to(tl.float32)
        b_g = tl.load(p_g).to(tl.float32)
        b_beta = tl.load(p_beta).to(tl.float32)

        tl.store(p_d, d_curr.to(p_d.dtype.element_ty), mask=mask)

        k_sq = b_k * b_k
        inner = tl.sum(d_curr * k_sq)
        term_a = 1.0 - b_beta * inner
        if USE_DENOMINATOR:
            grad_d = b_beta * term_a / (tl.sum(k_sq) + 1e-5)
        else:
            grad_d = b_beta * term_a

        d_next = tl.exp(b_g) * d_curr + eta * grad_d * k_sq
        if USE_PROJECTION:
            d_next = tl.maximum(d_next, d_min)
            d_next = tl.minimum(d_next, d_max)
        d_curr = d_next

        p_k += stride_kt
        p_d += stride_kt
        p_g += stride_gt
        p_beta += stride_bt


@triton.jit
def osgm_dd_decay_beta_wy_bwd_pass1_local_kernel(
    k_ptr, d_out_ptr, g_ptr, beta_ptr, dd_in_ptr, lambda_local_ptr,
    eta, d_min, d_max,
    stride_kb, stride_kt, stride_kh,
    stride_gb, stride_gt, stride_gh,
    stride_bb, stride_bt, stride_bh,
    stride_db, stride_dh, stride_dn,
    T: tl.constexpr, H: tl.constexpr, K: tl.constexpr, BK: tl.constexpr, BT: tl.constexpr,
    USE_DENOMINATOR: tl.constexpr, USE_PROJECTION: tl.constexpr,
):
    i_c, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H

    offs = tl.arange(0, BK)
    mask = offs < K
    p_k = k_ptr + i_b * stride_kb + (i_c * BT + BT - 1) * stride_kt + i_h * stride_kh + offs
    p_d = d_out_ptr + i_b * stride_kb + (i_c * BT + BT - 1) * stride_kt + i_h * stride_kh + offs
    p_dd = dd_in_ptr + i_b * stride_kb + (i_c * BT + BT - 1) * stride_kt + i_h * stride_kh + offs
    p_g = g_ptr + i_b * stride_gb + (i_c * BT + BT - 1) * stride_gt + i_h * stride_gh
    p_beta = beta_ptr + i_b * stride_bb + (i_c * BT + BT - 1) * stride_bt + i_h * stride_bh

    lam = tl.zeros([BK], dtype=tl.float32)
    for _ in range(BT):
        b_k = tl.load(p_k, mask=mask, other=0.0).to(tl.float32)
        b_d = tl.load(p_d, mask=mask, other=0.0).to(tl.float32)
        b_dd = tl.load(p_dd, mask=mask, other=0.0).to(tl.float32)
        b_g = tl.load(p_g).to(tl.float32)
        b_beta = tl.load(p_beta).to(tl.float32)
        gamma = tl.exp(b_g)

        k_sq = b_k * b_k
        inner = tl.sum(b_d * k_sq)
        term_a = 1.0 - b_beta * inner
        if USE_PROJECTION:
            if USE_DENOMINATOR:
                grad_d_pre = b_beta * term_a / (tl.sum(k_sq) + 1e-5)
            else:
                grad_d_pre = b_beta * term_a
            d_next_pre = gamma * b_d + eta * grad_d_pre * k_sq
            mask_proj = (d_next_pre >= d_min) & (d_next_pre <= d_max)
            lam = tl.where(mask_proj, lam, 0.0)

        term_b = tl.sum(lam * k_sq)
        if USE_DENOMINATOR:
            sum_k_sq = tl.sum(k_sq) + 1e-5
            lam = gamma * lam - eta * (b_beta * b_beta) * (term_b / sum_k_sq) * k_sq
        else:
            lam = gamma * lam - eta * (b_beta * b_beta) * term_b * k_sq
        lam += b_dd

        p_k -= stride_kt
        p_d -= stride_kt
        p_dd -= stride_kt
        p_g -= stride_gt
        p_beta -= stride_bt

    p_lambda = lambda_local_ptr + i_b * stride_db + i_h * stride_dh + i_c * stride_dn + offs
    tl.store(p_lambda, lam.to(p_lambda.dtype.element_ty), mask=mask)


@triton.jit
def osgm_dd_decay_beta_wy_bwd_pass2_scan_kernel(
    s_ptr, u_ptr, pre_ptr, W_ptr, lambda_local_ptr, chunk_decay_ptr, g_d_next_ptr,
    grad_final_d_ptr, grad_initial_d_ptr,
    stride_s_b, stride_s_h, stride_s_t,
    stride_p_b, stride_p_h, stride_p_t,
    stride_wb, stride_wh, stride_wt, stride_wc,
    stride_db, stride_dh, stride_dn,
    stride_cb, stride_ch, stride_cn,
    T: tl.constexpr, H: tl.constexpr, K: tl.constexpr, BK: tl.constexpr, BT: tl.constexpr,
    USE_FINAL_STATE_GRADIENT: tl.constexpr, STORE_INITIAL_STATE_GRADIENT: tl.constexpr,
):
    i_bh = tl.program_id(0)
    i_b, i_h = i_bh // H, i_bh % H

    offs_k = tl.arange(0, BK)
    offs_t = tl.arange(0, BT)
    mask_k = offs_k < K
    NT = tl.cdiv(T, BT)

    lam_next = tl.zeros([BK], dtype=tl.float32)
    if USE_FINAL_STATE_GRADIENT:
        p_gf = grad_final_d_ptr + i_b * H * K + i_h * K + offs_k
        lam_next += tl.load(p_gf, mask=mask_k, other=0.0).to(tl.float32)

    for i_c in range(NT - 1, -1, -1):
        p_g_next = g_d_next_ptr + i_b * stride_db + i_h * stride_dh + i_c * stride_dn + offs_k
        tl.store(p_g_next, lam_next.to(p_g_next.dtype.element_ty), mask=mask_k)

        p_s = s_ptr + i_b * stride_s_b + i_h * stride_s_h + i_c * BT * stride_s_t + offs_t[:, None] * stride_s_t + offs_k[None, :]
        p_u = u_ptr + i_b * stride_s_b + i_h * stride_s_h + i_c * BT * stride_s_t + offs_t[:, None] * stride_s_t + offs_k[None, :]
        p_pre = pre_ptr + i_b * stride_p_b + i_h * stride_p_h + i_c * BT * stride_p_t + offs_t * stride_p_t
        p_W = W_ptr + i_b * stride_wb + i_h * stride_wh + i_c * BT * stride_wt + offs_t[:, None] * stride_wt + offs_t[None, :] * stride_wc
        p_decay = chunk_decay_ptr + i_b * stride_cb + i_h * stride_ch + i_c * stride_cn
        p_lambda = lambda_local_ptr + i_b * stride_db + i_h * stride_dh + i_c * stride_dn + offs_k

        b_s = tl.load(p_s, mask=mask_k[None, :], other=0.0).to(tl.float32)
        b_u = tl.load(p_u, mask=mask_k[None, :], other=0.0).to(tl.float32)
        b_pre = tl.load(p_pre).to(tl.float32)
        b_W = tl.load(p_W).to(tl.float32)
        b_decay = tl.load(p_decay).to(tl.float32)
        b_end = tl.log(b_decay)

        U_lam = tl.sum(b_u * lam_next[None, :], axis=1) * tl.exp(b_end - b_pre)
        WT_U_lam = tl.sum(b_W * U_lam[:, None], axis=0)
        S_WT_U_lam = tl.sum(b_s * (tl.exp(b_pre) * WT_U_lam)[:, None], axis=0)
        b_lambda = tl.load(p_lambda, mask=mask_k, other=0.0).to(tl.float32)
        lam_next = b_decay * lam_next - S_WT_U_lam + b_lambda

    if STORE_INITIAL_STATE_GRADIENT:
        p_gi = grad_initial_d_ptr + i_b * H * K + i_h * K + offs_k
        tl.store(p_gi, lam_next.to(p_gi.dtype.element_ty), mask=mask_k)


@triton.jit
def osgm_dd_decay_beta_wy_bwd_pass3_param_kernel(
    k_ptr, d_out_ptr, g_ptr, beta_ptr, dd_in_ptr, g_d_next_ptr,
    dk_out_ptr, dg_out_ptr, dbeta_out_ptr,
    eta, d_min, d_max,
    stride_kb, stride_kt, stride_kh,
    stride_gb, stride_gt, stride_gh,
    stride_bb, stride_bt, stride_bh,
    stride_db, stride_dh, stride_dn,
    T: tl.constexpr, H: tl.constexpr, K: tl.constexpr, BK: tl.constexpr, BT: tl.constexpr,
    USE_DENOMINATOR: tl.constexpr, USE_PROJECTION: tl.constexpr,
):
    i_c, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H

    offs = tl.arange(0, BK)
    mask = offs < K
    p_next = g_d_next_ptr + i_b * stride_db + i_h * stride_dh + i_c * stride_dn + offs
    lam = tl.load(p_next, mask=mask, other=0.0).to(tl.float32)

    p_k = k_ptr + i_b * stride_kb + (i_c * BT + BT - 1) * stride_kt + i_h * stride_kh + offs
    p_d = d_out_ptr + i_b * stride_kb + (i_c * BT + BT - 1) * stride_kt + i_h * stride_kh + offs
    p_dd = dd_in_ptr + i_b * stride_kb + (i_c * BT + BT - 1) * stride_kt + i_h * stride_kh + offs
    p_dk = dk_out_ptr + i_b * stride_kb + (i_c * BT + BT - 1) * stride_kt + i_h * stride_kh + offs
    p_g = g_ptr + i_b * stride_gb + (i_c * BT + BT - 1) * stride_gt + i_h * stride_gh
    p_dg = dg_out_ptr + i_b * stride_gb + (i_c * BT + BT - 1) * stride_gt + i_h * stride_gh
    p_beta = beta_ptr + i_b * stride_bb + (i_c * BT + BT - 1) * stride_bt + i_h * stride_bh
    p_dbeta = dbeta_out_ptr + i_b * stride_bb + (i_c * BT + BT - 1) * stride_bt + i_h * stride_bh

    for _ in range(BT):
        b_k = tl.load(p_k, mask=mask, other=0.0).to(tl.float32)
        b_d = tl.load(p_d, mask=mask, other=0.0).to(tl.float32)
        b_dd = tl.load(p_dd, mask=mask, other=0.0).to(tl.float32)
        b_g = tl.load(p_g).to(tl.float32)
        b_beta = tl.load(p_beta).to(tl.float32)
        gamma = tl.exp(b_g)

        k_sq = b_k * b_k
        inner = tl.sum(b_d * k_sq)
        term_a = 1.0 - b_beta * inner

        if USE_PROJECTION:
            if USE_DENOMINATOR:
                sum_k_sq_pre = tl.sum(k_sq) + 1e-5
                grad_d_pre = b_beta * term_a / sum_k_sq_pre
            else:
                grad_d_pre = b_beta * term_a
            d_next_pre = gamma * b_d + eta * grad_d_pre * k_sq
            mask_proj = (d_next_pre >= d_min) & (d_next_pre <= d_max)
            lam = tl.where(mask_proj, lam, 0.0)

        b_dg = tl.sum(lam * b_d) * gamma
        tl.store(p_dg, b_dg)
        term_b = tl.sum(lam * k_sq)

        if USE_DENOMINATOR:
            sum_k_sq = tl.sum(k_sq) + 1e-5
            grad_k_sq = (eta * b_beta / sum_k_sq) * (
                term_a * lam
                - b_beta * term_b * b_d
                - (term_a * term_b / sum_k_sq)
            )
            lam = gamma * lam - eta * (b_beta * b_beta) * (term_b / sum_k_sq) * k_sq
            b_dbeta = eta * term_b * (1.0 - 2.0 * b_beta * inner) / sum_k_sq
        else:
            grad_k_sq = eta * (b_beta * term_a * lam - (b_beta * b_beta) * term_b * b_d)
            lam = gamma * lam - eta * (b_beta * b_beta) * term_b * k_sq
            b_dbeta = eta * term_b * (1.0 - 2.0 * b_beta * inner)

        b_dk = 2.0 * b_k * grad_k_sq
        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), mask=mask)
        tl.store(p_dbeta, b_dbeta.to(p_dbeta.dtype.element_ty))

        lam += b_dd

        p_k -= stride_kt
        p_d -= stride_kt
        p_dd -= stride_kt
        p_dk -= stride_kt
        p_g -= stride_gt
        p_dg -= stride_gt
        p_beta -= stride_bt
        p_dbeta -= stride_bt


def _make_wy_inputs(
    k: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    eta: float,
    use_denominator: bool,
    chunk_size: int = 64,
):
    B, T, H, K = k.shape
    if T % chunk_size != 0:
        raise NotImplementedError("dd-decay WY phase1 currently requires T divisible by chunk_size.")

    NT = T // chunk_size
    k_sq = k.float().square()
    beta_e = beta.float().unsqueeze(-1)
    s = beta_e * k_sq
    if use_denominator:
        denom = k_sq.sum(dim=-1, keepdim=True) + 1e-5
        u = eta * beta_e * k_sq / denom
    else:
        u = eta * beta_e * k_sq

    g_f = g.float()
    g_chunks = g_f.view(B, NT, chunk_size, H)
    post = g_chunks.cumsum(dim=2)
    pre = post - g_chunks
    chunk_decay = post[:, :, -1, :].exp().transpose(1, 2).contiguous()
    pre_flat = pre.view(B, T, H)

    # A_{ij} needs exp(pre_i - post_j).  The shared scaled-dot kernel uses
    # exp(g_i - g_j), so absorb exp(-g_j) into u_j.  This is a one-token
    # factor, unlike the unstable 1 / prod(gamma) transform.
    u_tilde = u * (-g_f).exp().unsqueeze(-1)
    return (
        s.to(k.dtype),
        u_tilde.to(k.dtype),
        pre_flat.to(torch.float32),
        chunk_decay.to(torch.float32),
    )


def _normalize_w(W: torch.Tensor, T: int, cu_seqlens: torch.Tensor = None):
    if W.dim() == 4 and (W.shape[1] == T or (cu_seqlens is not None and W.shape[1] == T)):
        return W.transpose(1, 2).contiguous()
    return W.contiguous()


def compute_osgm_dd_decay_beta_phase1_fwd_wy(
    k: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    eta: float,
    use_denominator: bool,
    d_min: float,
    d_max: float,
    cu_seqlens: torch.Tensor = None,
    initial_d: torch.Tensor = None,
    output_final_state: bool = False,
):
    if cu_seqlens is not None:
        raise NotImplementedError("dd-decay WY phase1 varlen support is not implemented yet.")

    B, T, H, K = k.shape
    BK = triton.next_power_of_2(K)
    s, u, pre, chunk_decay = _make_wy_inputs(k, g, beta, eta, use_denominator)
    beta_dummy = torch.ones_like(beta)

    with torch.no_grad():
        A = chunk_scaled_dot_qkw_fwd(q=s, k=u, g=pre, beta=beta_dummy, cu_seqlens=None, chunk_size=64)
        W = solve_tril(A=A, cu_seqlens=None, output_dtype=k.dtype)

    W_bh = _normalize_w(W, T, None)
    BT = W_bh.shape[-1]
    NT = triton.cdiv(T, BT)

    s_bh = s.transpose(1, 2).contiguous()
    u_bh = u.transpose(1, 2).contiguous()
    pre_bh = pre.transpose(1, 2).contiguous()
    d_out = torch.empty_like(k)
    d_starts_bh = torch.empty(B, H, NT, K, device=k.device, dtype=k.dtype)
    final_d = torch.empty(B, H, K, device=k.device, dtype=k.dtype) if output_final_state else None

    osgm_dd_decay_beta_wy_cross_chunk_scan_kernel[(B * H,)](
        s_bh, u_bh, pre_bh, W_bh, d_starts_bh, chunk_decay,
        initial_d, final_d,
        d_min or 0.0, d_max or 0.0,
        s_bh.stride(0), s_bh.stride(1), s_bh.stride(2),
        pre_bh.stride(0), pre_bh.stride(1), pre_bh.stride(2),
        W_bh.stride(0), W_bh.stride(1), W_bh.stride(2), W_bh.stride(3),
        d_starts_bh.stride(0), d_starts_bh.stride(1), d_starts_bh.stride(2),
        chunk_decay.stride(0), chunk_decay.stride(1), chunk_decay.stride(2),
        T, H, K, BK, BT,
        USE_PROJECTION=(d_min is not None and d_max is not None),
        USE_INITIAL_STATE=(initial_d is not None),
        USE_FINAL_STATE=output_final_state,
    )

    osgm_dd_decay_beta_wy_intra_chunk_kernel[(NT, B * H)](
        k, g, beta, d_out, d_starts_bh,
        eta, d_min or 0.0, d_max or 0.0,
        k.stride(0), k.stride(1), k.stride(2),
        g.stride(0), g.stride(1), g.stride(2),
        beta.stride(0), beta.stride(1), beta.stride(2),
        d_starts_bh.stride(0), d_starts_bh.stride(1), d_starts_bh.stride(2),
        T, H, K, BK, BT,
        USE_DENOMINATOR=use_denominator,
        USE_PROJECTION=(d_min is not None and d_max is not None),
    )
    return d_out, final_d, W_bh, d_starts_bh, s_bh, u_bh, pre_bh, chunk_decay


def compute_osgm_dd_decay_beta_phase1_bwd_wy(
    k: torch.Tensor,
    g: torch.Tensor,
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
    s_bh: torch.Tensor = None,
    u_bh: torch.Tensor = None,
    pre_bh: torch.Tensor = None,
    chunk_decay: torch.Tensor = None,
):
    if cu_seqlens is not None:
        raise NotImplementedError("dd-decay WY phase1 varlen support is not implemented yet.")
    if W_bh is None or d_starts_bh is None:
        raise ValueError("W_bh and d_starts_bh from forward are required.")
    if s_bh is None or u_bh is None or pre_bh is None or chunk_decay is None:
        s, u, pre, chunk_decay = _make_wy_inputs(k, g, beta, eta, use_denominator)
        s_bh = s.transpose(1, 2).contiguous()
        u_bh = u.transpose(1, 2).contiguous()
        pre_bh = pre.transpose(1, 2).contiguous()

    B, T, H, K = k.shape
    BK = triton.next_power_of_2(K)
    BT = W_bh.shape[-1]
    NT = d_starts_bh.shape[2]

    lambda_local_bh = torch.empty(B, H, NT, K, device=k.device, dtype=k.dtype)
    g_d_next_bh = torch.empty(B, H, NT, K, device=k.device, dtype=k.dtype)

    osgm_dd_decay_beta_wy_bwd_pass1_local_kernel[(NT, B * H)](
        k, d_out, g, beta, dd_in, lambda_local_bh,
        eta, d_min or 0.0, d_max or 0.0,
        k.stride(0), k.stride(1), k.stride(2),
        g.stride(0), g.stride(1), g.stride(2),
        beta.stride(0), beta.stride(1), beta.stride(2),
        lambda_local_bh.stride(0), lambda_local_bh.stride(1), lambda_local_bh.stride(2),
        T, H, K, BK, BT,
        USE_DENOMINATOR=use_denominator,
        USE_PROJECTION=(d_min is not None and d_max is not None),
    )

    dd_initial = torch.empty(B, H, K, device=k.device, dtype=k.dtype) if output_initial_state_gradient else None
    osgm_dd_decay_beta_wy_bwd_pass2_scan_kernel[(B * H,)](
        s_bh, u_bh, pre_bh, W_bh, lambda_local_bh, chunk_decay, g_d_next_bh,
        dd_final, dd_initial,
        s_bh.stride(0), s_bh.stride(1), s_bh.stride(2),
        pre_bh.stride(0), pre_bh.stride(1), pre_bh.stride(2),
        W_bh.stride(0), W_bh.stride(1), W_bh.stride(2), W_bh.stride(3),
        g_d_next_bh.stride(0), g_d_next_bh.stride(1), g_d_next_bh.stride(2),
        chunk_decay.stride(0), chunk_decay.stride(1), chunk_decay.stride(2),
        T, H, K, BK, BT,
        USE_FINAL_STATE_GRADIENT=(dd_final is not None),
        STORE_INITIAL_STATE_GRADIENT=output_initial_state_gradient,
    )

    dk_out = torch.empty_like(k)
    dg = torch.zeros(B, T, H, device=k.device, dtype=torch.float32)
    dbeta = torch.empty_like(beta)
    osgm_dd_decay_beta_wy_bwd_pass3_param_kernel[(NT, B * H)](
        k, d_out, g, beta, dd_in, g_d_next_bh,
        dk_out, dg, dbeta,
        eta, d_min or 0.0, d_max or 0.0,
        k.stride(0), k.stride(1), k.stride(2),
        g.stride(0), g.stride(1), g.stride(2),
        beta.stride(0), beta.stride(1), beta.stride(2),
        g_d_next_bh.stride(0), g_d_next_bh.stride(1), g_d_next_bh.stride(2),
        T, H, K, BK, BT,
        USE_DENOMINATOR=use_denominator,
        USE_PROJECTION=(d_min is not None and d_max is not None),
    )
    return dk_out, dd_initial, dg, dbeta
