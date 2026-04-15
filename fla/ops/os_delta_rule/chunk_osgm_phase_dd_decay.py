# -*- coding: utf-8 -*-
# OSGM Phase1 with data-dependent (per-token per-head) decay on preconditioner D.
# D_next = exp(g_t) * D + eta * grad_d * k^2
# where g_t = logsigmoid(a_proj(hidden_states)) is per-token per-head in log-space.

import torch
import triton
import triton.language as tl


@triton.jit
def osgm_dd_decay_phase1_fwd_kernel(
    k_ptr, d_out_ptr, g_ptr, cu_seqlens_ptr,
    initial_d_ptr, final_d_ptr,
    eta, d_min, d_max,
    stride_kb, stride_kt, stride_kh,
    stride_gb, stride_gt, stride_gh,
    T: tl.constexpr, H: tl.constexpr, K: tl.constexpr, BK: tl.constexpr,
    USE_DENOMINATOR: tl.constexpr, USE_PROJECTION: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr, STORE_FINAL_STATE: tl.constexpr
):
    i_nh = tl.program_id(0)
    i_n = i_nh // H
    i_h = i_nh % H

    if IS_VARLEN:
        bos = tl.load(cu_seqlens_ptr + i_n).to(tl.int32)
        eos = tl.load(cu_seqlens_ptr + i_n + 1).to(tl.int32)
        i_b = 0
    else:
        bos = 0
        eos = T
        i_b = i_n

    offset_k = i_b * stride_kb + bos * stride_kt + i_h * stride_kh + tl.arange(0, BK)
    p_k = k_ptr + offset_k
    p_d = d_out_ptr + offset_k
    mask = tl.arange(0, BK) < K

    # g pointer: [B, T, H] layout
    offset_g = i_b * stride_gb + bos * stride_gt + i_h * stride_gh
    p_g = g_ptr + offset_g

    if USE_INITIAL_STATE:
        p_initial_d = initial_d_ptr + i_n * H * K + i_h * K + tl.arange(0, BK)
        d_curr = tl.load(p_initial_d, mask=mask, other=0.0).to(tl.float32)
    else:
        d_curr = tl.zeros([BK], dtype=tl.float32)

    for _ in range(bos, eos):
        b_k = tl.load(p_k, mask=mask, other=0.0).to(tl.float32)

        tl.store(p_d, d_curr.to(p_d.dtype.element_ty), mask=mask)

        k_sq = b_k * b_k
        term_A = 1.0 - tl.sum(d_curr * k_sq)

        if USE_DENOMINATOR:
            sum_k_sq = tl.sum(k_sq) + 1e-5
            grad_d = term_A / sum_k_sq
        else:
            grad_d = term_A

        # Load per-token decay: g is in log-space, gamma = exp(g)
        b_g = tl.load(p_g).to(tl.float32)
        b_gamma = tl.exp(b_g)

        d_next = b_gamma * d_curr + eta * grad_d * k_sq

        if USE_PROJECTION:
            d_next = tl.maximum(d_next, d_min)
            d_next = tl.minimum(d_next, d_max)

        d_curr = d_next

        p_k += stride_kt
        p_d += stride_kt
        p_g += stride_gt

    if STORE_FINAL_STATE:
        p_final_d = final_d_ptr + i_n * H * K + i_h * K + tl.arange(0, BK)
        tl.store(p_final_d, d_curr.to(p_final_d.dtype.element_ty), mask=mask)


@triton.jit
def osgm_dd_decay_phase1_bwd_kernel(
    k_ptr, d_out_ptr, g_ptr, dd_in_ptr, dk_out_ptr, dg_out_ptr,
    cu_seqlens_ptr,
    dd_final_ptr, dd_initial_ptr,
    eta, d_min, d_max,
    stride_kb, stride_kt, stride_kh,
    stride_gb, stride_gt, stride_gh,
    T: tl.constexpr, H: tl.constexpr, K: tl.constexpr, BK: tl.constexpr,
    USE_DENOMINATOR: tl.constexpr, USE_PROJECTION: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_FINAL_STATE_GRADIENT: tl.constexpr, STORE_INITIAL_STATE_GRADIENT: tl.constexpr
):
    i_nh = tl.program_id(0)
    i_n = i_nh // H
    i_h = i_nh % H

    if IS_VARLEN:
        bos = tl.load(cu_seqlens_ptr + i_n).to(tl.int32)
        eos = tl.load(cu_seqlens_ptr + i_n + 1).to(tl.int32)
        i_b = 0
    else:
        bos = 0
        eos = T
        i_b = i_n

    # Start from last token (backward pass)
    offset_k = i_b * stride_kb + (eos - 1) * stride_kt + i_h * stride_kh + tl.arange(0, BK)
    p_k = k_ptr + offset_k
    p_d = d_out_ptr + offset_k
    p_dd = dd_in_ptr + offset_k
    p_dk = dk_out_ptr + offset_k
    mask = tl.arange(0, BK) < K

    # g and dg pointers at last token
    offset_g = i_b * stride_gb + (eos - 1) * stride_gt + i_h * stride_gh
    p_g = g_ptr + offset_g
    p_dg = dg_out_ptr + offset_g

    if USE_FINAL_STATE_GRADIENT:
        p_dd_final = dd_final_ptr + i_n * H * K + i_h * K + tl.arange(0, BK)
        b_dd_state = tl.load(p_dd_final, mask=mask, other=0.0).to(tl.float32)
    else:
        b_dd_state = tl.zeros([BK], dtype=tl.float32)

    for _ in range(bos, eos):
        b_k = tl.load(p_k, mask=mask, other=0.0).to(tl.float32)
        b_d = tl.load(p_d, mask=mask, other=0.0).to(tl.float32)
        b_dd_curr = tl.load(p_dd, mask=mask, other=0.0).to(tl.float32)
        b_g = tl.load(p_g).to(tl.float32)
        b_gamma = tl.exp(b_g)

        k_sq = b_k * b_k
        term_A = 1.0 - tl.sum(b_d * k_sq)

        if USE_DENOMINATOR:
            sum_k_sq = tl.sum(k_sq) + 1e-5
            grad_d = term_A / sum_k_sq
        else:
            grad_d = term_A

        if USE_PROJECTION:
            d_next_pre = b_gamma * b_d + eta * grad_d * k_sq
            mask_proj = (d_next_pre >= d_min) & (d_next_pre <= d_max)
            b_dd_state = tl.where(mask_proj, b_dd_state, 0.0)

        # Gradient for g_t: dg_t = sum(dd_state * d_t) * gamma_t
        # Because d(gamma*D)/d(g) = d(exp(g)*D)/d(g) = exp(g)*D = gamma*D
        b_dg = tl.sum(b_dd_state * b_d) * b_gamma
        tl.store(p_dg, b_dg)

        term_B = tl.sum(b_dd_state * k_sq)

        if USE_DENOMINATOR:
            grad_w = (eta / sum_k_sq) * (term_A * b_dd_state - term_B * b_d - (term_A * term_B / sum_k_sq))
            b_dd_state = b_gamma * b_dd_state - eta * (term_B / sum_k_sq) * k_sq
        else:
            grad_w = eta * (term_A * b_dd_state - term_B * b_d)
            b_dd_state = b_gamma * b_dd_state - eta * term_B * k_sq

        b_dk = 2.0 * b_k * grad_w
        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), mask=mask)

        b_dd_state += b_dd_curr

        p_k -= stride_kt
        p_d -= stride_kt
        p_dd -= stride_kt
        p_dk -= stride_kt
        p_g -= stride_gt
        p_dg -= stride_gt

    if STORE_INITIAL_STATE_GRADIENT:
        p_dd_initial = dd_initial_ptr + i_n * H * K + i_h * K + tl.arange(0, BK)
        tl.store(p_dd_initial, b_dd_state.to(p_dd_initial.dtype.element_ty), mask=mask)


def compute_osgm_dd_decay_phase1_fwd(
    k: torch.Tensor, g: torch.Tensor,
    eta: float, use_denominator: bool, d_min: float, d_max: float,
    cu_seqlens: torch.Tensor = None, initial_d: torch.Tensor = None,
    output_final_state: bool = False
):
    B, T, H, K = k.shape
    BK = triton.next_power_of_2(K)
    N = len(cu_seqlens) - 1 if cu_seqlens is not None else B

    d_out = torch.empty_like(k)
    final_d = torch.empty(N, H, K, device=k.device, dtype=k.dtype) if output_final_state else None

    osgm_dd_decay_phase1_fwd_kernel[(N * H,)](
        k, d_out, g, cu_seqlens,
        initial_d, final_d,
        eta, d_min or 0.0, d_max or 0.0,
        k.stride(0), k.stride(1), k.stride(2),
        g.stride(0), g.stride(1), g.stride(2),
        T, H, K, BK,
        USE_DENOMINATOR=use_denominator,
        USE_PROJECTION=(d_min is not None and d_max is not None),
        IS_VARLEN=(cu_seqlens is not None),
        USE_INITIAL_STATE=(initial_d is not None),
        STORE_FINAL_STATE=output_final_state,
        num_warps=1, num_stages=1
    )
    return d_out, final_d


def compute_osgm_dd_decay_phase1_bwd(
    k: torch.Tensor, g: torch.Tensor,
    d_out: torch.Tensor, dd_in: torch.Tensor,
    eta: float, use_denominator: bool, d_min: float, d_max: float,
    cu_seqlens: torch.Tensor = None, dd_final: torch.Tensor = None,
    output_initial_state_gradient: bool = False
):
    B, T, H, K = k.shape
    BK = triton.next_power_of_2(K)
    N = len(cu_seqlens) - 1 if cu_seqlens is not None else B

    dk_out = torch.empty_like(k)
    dd_initial = torch.empty(N, H, K, device=k.device, dtype=k.dtype) if output_initial_state_gradient else None
    dg = torch.zeros(B, T, H, device=k.device, dtype=torch.float32)

    osgm_dd_decay_phase1_bwd_kernel[(N * H,)](
        k, d_out, g, dd_in, dk_out, dg,
        cu_seqlens,
        dd_final, dd_initial,
        eta, d_min or 0.0, d_max or 0.0,
        k.stride(0), k.stride(1), k.stride(2),
        g.stride(0), g.stride(1), g.stride(2),
        T, H, K, BK,
        USE_DENOMINATOR=use_denominator,
        USE_PROJECTION=(d_min is not None and d_max is not None),
        IS_VARLEN=(cu_seqlens is not None),
        USE_FINAL_STATE_GRADIENT=(dd_final is not None),
        STORE_INITIAL_STATE_GRADIENT=output_initial_state_gradient,
        num_warps=1, num_stages=1
    )
    return dk_out, dd_initial, dg
