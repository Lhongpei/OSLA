# -*- coding: utf-8 -*-
# EMA Phase1: replace OSGM's SGD update with exponential moving average.
#
# Forward:
#   d(t) = 1 / (ema(t) + eps)          [unnormalized]
#   d(t) = d_raw(t) / <d_raw(t), k²>   [normalized, so <d,k²>=1]
#   ema(t+1) = alpha * ema(t) + (1-alpha) * k(t)²
#
# The recurrent state is `ema`, not `d`.

import torch
import triton
import triton.language as tl

EPS: tl.constexpr = 1e-6


@triton.jit
def ema_phase1_fwd_kernel(
    k_ptr, d_out_ptr, ema_out_ptr, cu_seqlens_ptr,
    initial_ema_ptr, final_ema_ptr,
    alpha,
    stride_kb, stride_kt, stride_kh,
    T: tl.constexpr, H: tl.constexpr, K: tl.constexpr, BK: tl.constexpr,
    NORMALIZE: tl.constexpr,
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

    offset = i_b * stride_kb + bos * stride_kt + i_h * stride_kh + tl.arange(0, BK)
    p_k = k_ptr + offset
    p_d = d_out_ptr + offset
    p_ema = ema_out_ptr + offset
    mask = tl.arange(0, BK) < K

    if USE_INITIAL_STATE:
        p_init = initial_ema_ptr + i_n * H * K + i_h * K + tl.arange(0, BK)
        b_ema = tl.load(p_init, mask=mask, other=1.0).to(tl.float32)
    else:
        # Default: ema=1.0 so d=1/(1+eps)≈1 and <d,k²>≈1 for L2-normed keys
        b_ema = tl.full([BK], 1.0, dtype=tl.float32)

    one_minus_alpha: tl.constexpr = 1.0 - alpha

    for _ in range(bos, eos):
        b_k = tl.load(p_k, mask=mask, other=0.0).to(tl.float32)
        k_sq = b_k * b_k

        # Store ema(t) for backward
        tl.store(p_ema, b_ema.to(p_ema.dtype.element_ty), mask=mask)

        # Compute d from ema
        b_d_raw = 1.0 / (b_ema + EPS)

        if NORMALIZE:
            c = tl.sum(b_d_raw * k_sq) + EPS
            b_d = b_d_raw / c
        else:
            b_d = b_d_raw

        # Store d(t) for phase 2
        tl.store(p_d, b_d.to(p_d.dtype.element_ty), mask=mask)

        # Update ema
        b_ema = alpha * b_ema + one_minus_alpha * k_sq

        p_k += stride_kt
        p_d += stride_kt
        p_ema += stride_kt

    if STORE_FINAL_STATE:
        p_final = final_ema_ptr + i_n * H * K + i_h * K + tl.arange(0, BK)
        tl.store(p_final, b_ema.to(p_final.dtype.element_ty), mask=mask)


@triton.jit
def ema_phase1_bwd_kernel(
    k_ptr, d_out_ptr, ema_out_ptr, dd_in_ptr, dk_out_ptr, cu_seqlens_ptr,
    dd_final_ptr, dd_initial_ptr,
    alpha,
    stride_kb, stride_kt, stride_kh,
    T: tl.constexpr, H: tl.constexpr, K: tl.constexpr, BK: tl.constexpr,
    NORMALIZE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_FINAL_STATE_GRADIENT: tl.constexpr, STORE_INITIAL_STATE_GRADIENT: tl.constexpr
):
    """
    Backward through EMA phase 1.

    Adjoint variable `adj_ema` propagates backward through:
      ema(t+1) = alpha * ema(t) + (1-alpha) * k(t)²   →  adj_ema *= alpha
      d(t) = f(ema(t), k(t))                            →  adj_ema += dd(t) * ∂d/∂ema
    """
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

    # Point to last timestep (iterate backward)
    offset = i_b * stride_kb + (eos - 1) * stride_kt + i_h * stride_kh + tl.arange(0, BK)
    p_k = k_ptr + offset
    p_d = d_out_ptr + offset
    p_ema = ema_out_ptr + offset
    p_dd = dd_in_ptr + offset
    p_dk = dk_out_ptr + offset
    mask = tl.arange(0, BK) < K

    one_minus_alpha: tl.constexpr = 1.0 - alpha

    # Initialize adjoint of ema state
    if USE_FINAL_STATE_GRADIENT:
        p_dd_final = dd_final_ptr + i_n * H * K + i_h * K + tl.arange(0, BK)
        b_adj_ema = tl.load(p_dd_final, mask=mask, other=0.0).to(tl.float32)
    else:
        b_adj_ema = tl.zeros([BK], dtype=tl.float32)

    for _ in range(bos, eos):
        b_k = tl.load(p_k, mask=mask, other=0.0).to(tl.float32)
        b_d = tl.load(p_d, mask=mask, other=0.0).to(tl.float32)
        b_ema_t = tl.load(p_ema, mask=mask, other=1.0).to(tl.float32)
        b_dd = tl.load(p_dd, mask=mask, other=0.0).to(tl.float32)

        k_sq = b_k * b_k
        b_d_raw = 1.0 / (b_ema_t + EPS)

        # --- dk from ema update: ema(t+1) = alpha*ema(t) + (1-alpha)*k²  ---
        b_dk = b_adj_ema * 2.0 * one_minus_alpha * b_k

        # --- Propagate adj_ema backward through ema recurrence ---
        b_adj_ema = alpha * b_adj_ema

        # --- Add contribution from d(t) → ema(t) ---
        if NORMALIZE:
            # d(t) = d_raw(t) / c(t),  c(t) = <d_raw, k²>
            c = tl.sum(b_d_raw * k_sq) + EPS
            # Gradient through normalization: dd_raw = (dd - k² * <dd, d>) / c
            dd_d_inner = tl.sum(b_dd * b_d)
            b_dd_raw = (b_dd - k_sq * dd_d_inner) / c
            # dk from normalization: ∂c/∂k_i = 2*k_i*d_raw_i → chain rule
            b_dk += -2.0 * b_k * b_d * dd_d_inner
            # adj_ema from d_raw = 1/(ema+eps): ∂d_raw/∂ema = -d_raw²
            b_adj_ema += b_dd_raw * (-b_d_raw * b_d_raw)
        else:
            # d(t) = 1/(ema(t)+eps):  ∂d/∂ema = -d_raw²
            b_adj_ema += b_dd * (-b_d_raw * b_d_raw)

        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), mask=mask)

        p_k -= stride_kt
        p_d -= stride_kt
        p_ema -= stride_kt
        p_dd -= stride_kt
        p_dk -= stride_kt

    if STORE_INITIAL_STATE_GRADIENT:
        p_dd_init = dd_initial_ptr + i_n * H * K + i_h * K + tl.arange(0, BK)
        tl.store(p_dd_init, b_adj_ema.to(p_dd_init.dtype.element_ty), mask=mask)


# ─── Python wrappers ───


def compute_ema_phase1_fwd(
    k: torch.Tensor, alpha: float, normalize: bool,
    cu_seqlens: torch.Tensor = None,
    initial_ema: torch.Tensor = None,
    output_final_state: bool = False,
):
    B, T, H, K = k.shape
    BK = triton.next_power_of_2(K)
    N = len(cu_seqlens) - 1 if cu_seqlens is not None else B

    d_out = torch.empty_like(k)
    ema_out = torch.empty_like(k)
    final_ema = torch.empty(N, H, K, device=k.device, dtype=k.dtype) if output_final_state else None

    ema_phase1_fwd_kernel[(N * H,)](
        k, d_out, ema_out, cu_seqlens,
        initial_ema, final_ema,
        alpha,
        k.stride(0), k.stride(1), k.stride(2),
        T, H, K, BK,
        NORMALIZE=normalize,
        IS_VARLEN=(cu_seqlens is not None),
        USE_INITIAL_STATE=(initial_ema is not None),
        STORE_FINAL_STATE=output_final_state,
        num_warps=1, num_stages=1
    )
    return d_out, ema_out, final_ema


def compute_ema_phase1_bwd(
    k: torch.Tensor, d_out: torch.Tensor, ema_out: torch.Tensor,
    dd_in: torch.Tensor, alpha: float, normalize: bool,
    cu_seqlens: torch.Tensor = None,
    dd_final: torch.Tensor = None,
    output_initial_state_gradient: bool = False,
):
    B, T, H, K = k.shape
    BK = triton.next_power_of_2(K)
    N = len(cu_seqlens) - 1 if cu_seqlens is not None else B

    dk_out = torch.empty_like(k)
    dd_initial = torch.empty(N, H, K, device=k.device, dtype=k.dtype) if output_initial_state_gradient else None

    ema_phase1_bwd_kernel[(N * H,)](
        k, d_out, ema_out, dd_in, dk_out, cu_seqlens,
        dd_final, dd_initial,
        alpha,
        k.stride(0), k.stride(1), k.stride(2),
        T, H, K, BK,
        NORMALIZE=normalize,
        IS_VARLEN=(cu_seqlens is not None),
        USE_FINAL_STATE_GRADIENT=(dd_final is not None),
        STORE_INITIAL_STATE_GRADIENT=output_initial_state_gradient,
        num_warps=1, num_stages=1
    )
    return dk_out, dd_initial
