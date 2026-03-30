import torch
import triton
import triton.language as tl

@triton.jit
def osgm_phase1_fwd_kernel(
    k_ptr, d_out_ptr, eta, d_min, d_max,
    stride_kb, stride_kt, stride_kh,
    T: tl.constexpr, H: tl.constexpr, K: tl.constexpr, BK: tl.constexpr,
    USE_DENOMINATOR: tl.constexpr, USE_PROJECTION: tl.constexpr
):
    i_bh = tl.program_id(0)
    i_b = i_bh // H
    i_h = i_bh % H

    offset = i_b * stride_kb + i_h * stride_kh + tl.arange(0, BK)
    p_k = k_ptr + offset
    p_d = d_out_ptr + offset
    mask = tl.arange(0, BK) < K

    d_curr = tl.zeros([BK], dtype=tl.float32)

    for _ in range(T):
        b_k = tl.load(p_k, mask=mask, other=0.0).to(tl.float32)
        
        tl.store(p_d, d_curr.to(p_d.dtype.element_ty), mask=mask)

        k_sq = b_k * b_k
        term_A = 1.0 - tl.sum(d_curr * k_sq)
        
        if USE_DENOMINATOR:
            sum_k_sq = tl.sum(k_sq) + 1e-5
            grad_d = term_A / sum_k_sq
        else:
            grad_d = term_A

        d_next = d_curr + eta * grad_d * k_sq

        if USE_PROJECTION:
            d_next = tl.maximum(d_next, d_min)
            d_next = tl.minimum(d_next, d_max)

        d_curr = d_next

        p_k += stride_kt
        p_d += stride_kt


@triton.jit
def osgm_phase1_bwd_kernel(
    k_ptr, d_out_ptr, dd_in_ptr, dk_out_ptr, eta, d_min, d_max,
    stride_kb, stride_kt, stride_kh,
    T: tl.constexpr, H: tl.constexpr, K: tl.constexpr, BK: tl.constexpr,
    USE_DENOMINATOR: tl.constexpr, USE_PROJECTION: tl.constexpr
):
    i_bh = tl.program_id(0)
    i_b = i_bh // H
    i_h = i_bh % H

    offset = i_b * stride_kb + (T - 1) * stride_kt + i_h * stride_kh + tl.arange(0, BK)
    p_k = k_ptr + offset
    p_d = d_out_ptr + offset
    p_dd = dd_in_ptr + offset
    p_dk = dk_out_ptr + offset
    mask = tl.arange(0, BK) < K

    b_dd_state = tl.zeros([BK], dtype=tl.float32)

    for _ in range(T):
        b_k = tl.load(p_k, mask=mask, other=0.0).to(tl.float32)
        b_d = tl.load(p_d, mask=mask, other=0.0).to(tl.float32)
        b_dd_curr = tl.load(p_dd, mask=mask, other=0.0).to(tl.float32)

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
            b_dd_state = tl.where(mask_proj, b_dd_state, 0.0)

        term_B = tl.sum(b_dd_state * k_sq)

        if USE_DENOMINATOR:
            grad_w = (eta / sum_k_sq) * (term_A * b_dd_state - term_B * b_d - (term_A * term_B / sum_k_sq))
            b_dd_state = b_dd_state - eta * (term_B / sum_k_sq) * k_sq
        else:
            grad_w = eta * (term_A * b_dd_state - term_B * b_d)
            b_dd_state = b_dd_state - eta * term_B * k_sq

        b_dk = 2.0 * b_k * grad_w
        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), mask=mask)

        b_dd_state += b_dd_curr

        p_k -= stride_kt
        p_d -= stride_kt
        p_dd -= stride_kt
        p_dk -= stride_kt


def compute_osgm_phase1_fwd(k: torch.Tensor, eta: float, use_denominator: bool, d_min: float, d_max: float):
    B, T, H, K = k.shape
    d_out = torch.empty_like(k)
    BK = triton.next_power_of_2(K)
    USE_PROJECTION = (d_min is not None) and (d_max is not None)
    d_min_val = float(d_min) if d_min is not None else 0.0
    d_max_val = float(d_max) if d_max is not None else 0.0
    
    osgm_phase1_fwd_kernel[(B * H,)](
        k, d_out, eta, d_min_val, d_max_val,
        k.stride(0), k.stride(1), k.stride(2),
        T, H, K, BK,
        USE_DENOMINATOR=use_denominator, USE_PROJECTION=USE_PROJECTION
    )
    return d_out

def compute_osgm_phase1_bwd(k: torch.Tensor, d_out: torch.Tensor, dd_in: torch.Tensor, eta: float, use_denominator: bool, d_min: float, d_max: float):
    B, T, H, K = k.shape
    dk_out = torch.empty_like(k)
    BK = triton.next_power_of_2(K)
    USE_PROJECTION = (d_min is not None) and (d_max is not None)
    d_min_val = float(d_min) if d_min is not None else 0.0
    d_max_val = float(d_max) if d_max is not None else 0.0
    
    osgm_phase1_bwd_kernel[(B * H,)](
        k, d_out, dd_in, dk_out, eta, d_min_val, d_max_val,
        k.stride(0), k.stride(1), k.stride(2),
        T, H, K, BK,
        USE_DENOMINATOR=use_denominator, USE_PROJECTION=USE_PROJECTION
    )
    return dk_out