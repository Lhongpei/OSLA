# -*- coding: utf-8 -*-
# OSGM Phase1 with data-dependent decay AND gate-aware hypergradient.
#
# This is a variant of `chunk_osgm_phase_dd_decay.py` that additionally
# weights the OSGM surrogate-loss k^2 term by `exp(g_gdn)` — the GDN
# state-forget gate. Motivation: in GatedDeltaNet, the state S_{t-1} is
# decayed by exp(g_gdn_t) each step, so a key k_t whose gate is small will
# contribute less to the persistent state. The plain OSGM hypergradient
# `1 - <d, k^2>` treats every k equally, which mis-calibrates d relative to
# what the gated recurrence actually remembers.
#
# Update rule (per token, per head, per K-dim):
#   k_sq_eff = k^2 * exp(g_gdn)
#   term_A   = 1 - <d, k_sq_eff>
#   grad_d   = term_A                       (when USE_DENOMINATOR=False)
#   d_next   = exp(g_osgm) * d + eta * grad_d * k_sq_eff
#
# When g_gdn ≡ 0 everywhere, exp(g_gdn) = 1 and this reduces EXACTLY to
# chunk_osgm_phase_dd_decay.py — so the plain un-gated OSDN path is
# unaffected.

import torch
import triton
import triton.language as tl


@triton.jit
def osgm_dd_decay_gated_phase1_fwd_kernel(
    k_ptr, d_out_ptr, g_osgm_ptr, g_gdn_ptr, cu_seqlens_ptr,
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

    # Both g_osgm and g_gdn share the same [B, T, H] layout / strides.
    offset_g = i_b * stride_gb + bos * stride_gt + i_h * stride_gh
    p_g_osgm = g_osgm_ptr + offset_g
    p_g_gdn = g_gdn_ptr + offset_g

    if USE_INITIAL_STATE:
        p_initial_d = initial_d_ptr + i_n * H * K + i_h * K + tl.arange(0, BK)
        d_curr = tl.load(p_initial_d, mask=mask, other=0.0).to(tl.float32)
    else:
        d_curr = tl.zeros([BK], dtype=tl.float32)

    for _ in range(bos, eos):
        b_k = tl.load(p_k, mask=mask, other=0.0).to(tl.float32)

        # Store d_curr (before update) at d_out[t] — same semantic as dd_decay.
        tl.store(p_d, d_curr.to(p_d.dtype.element_ty), mask=mask)

        b_g_osgm = tl.load(p_g_osgm).to(tl.float32)
        b_g_gdn = tl.load(p_g_gdn).to(tl.float32)
        b_gate_gdn = tl.exp(b_g_gdn)                # scalar per (n, h)

        # Gate-weighted k^2: this is the key math change.
        k_sq_eff = b_k * b_k * b_gate_gdn
        term_A = 1.0 - tl.sum(d_curr * k_sq_eff)

        if USE_DENOMINATOR:
            sum_k_sq_eff = tl.sum(k_sq_eff) + 1e-5
            grad_d = term_A / sum_k_sq_eff
        else:
            grad_d = term_A

        b_gamma_osgm = tl.exp(b_g_osgm)             # OSGM's own decay on d
        d_next = b_gamma_osgm * d_curr + eta * grad_d * k_sq_eff

        if USE_PROJECTION:
            d_next = tl.maximum(d_next, d_min)
            d_next = tl.minimum(d_next, d_max)

        d_curr = d_next

        p_k += stride_kt
        p_d += stride_kt
        p_g_osgm += stride_gt
        p_g_gdn += stride_gt

    if STORE_FINAL_STATE:
        p_final_d = final_d_ptr + i_n * H * K + i_h * K + tl.arange(0, BK)
        tl.store(p_final_d, d_curr.to(p_final_d.dtype.element_ty), mask=mask)


@triton.jit
def osgm_dd_decay_gated_phase1_bwd_kernel(
    k_ptr, d_out_ptr, g_osgm_ptr, g_gdn_ptr,
    dd_in_ptr, dk_out_ptr, dg_osgm_out_ptr, dg_gdn_out_ptr,
    cu_seqlens_ptr, dd_final_ptr, dd_initial_ptr,
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

    # Start from last token (walk backward).
    offset_k = i_b * stride_kb + (eos - 1) * stride_kt + i_h * stride_kh + tl.arange(0, BK)
    p_k = k_ptr + offset_k
    p_d = d_out_ptr + offset_k
    p_dd = dd_in_ptr + offset_k
    p_dk = dk_out_ptr + offset_k
    mask = tl.arange(0, BK) < K

    offset_g = i_b * stride_gb + (eos - 1) * stride_gt + i_h * stride_gh
    p_g_osgm = g_osgm_ptr + offset_g
    p_g_gdn = g_gdn_ptr + offset_g
    p_dg_osgm = dg_osgm_out_ptr + offset_g
    p_dg_gdn = dg_gdn_out_ptr + offset_g

    if USE_FINAL_STATE_GRADIENT:
        p_dd_final = dd_final_ptr + i_n * H * K + i_h * K + tl.arange(0, BK)
        b_dd_state = tl.load(p_dd_final, mask=mask, other=0.0).to(tl.float32)
    else:
        b_dd_state = tl.zeros([BK], dtype=tl.float32)

    for _ in range(bos, eos):
        b_k = tl.load(p_k, mask=mask, other=0.0).to(tl.float32)
        b_d = tl.load(p_d, mask=mask, other=0.0).to(tl.float32)
        b_dd_curr = tl.load(p_dd, mask=mask, other=0.0).to(tl.float32)
        b_g_osgm = tl.load(p_g_osgm).to(tl.float32)
        b_g_gdn = tl.load(p_g_gdn).to(tl.float32)
        b_gamma_osgm = tl.exp(b_g_osgm)
        b_gate_gdn = tl.exp(b_g_gdn)

        k_sq_eff = b_k * b_k * b_gate_gdn
        term_A = 1.0 - tl.sum(b_d * k_sq_eff)

        if USE_DENOMINATOR:
            sum_k_sq_eff = tl.sum(k_sq_eff) + 1e-5
            grad_d = term_A / sum_k_sq_eff
        else:
            grad_d = term_A

        if USE_PROJECTION:
            d_next_pre = b_gamma_osgm * b_d + eta * grad_d * k_sq_eff
            mask_proj = (d_next_pre >= d_min) & (d_next_pre <= d_max)
            b_dd_state = tl.where(mask_proj, b_dd_state, 0.0)

        # dg_osgm: same as dd_decay — gamma_osgm's only role is scaling d_curr.
        # ∂d_next/∂(gamma_osgm) = d_curr element-wise.
        # ∂(gamma_osgm)/∂g_osgm = gamma_osgm.
        b_dg_osgm = tl.sum(b_dd_state * b_d) * b_gamma_osgm
        tl.store(p_dg_osgm, b_dg_osgm)

        # term_B = <dd_state, k_sq_eff>
        term_B = tl.sum(b_dd_state * k_sq_eff)

        # grad_w_eff = dL/d(k_sq_eff), element-wise of size K.
        # Derivation (mirrors dd_decay's grad_w, with k_sq → k_sq_eff):
        #   With w = eta * term_A, d_next_i = gamma*d_i + w*k_sq_eff_i
        #     ∂L/∂k_sq_eff_j = dd_state_j * w + Σ_i dd_state_i * k_sq_eff_i * ∂w/∂k_sq_eff_j
        #                    = eta * term_A * dd_state_j - eta * d_j * term_B
        if USE_DENOMINATOR:
            grad_w_eff = (eta / sum_k_sq_eff) * (
                term_A * b_dd_state - term_B * b_d
                - (term_A * term_B / sum_k_sq_eff)
            )
            b_dd_state = b_gamma_osgm * b_dd_state - eta * (term_B / sum_k_sq_eff) * k_sq_eff
        else:
            grad_w_eff = eta * (term_A * b_dd_state - term_B * b_d)
            b_dd_state = b_gamma_osgm * b_dd_state - eta * term_B * k_sq_eff

        # Split grad_w_eff into dk and dg_gdn via chain rule on k_sq_eff = k^2 * gate_gdn:
        #   dL/dk_j     = grad_w_eff_j * ∂k_sq_eff_j/∂k_j     = grad_w_eff_j * 2·k_j·gate_gdn
        #   dL/dgate_j (per-element, then summed over K for scalar gate):
        #     dL/dgate  = Σ_j grad_w_eff_j * k_j^2   (since ∂k_sq_eff_j/∂gate = k_j^2)
        #   dL/dg_gdn   = dL/dgate * ∂gate/∂g_gdn    = dL/dgate * gate_gdn
        b_dk = 2.0 * b_k * b_gate_gdn * grad_w_eff
        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), mask=mask)

        b_dgate = tl.sum(b_k * b_k * grad_w_eff)
        b_dg_gdn = b_dgate * b_gate_gdn
        tl.store(p_dg_gdn, b_dg_gdn)

        # Absorb the incoming dd[t] (from other paths e.g. saved d_out used in wy)
        # into the state going back in time.
        b_dd_state += b_dd_curr

        p_k -= stride_kt
        p_d -= stride_kt
        p_dd -= stride_kt
        p_dk -= stride_kt
        p_g_osgm -= stride_gt
        p_g_gdn -= stride_gt
        p_dg_osgm -= stride_gt
        p_dg_gdn -= stride_gt

    if STORE_INITIAL_STATE_GRADIENT:
        p_dd_initial = dd_initial_ptr + i_n * H * K + i_h * K + tl.arange(0, BK)
        tl.store(p_dd_initial, b_dd_state.to(p_dd_initial.dtype.element_ty), mask=mask)


def compute_osgm_dd_decay_gated_phase1_fwd(
    k: torch.Tensor, g_osgm: torch.Tensor, g_gdn: torch.Tensor,
    eta: float, use_denominator: bool, d_min: float, d_max: float,
    cu_seqlens: torch.Tensor = None, initial_d: torch.Tensor = None,
    output_final_state: bool = False
):
    """Gate-aware OSGM dd-decay phase1 forward.

    Args:
        k:        [B, T, H, K] keys
        g_osgm:   [B, T, H]    OSGM's own decay in log space (logsigmoid(osgm_a_proj(h)))
        g_gdn:    [B, T, H]    GDN's state-forget gate in log space (fused_gdn_gate output)
        Other args: same as dd_decay version.

    Returns:
        d_out: [B, T, H, K] — d trajectory, d_out[t] is the d used at step t.
        final_d: [B, H, K] or None.
    """
    B, T, H, K = k.shape
    BK = triton.next_power_of_2(K)
    N = len(cu_seqlens) - 1 if cu_seqlens is not None else B

    d_out = torch.empty_like(k)
    final_d = torch.empty(N, H, K, device=k.device, dtype=k.dtype) if output_final_state else None

    # Shape / stride sanity: g_osgm and g_gdn must match.
    assert g_osgm.shape == g_gdn.shape, (
        f"g_osgm {tuple(g_osgm.shape)} and g_gdn {tuple(g_gdn.shape)} must match"
    )
    assert g_osgm.stride() == g_gdn.stride(), (
        f"g_osgm and g_gdn must share strides; got {g_osgm.stride()} vs {g_gdn.stride()}"
    )

    osgm_dd_decay_gated_phase1_fwd_kernel[(N * H,)](
        k, d_out, g_osgm, g_gdn, cu_seqlens,
        initial_d, final_d,
        eta, d_min or 0.0, d_max or 0.0,
        k.stride(0), k.stride(1), k.stride(2),
        g_osgm.stride(0), g_osgm.stride(1), g_osgm.stride(2),
        T, H, K, BK,
        USE_DENOMINATOR=use_denominator,
        USE_PROJECTION=(d_min is not None and d_max is not None),
        IS_VARLEN=(cu_seqlens is not None),
        USE_INITIAL_STATE=(initial_d is not None),
        STORE_FINAL_STATE=output_final_state,
        num_warps=1, num_stages=1
    )
    return d_out, final_d


def compute_osgm_dd_decay_gated_phase1_bwd(
    k: torch.Tensor, g_osgm: torch.Tensor, g_gdn: torch.Tensor,
    d_out: torch.Tensor, dd_in: torch.Tensor,
    eta: float, use_denominator: bool, d_min: float, d_max: float,
    cu_seqlens: torch.Tensor = None, dd_final: torch.Tensor = None,
    output_initial_state_gradient: bool = False
):
    """Gate-aware OSGM dd-decay phase1 backward.

    Returns:
        dk_out:     [B, T, H, K]
        dd_initial: [N, H, K] or None
        dg_osgm:    [B, T, H]  (fp32)
        dg_gdn:     [B, T, H]  (fp32)  — NEW: gradient into the GDN state gate
    """
    B, T, H, K = k.shape
    BK = triton.next_power_of_2(K)
    N = len(cu_seqlens) - 1 if cu_seqlens is not None else B

    dk_out = torch.empty_like(k)
    dd_initial = torch.empty(N, H, K, device=k.device, dtype=k.dtype) if output_initial_state_gradient else None
    dg_osgm = torch.zeros(B, T, H, device=k.device, dtype=torch.float32)
    dg_gdn = torch.zeros(B, T, H, device=k.device, dtype=torch.float32)

    osgm_dd_decay_gated_phase1_bwd_kernel[(N * H,)](
        k, d_out, g_osgm, g_gdn, dd_in, dk_out, dg_osgm, dg_gdn,
        cu_seqlens, dd_final, dd_initial,
        eta, d_min or 0.0, d_max or 0.0,
        k.stride(0), k.stride(1), k.stride(2),
        g_osgm.stride(0), g_osgm.stride(1), g_osgm.stride(2),
        T, H, K, BK,
        USE_DENOMINATOR=use_denominator,
        USE_PROJECTION=(d_min is not None and d_max is not None),
        IS_VARLEN=(cu_seqlens is not None),
        USE_FINAL_STATE_GRADIENT=(dd_final is not None),
        STORE_INITIAL_STATE_GRADIENT=output_initial_state_gradient,
        num_warps=1, num_stages=1
    )
    return dk_out, dd_initial, dg_osgm, dg_gdn
