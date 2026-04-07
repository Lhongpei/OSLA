# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

"""
Chunk-mode OSLA Delta Rule.

Forward: Triton building blocks (chunk-parallel).
Backward: fused_recurrent (memory-efficient, exact gradients through preconditioner).

u (innovation) is recovered from chunk forward's v_new via u = v_new / beta,
avoiding an extra fused_recurrent forward pass.
"""

from typing import Optional, Tuple

import torch

from fla.modules.l2norm import l2norm_bwd, l2norm_fwd
from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_h
from fla.ops.common.chunk_o import chunk_fwd_o
from fla.ops.delta_rule.wy_fast import recompute_w_u_fwd
from fla.ops.os_delta_rule.chunk_pkt import chunk_scaled_dot_pkt_fwd
from fla.ops.os_delta_rule.fused_recurrent import fused_recurrent_delta_rule_bwd
from fla.ops.utils.solve_tril import solve_tril
from fla.utils import input_guard

BT = 64


def _segment_cumsum(x, cu_seqlens, dim=1):
    if cu_seqlens is None:
        return torch.cumsum(x, dim=dim)
    result = torch.zeros_like(x)
    for i in range(len(cu_seqlens) - 1):
        s, e = cu_seqlens[i].item(), cu_seqlens[i + 1].item()
        result[:, s:e] = torch.cumsum(x[:, s:e], dim=dim)
    return result


def chunk_osla_fwd(q, k, v, beta, scale, initial_state, initial_scale, output_final_state, cu_seqlens=None):
    B, T, H, K = k.shape
    N = B if cu_seqlens is None else len(cu_seqlens) - 1

    # Precompute D, p
    D = _segment_cumsum(k * k, cu_seqlens, dim=1) + initial_scale[:N, None, :, :].to(k.dtype)
    p = k / (D + 1.0)

    # OSLA A matrix (Triton kernel, supports cu_seqlens)
    A = chunk_scaled_dot_pkt_fwd(k=k, p=p, beta=beta, cu_seqlens=cu_seqlens, chunk_size=BT)
    A = solve_tril(A=A, cu_seqlens=cu_seqlens, output_dtype=k.dtype)

    # WY transform: w uses READ key k, u uses v
    w, u_wy = recompute_w_u_fwd(k=k, v=v, beta=beta, A=A, cu_seqlens=cu_seqlens)

    # Inter-chunk: uses WRITE key p
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=p, w=w, u=u_wy, g=None,
        initial_state=initial_state, output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
    )

    # Output: uses WRITE key p
    o = chunk_fwd_o(q=q, k=p, v=v_new, h=h, g=None, scale=scale, cu_seqlens=cu_seqlens)

    # Recover u (raw innovation) from v_new: v_new = beta * u → u = v_new / beta
    u = v_new / beta.unsqueeze(-1).clamp(min=1e-6)

    # Final scale
    if output_final_state:
        if cu_seqlens is None:
            final_scale = D[:, -1]
        else:
            final_scale = torch.stack([D[0, cu_seqlens[i + 1] - 1] for i in range(N)])
    else:
        final_scale = None

    return o, u, final_state, final_scale


class ChunkOSLAFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    def forward(ctx, q, k, v, beta, scale, initial_state, initial_scale,
                output_final_state, use_qk_l2norm_in_kernel=False, cu_seqlens=None):
        if use_qk_l2norm_in_kernel:
            q, q_rstd = l2norm_fwd(q)
            k, k_rstd = l2norm_fwd(k)
        else:
            q_rstd, k_rstd = None, None

        o, u, final_state, final_scale = chunk_osla_fwd(
            q=q, k=k, v=v, beta=beta, scale=scale,
            initial_state=initial_state, initial_scale=initial_scale,
            output_final_state=output_final_state, cu_seqlens=cu_seqlens,
        )

        ctx.save_for_backward(q, q_rstd, k, k_rstd, u, beta, initial_state, initial_scale)
        ctx.scale = scale
        ctx.cu_seqlens = cu_seqlens
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        return o.to(q.dtype), final_state, final_scale

    @staticmethod
    @input_guard
    def backward(ctx, do, dht, d_final_scale):
        q, q_rstd, k, k_rstd, u, beta, initial_state, initial_scale = ctx.saved_tensors

        # Use fused_recurrent backward (memory-efficient, exact gradients)
        dq, dk, dv, db, dh0, d_scale_0 = fused_recurrent_delta_rule_bwd(
            q=q, k=k, v=u, beta=beta, dht=dht, do=do,
            scale=ctx.scale, initial_state=initial_state,
            initial_scale=initial_scale, cu_seqlens=ctx.cu_seqlens,
        )

        if ctx.use_qk_l2norm_in_kernel:
            dq = l2norm_bwd(q, q_rstd, dq)
            dk = l2norm_bwd(k, k_rstd, dk)

        return dq.to(q), dk.to(k), dv.to(u), db.to(beta), None, dh0, d_scale_0, None, None, None


@torch.compiler.disable
def chunk_os_delta_rule(q, k, v, beta=None, scale=None, initial_state=None,
                          initial_scale=None, output_final_state=False,
                          use_qk_l2norm_in_kernel=False, cu_seqlens=None):
    if cu_seqlens is not None and q.shape[0] != 1:
        raise ValueError(f"Batch size must be 1 with cu_seqlens, got {q.shape[0]}")
    if scale is None:
        scale = k.shape[-1] ** -0.5
    if beta is None:
        beta = torch.ones_like(q[..., 0])
    if initial_scale is None:
        B, H, K = q.shape[0], q.shape[2], q.shape[3]
        N = B if cu_seqlens is None else len(cu_seqlens) - 1
        initial_scale = q.new_zeros(N, H, K, dtype=torch.float32)

    o, final_state, final_scale = ChunkOSLAFunction.apply(
        q, k, v, beta, scale,
        initial_state, initial_scale,
        output_final_state, use_qk_l2norm_in_kernel, cu_seqlens,
    )
    if output_final_state:
        return o, (final_state, final_scale)
    return o, None
