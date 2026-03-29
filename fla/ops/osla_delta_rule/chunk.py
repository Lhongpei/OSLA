# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

"""
Chunk-mode OSLA Delta Rule.

Forward: chunk-parallel (PyTorch ops, ~C× less sequential work than fused_recurrent).
Backward: fused_recurrent (exact gradients through preconditioner).

For cu_seqlens (varlen), falls back to fused_recurrent for both forward and backward.
"""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from fla.modules.l2norm import l2norm_bwd, l2norm_fwd
from fla.ops.osla_delta_rule.fused_recurrent import (
    fused_recurrent_delta_rule_bwd,
    fused_recurrent_delta_rule_fwd,
)
from fla.utils import input_guard

CHUNK_SIZE = 64


def _chunk_osla_fwd(
    q: torch.Tensor,     # [B, T, H, K]
    k: torch.Tensor,     # [B, T, H, K]
    v: torch.Tensor,     # [B, T, H, V]
    beta: torch.Tensor,  # [B, T, H]
    scale: float,
    initial_state: torch.Tensor,   # [B, H, K, V] or None
    initial_scale: torch.Tensor,   # [B, H, K]
    output_final_state: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Chunk-parallel OSLA forward (padded batch, no cu_seqlens)."""
    B, T, H, K = q.shape
    V = v.shape[-1]
    C = CHUNK_SIZE

    # Pad T to multiple of C
    pad = (C - T % C) % C
    if pad > 0:
        q = F.pad(q, (0, 0, 0, 0, 0, pad))
        k = F.pad(k, (0, 0, 0, 0, 0, pad))
        v = F.pad(v, (0, 0, 0, 0, 0, pad))
        beta = F.pad(beta, (0, 0, 0, pad))
    T_padded = T + pad
    NT = T_padded // C

    # Precompute D_t = initial_scale + cumsum(k^2)
    k_sq = k * k
    D = torch.cumsum(k_sq, dim=1) + initial_scale[:, None, :, :]
    p = k / (D + 1.0)  # write key

    # Reshape into chunks
    q_c = (q * scale).view(B, NT, C, H, K)
    k_c = k.view(B, NT, C, H, K)
    p_c = p.view(B, NT, C, H, K)
    v_c = v.view(B, NT, C, H, V)
    beta_c = beta.view(B, NT, C, H)

    h = initial_state.float() if initial_state is not None else q.new_zeros(B, H, K, V, dtype=torch.float32)
    causal_mask = torch.tril(torch.ones(C, C, device=q.device, dtype=torch.float32))

    o_chunks = []
    for i in range(NT):
        Q = q_c[:, i].float()        # [B, C, H, K]
        K_r = k_c[:, i].float()      # [B, C, H, K]
        P = p_c[:, i].float()        # [B, C, H, K]
        V_ch = v_c[:, i].float()     # [B, C, H, V]
        Beta = beta_c[:, i].float()  # [B, C, H]

        # A[b,H,t,s] = -beta_t * (p_s . k_t) for t > s  (negative, matching WY convention)
        A = -torch.einsum('bsHd,btHd->bHts', P, K_r) * Beta.permute(0, 2, 1).unsqueeze(3)
        A = torch.tril(A, diagonal=-1)

        # Solve (I - A) for u and w
        I_minus_A = torch.eye(C, device=A.device, dtype=A.dtype) - A

        beta_v = (V_ch * Beta.unsqueeze(-1)).permute(0, 2, 1, 3)  # [B, H, C, V]
        beta_k = (K_r * Beta.unsqueeze(-1)).permute(0, 2, 1, 3)   # [B, H, C, K] — READ key for inter-chunk correction

        u = torch.linalg.solve_triangular(I_minus_A, beta_v, upper=False, unitriangular=True)
        w = torch.linalg.solve_triangular(I_minus_A, beta_k, upper=False, unitriangular=True)

        # Inter-chunk correction
        v_new = u - torch.einsum('bHcK,bHKV->bHcV', w, h)

        # Output
        o_inter = torch.einsum('bcHK,bHKV->bHcV', Q, h)
        QP = torch.einsum('bcHK,bsHK->bHcs', Q, P)
        QP = QP * causal_mask
        o_intra = torch.matmul(QP, v_new)
        o_chunk = (o_inter + o_intra).permute(0, 2, 1, 3)
        o_chunks.append(o_chunk)

        # State update
        h = h + torch.einsum('bHcK,bHcV->bHKV', P.permute(0, 2, 1, 3), v_new)

    o = torch.cat(o_chunks, dim=1).to(q.dtype)
    if pad > 0:
        o = o[:, :T]

    final_state = h if output_final_state else None
    D_final = D[:, T - 1]  # [B, H, K]
    final_scale = D_final if output_final_state else None

    return o, final_state, final_scale


class ChunkOSLAFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
        scale: float,
        initial_state: torch.Tensor,
        initial_scale: torch.Tensor,
        output_final_state: bool,
        use_qk_l2norm_in_kernel: bool = False,
        cu_seqlens: Optional[torch.LongTensor] = None,
    ):
        if use_qk_l2norm_in_kernel:
            q, q_rstd = l2norm_fwd(q)
            k, k_rstd = l2norm_fwd(k)
        else:
            q_rstd, k_rstd = None, None

        if cu_seqlens is not None:
            # Varlen: fall back to fused_recurrent for forward too
            o, u, final_state, final_scale = fused_recurrent_delta_rule_fwd(
                q=q, k=k, v=v, beta=beta, scale=scale,
                initial_state=initial_state, initial_scale=initial_scale,
                output_final_state=output_final_state, cu_seqlens=cu_seqlens,
            )
        else:
            o, final_state, final_scale = _chunk_osla_fwd(
                q=q, k=k, v=v, beta=beta, scale=scale,
                initial_state=initial_state, initial_scale=initial_scale,
                output_final_state=output_final_state,
            )
            # Compute u (innovation) needed by fused_recurrent backward
            _, u, _, _ = fused_recurrent_delta_rule_fwd(
                q=q, k=k, v=v, beta=beta, scale=scale,
                initial_state=initial_state, initial_scale=initial_scale,
                output_final_state=False, cu_seqlens=None,
            )

        ctx.save_for_backward(q, q_rstd, k, k_rstd, u, beta, initial_state, initial_scale)
        ctx.scale = scale
        ctx.cu_seqlens = cu_seqlens
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        return o, final_state, final_scale

    @staticmethod
    @input_guard
    def backward(ctx, do, dht, d_final_scale):
        q, q_rstd, k, k_rstd, u, beta, initial_state, initial_scale = ctx.saved_tensors

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
def chunk_osla_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor = None,
    scale: float = None,
    initial_state: torch.Tensor = None,
    initial_scale: torch.Tensor = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if cu_seqlens is not None:
        if q.shape[0] != 1:
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
