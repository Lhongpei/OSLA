# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

"""
Triton kernel for computing beta * P @ K^T (asymmetric dot product for OSLA).

Identical to chunk_scaled_dot_kkt_fwd but loads from two different key tensors:
- p (write key, left factor)
- k (read key, right factor)

A[t,s] = beta_t * sum_d(k_t[d] * p_s[d])  for t > s within each chunk.
"""

from typing import Optional

import torch
import triton
import triton.language as tl

from fla.ops.utils import prepare_chunk_indices


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BK': BK}, num_warps=num_warps, num_stages=num_stages)
        for BK in [32, 64, 128]
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=['H', 'K', 'BT', 'IS_VARLEN'],
)
@triton.jit(do_not_specialize=['T'])
def chunk_scaled_dot_pkt_fwd_kernel(
    k,      # read key [B, T, H, K]
    p,      # write key [B, T, H, K]
    beta,   # [B, T, H]
    A,      # output [B, T, H, BT]
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T
    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T

    p_beta = tl.make_block_ptr(beta + bos*H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    b_beta = tl.load(p_beta, boundary_check=(0,))

    # A = K @ P^T (asymmetric: k for rows, p for columns)
    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(k + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_p = tl.make_block_ptr(p + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_p = tl.load(p_p, boundary_check=(0, 1))
        b_A += tl.dot(b_k, tl.trans(b_p))  # k @ p^T

    b_A *= b_beta[:, None]  # multiply by beta_t (row index)

    # Lower triangular mask (t > s)
    m_A = (o_t[:, None] > o_t[None, :]) & (m_t[:, None] & m_t)
    b_A = tl.where(m_A, b_A, 0)
    p_A = tl.make_block_ptr(A + (bos*H + i_h) * BT, (T, BT), (BT*H, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    tl.store(p_A, b_A.to(p_A.dtype.element_ty), boundary_check=(0, 1))


def chunk_scaled_dot_pkt_fwd(
    k: torch.Tensor,
    p: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: Optional[torch.LongTensor] = None,
    chunk_size: int = 64,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    r"""
    Compute asymmetric beta * K @ P^T for OSLA.

    Args:
        k: Read key [B, T, H, K]
        p: Write key [B, T, H, K]
        beta: [B, T, H]
        cu_seqlens: Optional variable-length sequence boundaries
        chunk_size: Chunk size (default 64)
        output_dtype: Output dtype (default float32)

    Returns:
        A: [B, T, H, BT] where A[t, s] = beta_t * (k_t . p_s) for t > s within chunks
    """
    B, T, H, K = k.shape
    BT = chunk_size
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    A = torch.empty(B, T, H, BT, device=k.device, dtype=output_dtype)
    chunk_scaled_dot_pkt_fwd_kernel[(NT, B * H)](
        k=k,
        p=p,
        beta=beta,
        A=A,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        K=K,
        BT=BT,
    )
    return A
