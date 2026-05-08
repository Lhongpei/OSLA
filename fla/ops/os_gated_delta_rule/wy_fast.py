# -*- coding: utf-8 -*-
"""Gated WY backward for OS-GDN.

This is the d-aware counterpart of ``fla.ops.gated_delta_rule.wy_fast``:
the read key used by the WY matrix is ``k`` while the state key is
``kw = k * d``.  It replaces the pure-PyTorch backward replay in
``os_gated_delta_rule.chunk``.
"""

import torch
import triton
import triton.language as tl

from fla.ops.utils import chunk_local_cumsum, prepare_chunk_indices
from fla.ops.utils.op import exp
from fla.utils import check_shared_mem


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4]
        for num_stages in [2, 3, 4]
    ],
    key=['H', 'K', 'V', 'BT', 'BK', 'BV', 'IS_VARLEN'],
)
@triton.jit(do_not_specialize=['T'])
def prepare_os_gated_wy_repr_bwd_kernel(
    k,
    v,
    beta,
    g,
    A,
    d,
    dw,
    du,
    dk_read,
    dv,
    dbeta,
    dg,
    dkw_from_A,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
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

    p_b = tl.make_block_ptr(beta + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    p_db = tl.make_block_ptr(dbeta + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    p_g = tl.make_block_ptr(g + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    p_dg = tl.make_block_ptr(dg + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    p_A = tl.make_block_ptr(A + (bos * H + i_h) * BT, (BT, T), (1, H * BT), (0, i_t * BT), (BT, BT), (0, 1))

    offs_t = i_t * BT + tl.arange(0, BT)
    m_t = offs_t < T
    m_A = (offs_t[:, None] > offs_t[None, :]) & (m_t[:, None] & m_t[None, :])

    b_b = tl.load(p_b, boundary_check=(0,)).to(tl.float32)
    b_g = tl.load(p_g, boundary_check=(0,)).to(tl.float32)
    b_g_exp = exp(b_g)
    b_A = tl.load(p_A, boundary_check=(0, 1)).to(tl.float32)
    b_db = tl.zeros([BT], dtype=tl.float32)
    b_dg = tl.zeros([BT], dtype=tl.float32)
    b_dA = tl.zeros([BT, BT], dtype=tl.float32)

    # Direct u = A @ (v * beta).
    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(v + (bos * H + i_h) * V, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_dv = tl.make_block_ptr(dv + (bos * H + i_h) * V, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_du = tl.make_block_ptr(du + (bos * H + i_h) * V, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))

        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_du = tl.load(p_du, boundary_check=(0, 1))
        b_vb = (b_v * b_b[:, None]).to(b_v.dtype)
        b_dA += tl.dot(b_du, tl.trans(b_vb), allow_tf32=False)
        b_dvb = tl.dot(b_A.to(b_du.dtype), b_du, allow_tf32=False)
        b_dv = b_dvb * b_b[:, None]
        b_db += tl.sum(b_dvb * b_v, axis=1)
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))

    # Direct w = A @ (k * beta * exp(g)).
    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_dk = tl.make_block_ptr(dk_read + (bos * H + i_h) * K, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_dw = tl.make_block_ptr(dw + (bos * H + i_h) * K, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))

        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_dw = tl.load(p_dw, boundary_check=(0, 1))
        b_kbg = b_k * (b_b * b_g_exp)[:, None]
        b_dA += tl.dot(b_dw, tl.trans(b_kbg).to(b_dw.dtype), allow_tf32=False)
        b_dkbg = tl.dot(b_A.to(b_dw.dtype), b_dw, allow_tf32=False)
        b_dk = b_dkbg * (b_b * b_g_exp)[:, None]
        b_db += tl.sum(b_dkbg * b_k * b_g_exp[:, None], axis=1)
        b_dg += tl.sum(b_dkbg * b_kbg, axis=1)
        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))

    # A = (I + X)^-1, so dX = -A^T dA A^T on the strict lower triangle.
    b_dA = tl.where(m_A, b_dA, 0.0)
    b_dA = tl.dot(b_dA.to(b_A.dtype), b_A)
    b_dA = tl.dot(b_A, b_dA.to(b_A.dtype))
    b_dA *= exp(b_g[:, None] - b_g[None, :])
    b_dA = tl.where(m_A, -b_dA, 0.0).to(k.dtype.element_ty)

    b_A_dot = tl.zeros([BT, BT], dtype=tl.float32)
    tl.debug_barrier()
    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_d = tl.make_block_ptr(d + (bos * H + i_h) * K, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_dk = tl.make_block_ptr(dk_read + (bos * H + i_h) * K, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_dkw = tl.make_block_ptr(dkw_from_A + (bos * H + i_h) * K, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))

        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_d = tl.load(p_d, boundary_check=(0, 1))
        b_dk = tl.load(p_dk, boundary_check=(0, 1))
        b_kw = b_k * b_d
        b_kb = b_k * b_b[:, None]

        b_A_dot += tl.dot(b_k, tl.trans(b_kw).to(b_k.dtype), allow_tf32=False)
        b_dkw = tl.dot(tl.trans(b_dA), b_kb.to(b_dA.dtype), allow_tf32=False)
        b_dk_left_pre = tl.dot(b_dA, b_kw.to(b_dA.dtype), allow_tf32=False)

        b_db += tl.sum(b_dk_left_pre * b_k, axis=1)
        b_dk += b_dk_left_pre * b_b[:, None]

        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dkw, b_dkw.to(p_dkw.dtype.element_ty), boundary_check=(0, 1))

    b_A_dot *= b_b[:, None]
    b_AdA = b_dA * b_A_dot
    b_dg += tl.sum(b_AdA, axis=1) - tl.sum(b_AdA, axis=0)
    tl.store(p_db, b_db.to(p_db.dtype.element_ty), boundary_check=(0,))
    tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0,))


def prepare_os_gated_wy_repr_bwd(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    A: torch.Tensor,
    d: torch.Tensor,
    dw: torch.Tensor,
    du: torch.Tensor,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *k.shape, v.shape[-1]
    BT = A.shape[-1]
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    const_tiling = 64 if check_shared_mem() else 32
    BK = min(max(triton.next_power_of_2(K), 16), const_tiling)
    BV = min(max(triton.next_power_of_2(V), 16), const_tiling)

    dk_read = torch.empty_like(k)
    dv = torch.empty_like(v)
    db = torch.empty_like(beta)
    dg_cum = torch.empty_like(g)
    dkw_from_A = torch.empty_like(k)

    prepare_os_gated_wy_repr_bwd_kernel[(NT, B * H)](
        k, v, beta, g, A, d, dw, du,
        dk_read, dv, db, dg_cum, dkw_from_A,
        cu_seqlens, chunk_indices, T,
        H, K, V, BT, BK, BV,
    )
    return dk_read, dv, db, dg_cum, dkw_from_A


def cumsum_g_cum_bwd(dg_cum: torch.Tensor, cu_seqlens: torch.LongTensor | None = None):
    return chunk_local_cumsum(dg_cum, chunk_size=64, reverse=True, cu_seqlens=cu_seqlens)
