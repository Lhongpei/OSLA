# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import triton
import triton.language as tl

from fla.ops.os_kda.chunk_intra_token_parallel import chunk_kda_fwd_intra_token_parallel
from fla.ops.os_kda.wy_fast_os import recompute_w_u_fwd
from fla.ops.utils import prepare_chunk_indices
from fla.ops.utils.op import exp2, gather
from fla.utils import IS_GATHER_SUPPORTED, IS_TF32_SUPPORTED, autotune_cache_kwargs

if IS_TF32_SUPPORTED:
    SOLVE_TRIL_DOT_PRECISION = tl.constexpr('tf32')
else:
    SOLVE_TRIL_DOT_PRECISION = tl.constexpr('ieee')

################################################################################
# Fused inter + solve_tril kernel: compute off-diagonal Akk and solve in one pass
################################################################################


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BK': BK}, num_warps=num_warps)
        for BK in [32, 64]
        for num_warps in [1, 2, 4]
    ],
    key=["H", "K", "BC"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_kda_fwd_kernel_inter_solve_fused(
    q, k, d, g, beta, Aqk, Akkd, Akk, scale, cu_seqlens, chunk_indices, T,
    H: tl.constexpr, K: tl.constexpr, BT: tl.constexpr, BC: tl.constexpr, BK: tl.constexpr, IS_VARLEN: tl.constexpr, USE_SAFE_GATE: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    if i_t * BT >= T:
        return

    i_tc0 = i_t * BT
    i_tc1 = i_t * BT + BC
    i_tc2 = i_t * BT + 2 * BC
    i_tc3 = i_t * BT + 3 * BC

    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    d += (bos * H + i_h) * K
    g += (bos * H + i_h) * K
    Aqk += (bos * H + i_h) * BT
    Akk += (bos * H + i_h) * BT
    Akkd += (bos * H + i_h) * BC

    o_i = tl.arange(0, BC)
    m_tc1 = (i_tc1 + o_i) < T
    m_tc2 = (i_tc2 + o_i) < T
    m_tc3 = (i_tc3 + o_i) < T

    b_Aqk10 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk10 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Aqk20 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk20 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Aqk21 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk21 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Aqk30 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk30 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Aqk31 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk31 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Aqk32 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk32 = tl.zeros([BC, BC], dtype=tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K

        p_k0 = tl.make_block_ptr(k, (T, K), (H*K, 1), (i_tc0, i_k * BK), (BC, BK), (1, 0))
        p_d0 = tl.make_block_ptr(d, (T, K), (H*K, 1), (i_tc0, i_k * BK), (BC, BK), (1, 0))
        p_g0 = tl.make_block_ptr(g, (T, K), (H*K, 1), (i_tc0, i_k * BK), (BC, BK), (1, 0))
        b_k0 = tl.load(p_k0, boundary_check=(0, 1)).to(tl.float32)
        b_d0 = tl.load(p_d0, boundary_check=(0, 1)).to(tl.float32)
        b_g0 = tl.load(p_g0, boundary_check=(0, 1)).to(tl.float32)
        
        b_kd0 = b_k0 * b_d0

        if i_tc1 < T:
            p_q1 = tl.make_block_ptr(q, (T, K), (H*K, 1), (i_tc1, i_k * BK), (BC, BK), (1, 0))
            p_k1 = tl.make_block_ptr(k, (T, K), (H*K, 1), (i_tc1, i_k * BK), (BC, BK), (1, 0))
            p_d1 = tl.make_block_ptr(d, (T, K), (H*K, 1), (i_tc1, i_k * BK), (BC, BK), (1, 0))
            p_g1 = tl.make_block_ptr(g, (T, K), (H*K, 1), (i_tc1, i_k * BK), (BC, BK), (1, 0))
            
            b_q1 = tl.load(p_q1, boundary_check=(0, 1)).to(tl.float32)
            b_k1 = tl.load(p_k1, boundary_check=(0, 1)).to(tl.float32)
            b_d1 = tl.load(p_d1, boundary_check=(0, 1)).to(tl.float32)
            b_g1 = tl.load(p_g1, boundary_check=(0, 1)).to(tl.float32)
            
            b_gn1 = tl.load(g + i_tc1 * H*K + o_k, mask=m_k, other=0).to(tl.float32)
            b_gqn = tl.where(m_tc1[:, None], exp2(b_g1 - b_gn1[None, :]), 0)
            
            b_kgt = tl.trans(b_kd0 * exp2(b_gn1[None, :] - b_g0))
            
            b_Aqk10 += tl.dot(b_q1 * b_gqn, b_kgt)
            b_Akk10 += tl.dot(b_k1 * b_gqn, b_kgt)

            if i_tc2 < T:
                p_q2 = tl.make_block_ptr(q, (T, K), (H*K, 1), (i_tc2, i_k * BK), (BC, BK), (1, 0))
                p_k2 = tl.make_block_ptr(k, (T, K), (H*K, 1), (i_tc2, i_k * BK), (BC, BK), (1, 0))
                p_d2 = tl.make_block_ptr(d, (T, K), (H*K, 1), (i_tc2, i_k * BK), (BC, BK), (1, 0))
                p_g2 = tl.make_block_ptr(g, (T, K), (H*K, 1), (i_tc2, i_k * BK), (BC, BK), (1, 0))
                
                b_q2 = tl.load(p_q2, boundary_check=(0, 1)).to(tl.float32)
                b_k2 = tl.load(p_k2, boundary_check=(0, 1)).to(tl.float32)
                b_d2 = tl.load(p_d2, boundary_check=(0, 1)).to(tl.float32)
                b_g2 = tl.load(p_g2, boundary_check=(0, 1)).to(tl.float32)
                
                b_gn2 = tl.load(g + i_tc2 * H*K + o_k, mask=m_k, other=0).to(tl.float32)
                b_gqn2 = tl.where(m_tc2[:, None], exp2(b_g2 - b_gn2[None, :]), 0)
                b_qg2 = b_q2 * b_gqn2
                b_kg2 = b_k2 * b_gqn2
                
                b_kgt = tl.trans(b_kd0 * exp2(b_gn2[None, :] - b_g0))
                b_Aqk20 += tl.dot(b_qg2, b_kgt)
                b_Akk20 += tl.dot(b_kg2, b_kgt)
                
                b_kgt = tl.trans(b_k1 * b_d1 * exp2(b_gn2[None, :] - b_g1))
                b_Aqk21 += tl.dot(b_qg2, b_kgt)
                b_Akk21 += tl.dot(b_kg2, b_kgt)

                if i_tc3 < T:
                    p_q3 = tl.make_block_ptr(q, (T, K), (H*K, 1), (i_tc3, i_k * BK), (BC, BK), (1, 0))
                    p_k3 = tl.make_block_ptr(k, (T, K), (H*K, 1), (i_tc3, i_k * BK), (BC, BK), (1, 0))
                    p_d3 = tl.make_block_ptr(d, (T, K), (H*K, 1), (i_tc3, i_k * BK), (BC, BK), (1, 0))
                    p_g3 = tl.make_block_ptr(g, (T, K), (H*K, 1), (i_tc3, i_k * BK), (BC, BK), (1, 0))
                    
                    b_q3 = tl.load(p_q3, boundary_check=(0, 1)).to(tl.float32)
                    b_k3 = tl.load(p_k3, boundary_check=(0, 1)).to(tl.float32)
                    b_d3 = tl.load(p_d3, boundary_check=(0, 1)).to(tl.float32)
                    b_g3 = tl.load(p_g3, boundary_check=(0, 1)).to(tl.float32)
                    
                    b_gn3 = tl.load(g + i_tc3 * H*K + o_k, mask=m_k, other=0).to(tl.float32)
                    b_gqn3 = tl.where(m_tc3[:, None], exp2(b_g3 - b_gn3[None, :]), 0)
                    b_qg3 = b_q3 * b_gqn3
                    b_kg3 = b_k3 * b_gqn3
                    
                    b_kgt = tl.trans(b_kd0 * exp2(b_gn3[None, :] - b_g0))
                    b_Aqk30 += tl.dot(b_qg3, b_kgt)
                    b_Akk30 += tl.dot(b_kg3, b_kgt)
                    
                    b_kgt = tl.trans(b_k1 * b_d1 * exp2(b_gn3[None, :] - b_g1))
                    b_Aqk31 += tl.dot(b_qg3, b_kgt)
                    b_Akk31 += tl.dot(b_kg3, b_kgt)
                    
                    b_kgt = tl.trans(b_k2 * b_d2 * exp2(b_gn3[None, :] - b_g2))
                    b_Aqk32 += tl.dot(b_qg3, b_kgt)
                    b_Akk32 += tl.dot(b_kg3, b_kgt)

    if i_tc1 < T:
        p_Aqk10 = tl.make_block_ptr(Aqk, (T, BT), (H*BT, 1), (i_tc1, 0), (BC, BC), (1, 0))
        tl.store(p_Aqk10, (b_Aqk10 * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1))

        p_b1 = tl.make_block_ptr(beta + bos * H + i_h, (T,), (H,), (i_tc1,), (BC,), (0,))
        b_b1 = tl.load(p_b1, boundary_check=(0,)).to(tl.float32)
        b_Akk10 = b_Akk10 * b_b1[:, None]
    if i_tc2 < T:
        p_Aqk20 = tl.make_block_ptr(Aqk, (T, BT), (H*BT, 1), (i_tc2, 0), (BC, BC), (1, 0))
        p_Aqk21 = tl.make_block_ptr(Aqk, (T, BT), (H*BT, 1), (i_tc2, BC), (BC, BC), (1, 0))
        tl.store(p_Aqk20, (b_Aqk20 * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_Aqk21, (b_Aqk21 * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1))

        p_b2 = tl.make_block_ptr(beta + bos * H + i_h, (T,), (H,), (i_tc2,), (BC,), (0,))
        b_b2 = tl.load(p_b2, boundary_check=(0,)).to(tl.float32)
        b_Akk20 = b_Akk20 * b_b2[:, None]
        b_Akk21 = b_Akk21 * b_b2[:, None]
    if i_tc3 < T:
        p_Aqk30 = tl.make_block_ptr(Aqk, (T, BT), (H*BT, 1), (i_tc3, 0), (BC, BC), (1, 0))
        p_Aqk31 = tl.make_block_ptr(Aqk, (T, BT), (H*BT, 1), (i_tc3, BC), (BC, BC), (1, 0))
        p_Aqk32 = tl.make_block_ptr(Aqk, (T, BT), (H*BT, 1), (i_tc3, 2*BC), (BC, BC), (1, 0))
        tl.store(p_Aqk30, (b_Aqk30 * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_Aqk31, (b_Aqk31 * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_Aqk32, (b_Aqk32 * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1))

        p_b3 = tl.make_block_ptr(beta + bos * H + i_h, (T,), (H,), (i_tc3,), (BC,), (0,))
        b_b3 = tl.load(p_b3, boundary_check=(0,)).to(tl.float32)
        b_Akk30 = b_Akk30 * b_b3[:, None]
        b_Akk31 = b_Akk31 * b_b3[:, None]
        b_Akk32 = b_Akk32 * b_b3[:, None]

    p_Akk00 = tl.make_block_ptr(Akkd, (T, BC), (H*BC, 1), (i_tc0, 0), (BC, BC), (1, 0))
    p_Akk11 = tl.make_block_ptr(Akkd, (T, BC), (H*BC, 1), (i_tc1, 0), (BC, BC), (1, 0))
    p_Akk22 = tl.make_block_ptr(Akkd, (T, BC), (H*BC, 1), (i_tc2, 0), (BC, BC), (1, 0))
    p_Akk33 = tl.make_block_ptr(Akkd, (T, BC), (H*BC, 1), (i_tc3, 0), (BC, BC), (1, 0))
    b_Ai00 = tl.load(p_Akk00, boundary_check=(0, 1)).to(tl.float32)
    b_Ai11 = tl.load(p_Akk11, boundary_check=(0, 1)).to(tl.float32)
    b_Ai22 = tl.load(p_Akk22, boundary_check=(0, 1)).to(tl.float32)
    b_Ai33 = tl.load(p_Akk33, boundary_check=(0, 1)).to(tl.float32)


    if not USE_SAFE_GATE:
        m_A = o_i[:, None] > o_i[None, :]
        m_I = o_i[:, None] == o_i[None, :]

        b_Ai00 = -tl.where(m_A, b_Ai00, 0)
        b_Ai11 = -tl.where(m_A, b_Ai11, 0)
        b_Ai22 = -tl.where(m_A, b_Ai22, 0)
        b_Ai33 = -tl.where(m_A, b_Ai33, 0)

        for i in range(2, min(BC, T - i_tc0)):
            b_a00 = -tl.load(Akkd + (i_tc0 + i) * H*BC + o_i)
            b_a00 = tl.where(o_i < i, b_a00, 0.)
            b_a00 += tl.sum(b_a00[:, None] * b_Ai00, 0)
            b_Ai00 = tl.where((o_i == i)[:, None], b_a00, b_Ai00)
        for i in range(BC + 2, min(2*BC, T - i_tc0)):
            b_a11 = -tl.load(Akkd + (i_tc0 + i) * H*BC + o_i)
            b_a11 = tl.where(o_i < i - BC, b_a11, 0.)
            b_a11 += tl.sum(b_a11[:, None] * b_Ai11, 0)
            b_Ai11 = tl.where((o_i == i - BC)[:, None], b_a11, b_Ai11)
        for i in range(2*BC + 2, min(3*BC, T - i_tc0)):
            b_a22 = -tl.load(Akkd + (i_tc0 + i) * H*BC + o_i)
            b_a22 = tl.where(o_i < i - 2*BC, b_a22, 0.)
            b_a22 += tl.sum(b_a22[:, None] * b_Ai22, 0)
            b_Ai22 = tl.where((o_i == i - 2*BC)[:, None], b_a22, b_Ai22)
        for i in range(3*BC + 2, min(4*BC, T - i_tc0)):
            b_a33 = -tl.load(Akkd + (i_tc0 + i) * H*BC + o_i)
            b_a33 = tl.where(o_i < i - 3*BC, b_a33, 0.)
            b_a33 += tl.sum(b_a33[:, None] * b_Ai33, 0)
            b_Ai33 = tl.where((o_i == i - 3*BC)[:, None], b_a33, b_Ai33)

        b_Ai00 += m_I
        b_Ai11 += m_I
        b_Ai22 += m_I
        b_Ai33 += m_I

    b_Ai10 = -tl.dot(
        tl.dot(b_Ai11, b_Akk10, input_precision=SOLVE_TRIL_DOT_PRECISION),
        b_Ai00,
        input_precision=SOLVE_TRIL_DOT_PRECISION
    )
    b_Ai21 = -tl.dot(
        tl.dot(b_Ai22, b_Akk21, input_precision=SOLVE_TRIL_DOT_PRECISION),
        b_Ai11,
        input_precision=SOLVE_TRIL_DOT_PRECISION
    )
    b_Ai32 = -tl.dot(
        tl.dot(b_Ai33, b_Akk32, input_precision=SOLVE_TRIL_DOT_PRECISION),
        b_Ai22,
        input_precision=SOLVE_TRIL_DOT_PRECISION
    )

    b_Ai20 = -tl.dot(
        b_Ai22,
        tl.dot(b_Akk20, b_Ai00, input_precision=SOLVE_TRIL_DOT_PRECISION) +
        tl.dot(b_Akk21, b_Ai10, input_precision=SOLVE_TRIL_DOT_PRECISION),
        input_precision=SOLVE_TRIL_DOT_PRECISION
    )
    b_Ai31 = -tl.dot(
        b_Ai33,
        tl.dot(b_Akk31, b_Ai11, input_precision=SOLVE_TRIL_DOT_PRECISION) +
        tl.dot(b_Akk32, b_Ai21, input_precision=SOLVE_TRIL_DOT_PRECISION),
        input_precision=SOLVE_TRIL_DOT_PRECISION
    )
    b_Ai30 = -tl.dot(
        b_Ai33,
        tl.dot(b_Akk30, b_Ai00, input_precision=SOLVE_TRIL_DOT_PRECISION) +
        tl.dot(b_Akk31, b_Ai10, input_precision=SOLVE_TRIL_DOT_PRECISION) +
        tl.dot(b_Akk32, b_Ai20, input_precision=SOLVE_TRIL_DOT_PRECISION),
        input_precision=SOLVE_TRIL_DOT_PRECISION
    )


    p_Akk00 = tl.make_block_ptr(Akk, (T, BT), (H*BT, 1), (i_tc0, 0), (BC, BC), (1, 0))
    p_Akk10 = tl.make_block_ptr(Akk, (T, BT), (H*BT, 1), (i_tc1, 0), (BC, BC), (1, 0))
    p_Akk11 = tl.make_block_ptr(Akk, (T, BT), (H*BT, 1), (i_tc1, BC), (BC, BC), (1, 0))
    p_Akk20 = tl.make_block_ptr(Akk, (T, BT), (H*BT, 1), (i_tc2, 0), (BC, BC), (1, 0))
    p_Akk21 = tl.make_block_ptr(Akk, (T, BT), (H*BT, 1), (i_tc2, BC), (BC, BC), (1, 0))
    p_Akk22 = tl.make_block_ptr(Akk, (T, BT), (H*BT, 1), (i_tc2, 2*BC), (BC, BC), (1, 0))
    p_Akk30 = tl.make_block_ptr(Akk, (T, BT), (H*BT, 1), (i_tc3, 0), (BC, BC), (1, 0))
    p_Akk31 = tl.make_block_ptr(Akk, (T, BT), (H*BT, 1), (i_tc3, BC), (BC, BC), (1, 0))
    p_Akk32 = tl.make_block_ptr(Akk, (T, BT), (H*BT, 1), (i_tc3, 2*BC), (BC, BC), (1, 0))
    p_Akk33 = tl.make_block_ptr(Akk, (T, BT), (H*BT, 1), (i_tc3, 3*BC), (BC, BC), (1, 0))

    tl.store(p_Akk00, b_Ai00.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk10, b_Ai10.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk11, b_Ai11.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk20, b_Ai20.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk21, b_Ai21.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk22, b_Ai22.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk30, b_Ai30.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk31, b_Ai31.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk32, b_Ai32.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk33, b_Ai33.to(Akk.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [1, 2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=['BK', 'NC', 'BT'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['B', 'T'])
def chunk_kda_bwd_kernel_intra(
    q, k, d, g, beta, dAqk, dAkk, dq, dk, dd, dg, db, cu_seqlens, chunk_indices, B, T,
    H: tl.constexpr, K: tl.constexpr, BT: tl.constexpr, BC: tl.constexpr, BK: tl.constexpr, NC: tl.constexpr, IS_VARLEN: tl.constexpr, SAFE_GATE: tl.constexpr, USE_GATHER: tl.constexpr,
):
    i_kc, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    i_k, i_i = i_kc // NC, i_kc % NC

    all = B * T
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
    else:
        bos, eos = i_b * T, i_b * T + T
    T = eos - bos

    i_ti = i_t * BT + i_i * BC
    if i_ti >= T: return

    o_k = i_k * BK + tl.arange(0, BK)
    m_k = o_k < K

    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    d += (bos * H + i_h) * K
    g += (bos * H + i_h) * K
    beta += bos * H + i_h

    dAqk += (bos * H + i_h) * BT
    dAkk += (bos * H + i_h) * BT
    dq += (bos * H + i_h) * K
    dk += (bos * H + i_h) * K
    dd += (bos * H + i_h) * K
    dg += (bos * H + i_h) * K
    db += (i_k * all + bos) * H + i_h

    p_g = tl.make_block_ptr(g, (T, K), (H*K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
    b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)

    p_b = tl.make_block_ptr(beta, (T,), (H,), (i_ti,), (BC,), (0,))
    b_b = tl.load(p_b, boundary_check=(0,))

    b_dq2 = tl.zeros([BC, BK], dtype=tl.float32)
    b_dk_left = tl.zeros([BC, BK], dtype=tl.float32)
    if i_i > 0:
        p_gn = g + i_ti * H*K + o_k
        b_gn = tl.load(p_gn, mask=m_k, other=0).to(tl.float32)[None, :]
        for i_j in range(0, i_i):
            p_k = tl.make_block_ptr(k, (T, K), (H*K, 1), (i_t * BT + i_j * BC, i_k * BK), (BC, BK), (1, 0))
            p_d = tl.make_block_ptr(d, (T, K), (H*K, 1), (i_t * BT + i_j * BC, i_k * BK), (BC, BK), (1, 0))
            p_gk = tl.make_block_ptr(g, (T, K), (H*K, 1), (i_t * BT + i_j * BC, i_k * BK), (BC, BK), (1, 0))
            p_dAqk = tl.make_block_ptr(dAqk, (T, BT), (H*BT, 1), (i_ti, i_j * BC), (BC, BC), (1, 0))
            p_dAkk = tl.make_block_ptr(dAkk, (T, BT), (H*BT, 1), (i_ti, i_j * BC), (BC, BC), (1, 0))
            
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_d = tl.load(p_d, boundary_check=(0, 1))
            b_gk = tl.load(p_gk, boundary_check=(0, 1))
            b_kg = b_k * b_d * exp2(b_gn - b_gk)
            
            b_dAqk = tl.load(p_dAqk, boundary_check=(0, 1))
            b_dAkk = tl.load(p_dAkk, boundary_check=(0, 1))
            
            b_dq2 += tl.dot(b_dAqk, b_kg)
            b_dk_left += tl.dot(b_dAkk, b_kg)

        b_gqn = exp2(b_g - b_gn)
        b_dq2 *= b_gqn
        b_dk_left *= b_gqn

    o_i = tl.arange(0, BC)
    m_dA = (i_ti + o_i) < T
    o_dA = (i_ti + o_i) * H*BT + i_i * BC
    p_kj = k + i_ti * H*K + o_k
    p_dj = d + i_ti * H*K + o_k
    p_gkj = g + i_ti * H*K + o_k

    if SAFE_GATE:
        if USE_GATHER:
            b_gn = gather(b_g, tl.full([1, BK], min(BC//2, T - i_ti - 1), dtype=tl.int16), axis=0)
        else:
            p_gn = g + (i_ti + min(BC // 2, T - i_ti - 1)) * H*K + o_k
            b_gn = tl.load(p_gn, mask=m_k, other=0)[None, :]

        p_k = tl.make_block_ptr(k, (T, K), (H*K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
        p_d = tl.make_block_ptr(d, (T, K), (H*K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_d = tl.load(p_d, boundary_check=(0, 1))
        b_kd = b_k * b_d

        p_dAqk = tl.make_block_ptr(dAqk, (T, BT), (H*BT, 1), (i_ti, i_i * BC), (BC, BC), (1, 0))
        p_dAkk = tl.make_block_ptr(dAkk, (T, BT), (H*BT, 1), (i_ti, i_i * BC), (BC, BC), (1, 0))
        b_dAqk_diag_qk = tl.load(p_dAqk, boundary_check=(0, 1)).to(tl.float32)
        b_dAkk_diag_qk = tl.load(p_dAkk, boundary_check=(0, 1)).to(tl.float32)

        m_i_diag_qk = (o_i[:, None] >= o_i[None, :]) & ((i_ti + o_i[:, None]) < T) & ((i_ti + o_i[None, :]) < T)
        m_j_diag_qk = (i_ti + o_i[:, None]) < T

        b_dAqk_diag_qk = tl.where(m_i_diag_qk, b_dAqk_diag_qk, 0.)
        b_dAkk_diag_qk = tl.where(m_i_diag_qk, b_dAkk_diag_qk, 0.)
        b_g_diag_qk = tl.where(m_j_diag_qk, b_g - b_gn, 0.)
        exp_b_g_diag_qk = tl.where(m_j_diag_qk, exp2(b_g_diag_qk), 0.)
        exp_neg_b_g_diag_qk = tl.where(m_j_diag_qk, exp2(-b_g_diag_qk), 0.)

        b_kd_exp_diag_qk = b_kd * exp_neg_b_g_diag_qk
        b_dq2 += tl.dot(b_dAqk_diag_qk, b_kd_exp_diag_qk) * exp_b_g_diag_qk
        b_dk_left += tl.dot(b_dAkk_diag_qk, b_kd_exp_diag_qk) * exp_b_g_diag_qk
    else:
        for j in range(0, min(BC, T - i_t * BT - i_i * BC)):
            b_dAqk = tl.load(dAqk + o_dA + j, mask=m_dA, other=0)
            b_dAkk = tl.load(dAkk + o_dA + j, mask=m_dA, other=0)
            b_kj = tl.load(p_kj, mask=m_k, other=0).to(tl.float32)
            b_dj = tl.load(p_dj, mask=m_k, other=0).to(tl.float32)
            b_gkj = tl.load(p_gkj, mask=m_k, other=0).to(tl.float32)
            m_i = o_i[:, None] >= j
            b_gqk = exp2(b_g - b_gkj[None, :])
            b_dq2 += tl.where(m_i, b_dAqk[:, None] * b_kj[None, :] * b_dj[None, :] * b_gqk, 0.)
            b_dk_left += tl.where(m_i, b_dAkk[:, None] * b_kj[None, :] * b_dj[None, :] * b_gqk, 0.)
            p_kj += H*K
            p_dj += H*K
            p_gkj += H*K

    p_k_out = tl.make_block_ptr(k, (T, K), (H*K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
    b_k_out = tl.load(p_k_out, boundary_check=(0, 1))

    b_db_intra = tl.sum(b_dk_left * b_k_out, 1) 
    b_dk_left *= b_b[:, None]

    p_q = tl.make_block_ptr(q, (T, K), (H*K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
    b_q = tl.load(p_q, boundary_check=(0, 1))
    
    p_dq = tl.make_block_ptr(dq, (T, K), (H*K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
    b_dg2 = b_q * b_dq2
    b_dq2 = b_dq2 + tl.load(p_dq, boundary_check=(0, 1))
    tl.store(p_dq, b_dq2.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
    tl.debug_barrier()

    b_dkt = tl.zeros([BC, BK], dtype=tl.float32)
    NC = min(NC, tl.cdiv(T - i_t * BT, BC))
    if i_i < NC - 1:
        p_gn = g + (min(i_ti + BC, T) - 1) * H*K + o_k
        b_gn = tl.load(p_gn, mask=m_k, other=0).to(tl.float32)[None, :]
        for i_j in range(i_i + 1, NC):
            p_q = tl.make_block_ptr(q, (T, K), (H*K, 1), (i_t*BT+i_j*BC, i_k*BK), (BC, BK), (1, 0))
            p_k = tl.make_block_ptr(k, (T, K), (H*K, 1), (i_t * BT + i_j * BC, i_k * BK), (BC, BK), (1, 0))
            p_gk = tl.make_block_ptr(g, (T, K), (H*K, 1), (i_t * BT + i_j * BC, i_k*BK), (BC, BK), (1, 0))
            p_b = tl.make_block_ptr(beta, (T,), (H,), (i_t * BT + i_j * BC,), (BC,), (0,))
            p_dAqk = tl.make_block_ptr(dAqk, (BT, T), (1, H*BT), (i_i * BC, i_t * BT + i_j * BC), (BC, BC), (0, 1))
            p_dAkk = tl.make_block_ptr(dAkk, (BT, T), (1, H*BT), (i_i * BC, i_t * BT + i_j * BC), (BC, BC), (0, 1))
            
            b_b_j = tl.load(p_b, boundary_check=(0,))
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_k_loaded = tl.load(p_k, boundary_check=(0, 1))
            b_kb = b_k_loaded * b_b_j[:, None] 
            
            b_gk = tl.load(p_gk, boundary_check=(0, 1)).to(tl.float32)
            b_dAqk = tl.load(p_dAqk, boundary_check=(0, 1))
            b_dAkk = tl.load(p_dAkk, boundary_check=(0, 1))

            o_j = i_t * BT + i_j * BC + o_i
            m_j = o_j < T
            
            b_gkn = exp2(b_gk - b_gn)
            b_qg = b_q * tl.where(m_j[:, None], b_gkn, 0)
            b_kbg = b_kb * tl.where(m_j[:, None], b_gkn, 0)
            
            b_dkt += tl.dot(b_dAqk, b_qg)
            b_dkt += tl.dot(b_dAkk, b_kbg) 
        b_dkt *= exp2(b_gn - b_g)

    if SAFE_GATE:
        p_dAqk_t = tl.make_block_ptr(dAqk, (BT, T), (1, H*BT), (i_i * BC, i_ti), (BC, BC), (0, 1))
        p_dAkk_t = tl.make_block_ptr(dAkk, (BT, T), (1, H*BT), (i_i * BC, i_ti), (BC, BC), (0, 1))
        b_dAqk_t = tl.load(p_dAqk_t, boundary_check=(0, 1)).to(tl.float32)
        b_dAkk_t = tl.load(p_dAkk_t, boundary_check=(0, 1)).to(tl.float32)
        
        m_j_diag = (o_i[:, None] <= o_i[None, :]) & ((i_ti + o_i[:, None]) < T) & ((i_ti + o_i[None, :]) < T)
        b_dAqk_t = tl.where(m_j_diag, b_dAqk_t, 0.)
        b_dAkk_t = tl.where(m_j_diag, b_dAkk_t, 0.)
        
        b_q_exp = b_q * exp2(b_g - b_gn)
        b_kb_exp = b_k_out * b_b[:, None] * exp2(b_g - b_gn)
        
        b_dkt += tl.dot(b_dAqk_t, b_q_exp) * exp2(b_gn - b_g)
        b_dkt += tl.dot(b_dAkk_t, b_kb_exp) * exp2(b_gn - b_g)
    else:
        for i in range(0, min(BC, T - i_ti)):
            b_dAqk_i = tl.load(dAqk + (i_ti + i)*H*BT + i_i*BC + o_i, mask=o_i <= i, other=0)
            b_dAkk_i = tl.load(dAkk + (i_ti + i)*H*BT + i_i*BC + o_i, mask=o_i <= i, other=0)
            
            b_qi = tl.load(q + (i_ti + i)*H*K + o_k, mask=m_k, other=0).to(tl.float32)
            b_kbi = tl.load(k + (i_ti + i)*H*K + o_k, mask=m_k, other=0).to(tl.float32) * tl.load(beta + i_ti + i)
            b_gi = tl.load(g + (i_ti + i)*H*K + o_k, mask=m_k, other=0).to(tl.float32)
            
            b_g_exp = exp2(b_gi[None, :] - b_g)
            b_dkt += tl.where(o_i[:, None] <= i, b_dAqk_i[:, None] * b_qi[None, :] * b_g_exp, 0.)
            b_dkt += tl.where(o_i[:, None] <= i, b_dAkk_i[:, None] * b_kbi[None, :] * b_g_exp, 0.)

    p_d_out = tl.make_block_ptr(d, (T, K), (H*K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
    p_dk = tl.make_block_ptr(dk, (T, K), (H*K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
    p_dd = tl.make_block_ptr(dd, (T, K), (H*K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
    p_dg = tl.make_block_ptr(dg, (T, K), (H*K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
    p_db = tl.make_block_ptr(db, (T,), (H,), (i_ti,), (BC,), (0,))

    b_d_out = tl.load(p_d_out, boundary_check=(0, 1))

    b_dkw_inter = tl.load(p_dk, boundary_check=(0, 1))
    b_dk_raw_inter = tl.load(p_dd, boundary_check=(0, 1))

    b_dkw_total = b_dkw_inter + b_dkt       
    b_dk_raw_total = b_dk_raw_inter + b_dk_left 

    b_dk_final = b_dk_raw_total + b_dkw_total * b_d_out
    b_dd_final = b_dkw_total * b_k_out

    b_dg2 += b_dk_left * b_k_out - b_dkt * (b_k_out * b_d_out) + tl.load(p_dg, boundary_check=(0, 1))
    b_db_total = tl.load(p_db, boundary_check=(0,)) + b_db_intra

    tl.store(p_dk, b_dk_final.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dd, b_dd_final.to(p_dd.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dg, b_dg2.to(g.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_db, b_db_total.to(beta.dtype.element_ty), boundary_check=(0,))
    
@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [1, 2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=["BT", "BC"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_kda_fwd_kernel_intra_sub_chunk(
    q, k, d, g, beta, Aqk, Akk, scale, cu_seqlens, chunk_indices, T,
    H: tl.constexpr, K: tl.constexpr, BT: tl.constexpr, BC: tl.constexpr, BK: tl.constexpr, IS_VARLEN: tl.constexpr, USE_GATHER: tl.constexpr,
):
    i_t, i_i, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    i_ti = i_t * BT + i_i * BC
    if i_ti >= T: return

    o_c = i_ti + tl.arange(0, BC)
    m_c = o_c < T

    q = q + (bos * H + i_h) * K
    k = k + (bos * H + i_h) * K
    d = d + (bos * H + i_h) * K
    g = g + (bos * H + i_h) * K
    beta = beta + bos * H + i_h
    Aqk = Aqk + (bos * H + i_h) * BT
    Akk = Akk + (bos * H + i_h) * BC

    p_q = tl.make_block_ptr(q, (T, K), (H*K, 1), (i_ti, 0), (BC, BK), (1, 0))
    p_k = tl.make_block_ptr(k, (T, K), (H*K, 1), (i_ti, 0), (BC, BK), (1, 0))
    p_d = tl.make_block_ptr(d, (T, K), (H*K, 1), (i_ti, 0), (BC, BK), (1, 0))
    p_g = tl.make_block_ptr(g, (T, K), (H*K, 1), (i_ti, 0), (BC, BK), (1, 0))
    p_beta = tl.make_block_ptr(beta, (T,), (H,), (i_ti,), (BC,), (0,))

    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_d = tl.load(p_d, boundary_check=(0, 1))
    b_g = tl.load(p_g, boundary_check=(0, 1))
    b_beta = tl.load(p_beta, boundary_check=(0,))

    if USE_GATHER:
        b_gn = gather(b_g, tl.full([1, BK], min(BC//2, T - i_ti - 1), dtype=tl.int16), axis=0)
    else:
        p_gn = g + (i_ti + min(BC // 2, T - i_ti - 1)) * H*K + tl.arange(0, BK)
        b_gn = tl.load(p_gn, mask=tl.arange(0, BK) < K, other=0.0)[None, :]

    b_gm = (b_g - b_gn).to(tl.float32)
    b_gq = tl.where(m_c[:, None], exp2(b_gm), 0.)
    b_gk = tl.where(m_c[:, None], exp2(-b_gm), 0.)

    b_kgt = tl.trans(b_k * b_d * b_gk)

    b_Aqk = tl.dot(b_q * b_gq, b_kgt) * scale
    b_Akk = tl.dot(b_k * b_gq, b_kgt) * b_beta[:, None]

    o_i = tl.arange(0, BC)
    m_Aqk = o_i[:, None] >= o_i[None, :]
    m_Akk = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]

    b_Aqk = tl.where(m_Aqk, b_Aqk, 0.0)
    b_Akk = tl.where(m_Akk, b_Akk, 0.0)

    tl.store(tl.make_block_ptr(Aqk, (T, BT), (H*BT, 1), (i_ti, i_i * BC), (BC, BC), (1, 0)), b_Aqk.to(Aqk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(tl.make_block_ptr(Akk, (T, BC), (H*BC, 1), (i_ti, 0), (BC, BC), (1, 0)), b_Akk.to(Akk.dtype.element_ty), boundary_check=(0, 1))

    tl.debug_barrier()

    b_Ai = -b_Akk
    for i in range(2, min(BC, T - i_ti)):
        b_a = -tl.load(Akk + (i_ti + i) * H*BC + o_i)
        b_a = tl.where(o_i < i, b_a, 0.)
        b_a += tl.sum(b_a[:, None] * b_Ai, 0)
        b_Ai = tl.where((o_i == i)[:, None], b_a, b_Ai)
    b_Ai += m_I
    tl.store(tl.make_block_ptr(Akk, (T, BC), (H*BC, 1), (i_ti, 0), (BC, BC), (1, 0)), b_Ai.to(Akk.dtype.element_ty), boundary_check=(0, 1))


def chunk_kda_fwd_intra(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, gk: torch.Tensor | None = None, beta: torch.Tensor | None = None,
    d: torch.Tensor | None = None, scale: float | None = None, cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64, chunk_indices: torch.LongTensor | None = None, safe_gate: bool = False, disable_recompute: bool = False,
):
    B, T, H, K = k.shape
    BT = chunk_size
    BC = 16
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    NC = triton.cdiv(BT, BC)

    Aqk = torch.empty(B, T, H, BT, device=k.device, dtype=k.dtype)
    Akk = torch.zeros(B, T, H, BT, device=k.device, dtype=k.dtype)
    Akkd = torch.empty(B, T, H, BC, device=k.device, dtype=torch.float32)

    if safe_gate:
        chunk_kda_fwd_kernel_intra_sub_chunk[(NT, NC, B * H)](
            q=q, k=k, d=d, g=gk, beta=beta, Aqk=Aqk, Akk=Akkd, scale=scale, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
            T=T, H=H, K=K, BT=BT, BC=BC, BK=triton.next_power_of_2(K), USE_GATHER=IS_GATHER_SUPPORTED,
        )
    else:
        Aqk, Akkd = chunk_kda_fwd_intra_token_parallel(q=q, k=k, gk=gk, beta=beta, d=d, Aqk=Aqk, Akk=Akkd, scale=scale, cu_seqlens=cu_seqlens, chunk_size=BT, sub_chunk_size=BC)

    chunk_kda_fwd_kernel_inter_solve_fused[(NT, B * H)](
        q=q, k=k, d=d, g=gk, beta=beta, Aqk=Aqk, Akkd=Akkd, Akk=Akk, scale=scale, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
        T=T, H=H, K=K, BT=BT, BC=BC, IS_VARLEN=(cu_seqlens is not None), USE_SAFE_GATE=safe_gate,
    )
    
    w, u, qg, kg = recompute_w_u_fwd(k=k, v=v, beta=beta, A=Akk, d=d, q=q if disable_recompute else None, gk=gk, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices)
    return w, u, qg, kg, Aqk, Akk


def chunk_kda_bwd_intra(
    q: torch.Tensor, k: torch.Tensor, g: torch.Tensor, beta: torch.Tensor, d: torch.Tensor, dd: torch.Tensor,
    dAqk: torch.Tensor, dAkk: torch.Tensor, dq: torch.Tensor, dk: torch.Tensor, db: torch.Tensor, dg: torch.Tensor,
    cu_seqlens: torch.LongTensor | None = None, chunk_indices: torch.LongTensor | None = None, chunk_size: int = 64, safe_gate: bool = False,
):
    B, T, H, K = k.shape
    BT = chunk_size
    BC = min(16, BT)
    BK = min(32, triton.next_power_of_2(K))

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    NC = triton.cdiv(BT, BC)
    NK = triton.cdiv(K, BK)

    chunk_kda_bwd_kernel_intra[(NK * NC, NT, B * H)](
        q=q, k=k, d=d, g=g, beta=beta, dAqk=dAqk, dAkk=dAkk, dq=dq, dk=dk, dd=dd, dg=dg, db=db, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
        B=B, T=T, H=H, K=K, BT=BT, BC=BC, BK=BK, NC=NC, IS_VARLEN=(cu_seqlens is not None), SAFE_GATE=safe_gate, USE_GATHER=IS_GATHER_SUPPORTED,
    )
    return dq, dk, db, dg