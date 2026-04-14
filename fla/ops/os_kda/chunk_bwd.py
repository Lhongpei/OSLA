# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
import torch
import triton
import triton.language as tl

from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_bwd_dhu, chunk_gated_delta_rule_fwd_h
from fla.ops.cp import FLACPContext
from fla.ops.cp.chunk_delta_h import chunk_gated_delta_rule_bwd_dhu_pre_process, expand_h0
from fla.ops.kda.gate import kda_gate_bwd, kda_gate_chunk_cumsum
from fla.ops.os_kda.wy_fast_os import recompute_w_u_fwd
from fla.ops.os_kda.chunk_intra import chunk_kda_bwd_intra
from fla.ops.utils import chunk_local_cumsum, prepare_chunk_indices
from fla.ops.utils.constant import RCP_LN2
from fla.ops.utils.op import exp2
from fla.utils import IS_NVIDIA_HOPPER, autotune_cache_kwargs, check_shared_mem, IS_GATHER_SUPPORTED

BK_LIST = [32, 64] if check_shared_mem() else [16, 32]
BV_LIST = [64, 128] if check_shared_mem('ampere') else [16, 32]
NUM_WARPS = [2, 4] if IS_NVIDIA_HOPPER else [2, 4, 8]

@triton.heuristics({'IS_VARLEN': lambda args: args['cu_seqlens'] is not None})
@triton.autotune(configs=[triton.Config({}, num_warps=nw, num_stages=ns) for nw in NUM_WARPS for ns in [2, 3, 4]], key=['H', 'K', 'V', 'BT', 'BK', 'BV'], **autotune_cache_kwargs)
@triton.jit(do_not_specialize=['T'])
def chunk_kda_bwd_kernel_dAv(
    q, k, v, A, do, dv, dA, cu_seqlens, chunk_indices, scale, T,
    H: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    q += (bos * H + i_h) * K; k += (bos * H + i_h) * K; v += (bos * H + i_h) * V
    do += (bos * H + i_h) * V; dv += (bos * H + i_h) * V; dA += (bos * H + i_h) * BT

    p_A = tl.make_block_ptr(A + (bos * H + i_h) * BT, (BT, T), (1, H*BT), (0, i_t * BT), (BT, BT), (0, 1))
    b_A = tl.load(p_A, boundary_check=(0, 1))
    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    m_A = (o_t[:, None] <= o_t[None, :]) & (m_t[:, None] & m_t)
    b_A = tl.where(m_A, b_A, 0).to(do.dtype.element_ty)

    b_dA = tl.zeros([BT, BT], dtype=tl.float32)
    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(v, (V, T), (1, H*V), (i_v * BV, i_t * BT), (BV, BT), (0, 1))
        p_do = tl.make_block_ptr(do, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_dv = tl.make_block_ptr(dv, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_dA += tl.dot(b_do, b_v)
        b_dv = tl.dot(b_A.to(b_do.dtype), b_do)
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))

    p_dA = tl.make_block_ptr(dA, (T, BT), (H*BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    b_dA = tl.where(o_t[:, None] >= o_t, b_dA * scale, 0.)
    tl.store(p_dA, b_dA.to(p_dA.dtype.element_ty), boundary_check=(0, 1))

@triton.heuristics({'IS_VARLEN': lambda args: args['cu_seqlens'] is not None})
@triton.autotune(configs=[triton.Config({'BK': BK, 'BV': BV}, num_warps=nw, num_stages=ns) for BK in BK_LIST for BV in BV_LIST for nw in NUM_WARPS for ns in [2, 3, 4] if not (IS_NVIDIA_HOPPER and BK == 32 and nw == 4)], key=['BT', 'TRANSPOSE_STATE'], **autotune_cache_kwargs)
@triton.jit(do_not_specialize=['T'])
def chunk_kda_bwd_kernel_wy_dqkg_fused(
    q, k, v, v_new, g, beta, d, A, h, do, dh, dq, dk, dd, dv, dv2, dg, db, dA, cu_seqlens, chunk_indices, scale, T, 
    H: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, TRANSPOSE_STATE: tl.constexpr, IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H

    if IS_VARLEN:
        i_tg = i_t.to(tl.int64)
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = (eos - bos).to(tl.int32)
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = (i_b * NT + i_t).to(tl.int64)
        bos, eos = (i_b * T).to(tl.int64), (i_b * T + T).to(tl.int64)

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    m_last = (o_t == min(T, i_t * BT + BT) - 1)

    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    d += (bos * H + i_h) * K 
    v += (bos * H + i_h) * V
    v_new += (bos * H + i_h) * V
    g += (bos * H + i_h) * K
    beta += bos * H + i_h
    A += (bos * H + i_h) * BT
    h += (i_tg * H + i_h) * K*V
    do += (bos * H + i_h) * V
    dh += (i_tg * H + i_h) * K*V
    dq += (bos * H + i_h) * K
    dk += (bos * H + i_h) * K
    dd += (bos * H + i_h) * K 
    dv += (bos * H + i_h) * V
    dv2 += (bos * H + i_h) * V
    dg += (bos * H + i_h) * K
    db += bos * H + i_h
    dA += (bos * H + i_h) * BT

    p_beta = tl.make_block_ptr(beta, (T,), (H,), (i_t * BT,), (BT,), (0,))
    b_beta = tl.load(p_beta, boundary_check=(0,))
    p_A = tl.make_block_ptr(A, (BT, T), (1, H * BT), (0, i_t * BT), (BT, BT), (0, 1))
    b_A = tl.load(p_A, boundary_check=(0, 1))

    b_dA = tl.zeros([BT, BT], dtype=tl.float32)
    b_db = tl.zeros([BT], dtype=tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K

        p_k = tl.make_block_ptr(k, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_d = tl.make_block_ptr(d, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)) 
        p_g = tl.make_block_ptr(g, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        
        b_k = tl.load(p_k, boundary_check=(0, 1)).to(tl.float32)
        b_d = tl.load(p_d, boundary_check=(0, 1)).to(tl.float32) 
        b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)

        p_gn = g + (min(T, i_t * BT + BT) - 1).to(tl.int64) * H*K + o_k
        b_gn = tl.load(p_gn, mask=m_k, other=0).to(tl.float32)

        b_dq = tl.zeros([BT, BK], dtype=tl.float32)
        b_dk = tl.zeros([BT, BK], dtype=tl.float32)
        b_dw = tl.zeros([BT, BK], dtype=tl.float32)
        b_dgk = tl.zeros([BK], dtype=tl.float32)

        for i_v in range(tl.cdiv(V, BV)):
            p_v_new = tl.make_block_ptr(v_new, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            p_do = tl.make_block_ptr(do, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            if TRANSPOSE_STATE:
                p_h = tl.make_block_ptr(h, (V, K), (K, 1), (i_v * BV, i_k * BK), (BV, BK), (1, 0))
                p_dh = tl.make_block_ptr(dh, (V, K), (K, 1), (i_v * BV, i_k * BK), (BV, BK), (1, 0))
            else:
                p_h = tl.make_block_ptr(h, (V, K), (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
                p_dh = tl.make_block_ptr(dh, (V, K), (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
            p_dv = tl.make_block_ptr(dv, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            
            b_v_new = tl.load(p_v_new, boundary_check=(0, 1))
            b_do = tl.load(p_do, boundary_check=(0, 1))
            b_h = tl.load(p_h, boundary_check=(0, 1))
            b_dh = tl.load(p_dh, boundary_check=(0, 1))
            b_dv = tl.load(p_dv, boundary_check=(0, 1))

            b_dgk += tl.sum(b_h * b_dh, axis=0)
            b_dq += tl.dot(b_do, b_h.to(b_do.dtype))
            b_dk += tl.dot(b_v_new, b_dh.to(b_v_new.dtype)) 
            b_dw += tl.dot(b_dv.to(b_v_new.dtype), b_h.to(b_v_new.dtype))
            tl.debug_barrier()
            
            if i_k == 0:
                p_v = tl.make_block_ptr(v, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
                p_dv2 = tl.make_block_ptr(dv2, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
                b_v = tl.load(p_v, boundary_check=(0, 1))
                b_dA += tl.dot(b_dv, tl.trans(b_v))
                b_dvb = tl.dot(b_A, b_dv)
                b_dv2 = b_dvb * b_beta[:, None]
                b_db += tl.sum(b_dvb * b_v, 1)
                tl.store(p_dv2, b_dv2.to(p_dv2.dtype.element_ty), boundary_check=(0, 1))

        b_gk_exp = exp2(b_g)
        b_gb = b_gk_exp * b_beta[:, None]
        b_dgk *= exp2(b_gn)
        b_dq = b_dq * b_gk_exp * scale
        
        b_dk_tilde = b_dk * tl.where(m_t[:, None], exp2(b_gn[None, :] - b_g), 0)

        b_kbg = b_k * b_beta[:, None] * b_gk_exp 
        b_dw = -b_dw.to(b_A.dtype)
        b_dA += tl.dot(b_dw, tl.trans(b_kbg.to(b_A.dtype)))

        b_dkgb = tl.dot(b_A, b_dw)
        b_dk_raw = b_dkgb * b_gb 
        b_db += tl.sum(b_dkgb * b_k * b_gk_exp, 1)

        p_q = tl.make_block_ptr(q, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        
        b_kdk = (b_k * b_dk_raw) + (b_k * b_d) * b_dk_tilde
        b_dgk += tl.sum(b_kdk, axis=0)
        
        b_dg = b_q * b_dq - b_kdk + m_last[:, None] * b_dgk + (b_k * b_gk_exp) * b_dkgb * b_beta[:, None] 
        
        b_dk_final = b_dk_raw + b_dk_tilde * b_d
        b_dd_final = b_dk_tilde * b_k

        p_dq = tl.make_block_ptr(dq, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_dk = tl.make_block_ptr(dk, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_dd = tl.make_block_ptr(dd, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)) 
        p_dg = tl.make_block_ptr(dg, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        
        tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dk, b_dk_tilde.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dd, b_dk_raw.to(p_dd.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0, 1))

    m_A = (o_t[:, None] > o_t[None, :]) & (m_t[:, None] & m_t)
    b_dA = tl.where(m_A, b_dA * b_beta[None, :], 0)
    b_dA = tl.dot(b_dA.to(b_A.dtype), b_A)
    b_dA = tl.dot(b_A, b_dA.to(b_A.dtype))
    b_dA = tl.where(m_A, -b_dA, 0)

    p_dA = tl.make_block_ptr(dA, (T, BT), (H * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    p_db = tl.make_block_ptr(db, (T,), (H,), (i_t * BT,), (BT,), (0,))
    tl.store(p_dA, b_dA.to(p_dA.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_db, b_db.to(p_db.dtype.element_ty), boundary_check=(0,))

@triton.heuristics({'IS_VARLEN': lambda args: args['cu_seqlens'] is not None})
@triton.autotune(configs=[triton.Config({}, num_warps=nw, num_stages=ns) for nw in [1, 2, 4, 8] for ns in [2, 3, 4]], key=['BK', 'NC', 'BT'], **autotune_cache_kwargs)
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

    q += (bos * H + i_h) * K; k += (bos * H + i_h) * K; d += (bos * H + i_h) * K
    g += (bos * H + i_h) * K; beta += bos * H + i_h
    dAqk += (bos * H + i_h) * BT; dAkk += (bos * H + i_h) * BT
    dq += (bos * H + i_h) * K; dk += (bos * H + i_h) * K
    dd += (bos * H + i_h) * K; dg += (bos * H + i_h) * K
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

    p_k_out = tl.make_block_ptr(k, (T, K), (H*K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
    p_d_out = tl.make_block_ptr(d, (T, K), (H*K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
    p_dk = tl.make_block_ptr(dk, (T, K), (H*K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
    p_dd = tl.make_block_ptr(dd, (T, K), (H*K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
    p_dg = tl.make_block_ptr(dg, (T, K), (H*K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
    p_db = tl.make_block_ptr(db, (T,), (H,), (i_ti,), (BC,), (0,))

    b_k_out = tl.load(p_k_out, boundary_check=(0, 1))
    b_d_out = tl.load(p_d_out, boundary_check=(0, 1))

    b_db_intra = tl.sum(b_dk_left * b_k_out, 1) 
    
    b_dk_raw_intra = b_dk_left * b_b[:, None]

    b_dkw_inter = tl.load(p_dk, boundary_check=(0, 1))
    b_dk_raw_inter = tl.load(p_dd, boundary_check=(0, 1))

    b_dkw_total = b_dkw_inter + b_dkt       
    b_dk_raw_total = b_dk_raw_inter + b_dk_raw_intra 

    b_dk_final = b_dk_raw_total + b_dkw_total * b_d_out
    b_dd_final = b_dkw_total * b_k_out

    b_dg2 += b_dk_left * b_k_out + tl.load(p_dg, boundary_check=(0, 1))
    
    b_db_total = tl.load(p_db, boundary_check=(0,)) + b_db_intra

    tl.store(p_dk, b_dk_final.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dd, b_dd_final.to(p_dd.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dg, b_dg2.to(g.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_db, b_db_total.to(beta.dtype.element_ty), boundary_check=(0,))

def chunk_kda_bwd_dAv(q, k, v, do, A=None, scale=None, cu_seqlens=None, chunk_size=64, chunk_indices=None):
    B, T, H, K, V = *k.shape, do.shape[-1]
    BT = chunk_size
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    if check_shared_mem('hopper', k.device.index):
        CONST_TILING = 128
    elif check_shared_mem:
        CONST_TILING = 64
    else:
        CONST_TILING = 32
    BK = min(max(triton.next_power_of_2(K), 16), CONST_TILING)
    BV = min(max(triton.next_power_of_2(V), 16), CONST_TILING)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    dA = v.new_empty(B, T, H, BT, dtype=torch.float)
    dv = torch.empty_like(do)
    grid = (NT, B * H)
    chunk_kda_bwd_kernel_dAv[grid](q=q, k=k, v=v, A=A, do=do, dv=dv, dA=dA, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, scale=scale, T=T, H=H, K=K, V=V, BT=BT, BK=BK, BV=BV)
    return dA, dv

def chunk_kda_bwd_wy_dqkg_fused(
    q, k, v, v_new, g, beta, d, dd, A, h, do, dh, dv, 
    scale=None, cu_seqlens=None, chunk_size=64, chunk_indices=None, transpose_state_layout=False,
):
    B, T, H, K, V = *k.shape, v.shape[-1]
    BT = chunk_size
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    dq = torch.empty_like(q, dtype=torch.float)
    dk = torch.empty_like(k, dtype=torch.float)
    dv2 = torch.empty_like(v)
    dg = torch.empty_like(g, dtype=torch.float)
    db = torch.empty_like(beta, dtype=torch.float)
    dA = torch.empty_like(A, dtype=torch.float)

    grid = (NT, B * H)
    chunk_kda_bwd_kernel_wy_dqkg_fused[grid](
        q=q, k=k, v=v, v_new=v_new, g=g, beta=beta, d=d, A=A, h=h, do=do, dh=dh, dq=dq, dk=dk, dd=dd, 
        dv=dv, dv2=dv2, dg=dg, db=db, dA=dA, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, scale=scale,
        T=T, H=H, K=K, V=V, BT=BT, TRANSPOSE_STATE=transpose_state_layout,
    )
    dv = dv2
    return dq, dk, dv, db, dg, dA

def chunk_kda_bwd(
    q, k, v, beta, Aqk, Akk, scale, initial_state, do, dht, d, dd_final, eta, use_denominator, d_min, d_max, output_initial_d_gradient, 
    g=None, g_org=None, cu_seqlens=None, chunk_indices=None, chunk_size=64, safe_gate=False, lower_bound=None,
    use_gate_in_kernel=False, A_log=None, dt_bias=None, disable_recompute=False, cp_context=None, transpose_state_layout=False, **kwargs,
):
    from fla.ops.os_delta_rule.chunk_osgm_phase import compute_osgm_phase1_bwd  # 🚀

    if disable_recompute is False:
        if use_gate_in_kernel:
            g = kda_gate_chunk_cumsum(g=g_org, A_log=A_log, dt_bias=dt_bias, scale=RCP_LN2, chunk_size=chunk_size, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, lower_bound=lower_bound)
        w, u, qg, kg = recompute_w_u_fwd(q=q, k=k, v=v, beta=beta, A=Akk, d=d, gk=g, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices) 
        if cp_context is not None:
            initial_state = expand_h0(initial_state, context=cp_context)
        h, v_new, _ = chunk_gated_delta_rule_fwd_h(k=kg, w=w, u=u, gk=g, initial_state=initial_state, output_final_state=False, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, use_exp2=True, transpose_state_layout=transpose_state_layout)
    else:
        w, u, qg, kg, v_new, h = kwargs["w"], kwargs["u"], kwargs["qg"], kwargs["kg"], kwargs["v_new"], kwargs["h"]

    dAqk, dv = chunk_kda_bwd_dAv(q=q, k=k, v=v_new, do=do, A=Aqk, scale=scale, cu_seqlens=cu_seqlens, chunk_size=chunk_size, chunk_indices=chunk_indices)

    if cp_context is not None:
        dht, initial_state = chunk_gated_delta_rule_bwd_dhu_pre_process(q=qg, k=kg, w=w, do=do, dv=dv, gk=g, scale=scale, cu_seqlens=cu_seqlens, dht=dht, initial_state=initial_state, use_exp2=True, context=cp_context, transpose_state_layout=transpose_state_layout)

    dh, dh0, dv = chunk_gated_delta_rule_bwd_dhu(q=qg, k=kg, w=w, gk=g, h0=initial_state, dht=dht, do=do, dv=dv, scale=scale, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, use_exp2=True, transpose_state_layout=transpose_state_layout)

    dd = torch.zeros_like(k)

    dq, dk, dv, db, dg, dAkk = chunk_kda_bwd_wy_dqkg_fused(
        q=q, k=k, v=v, v_new=v_new, g=g, beta=beta, d=d, dd=dd, 
        A=Akk, h=h, do=do, dh=dh, dv=dv, scale=scale, cu_seqlens=cu_seqlens, chunk_size=chunk_size, chunk_indices=chunk_indices, transpose_state_layout=transpose_state_layout,
    )

    dq, dk, db, dg = chunk_kda_bwd_intra(
        q=q, 
        k=k, 
        g=g, 
        beta=beta, 
        d=d, 
        dd=dd, 
        dAqk=dAqk, 
        dAkk=dAkk, 
        dq=dq, 
        dk=dk, 
        db=db, 
        dg=dg, 
        cu_seqlens=cu_seqlens, 
        chunk_indices=chunk_indices, 
        chunk_size=chunk_size, 
        safe_gate=safe_gate
    )

    dk_phase1, dd_initial = compute_osgm_phase1_bwd(
        k=k, d_out=d, dd_in=dd, eta=eta, use_denominator=use_denominator, d_min=d_min, d_max=d_max, cu_seqlens=cu_seqlens, dd_final=dd_final, output_initial_state_gradient=output_initial_d_gradient
    )

    dk_final = dk + dk_phase1

    dA, dbias = None, None
    dg = chunk_local_cumsum(dg, chunk_size=chunk_size, reverse=True, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices)
    if use_gate_in_kernel:
        dg, dA, dbias = kda_gate_bwd(g=g_org, A_log=A_log, dt_bias=dt_bias, dyg=dg, lower_bound=lower_bound)

    return dq, dk_final, dv, db, dg, dh0, dA, dbias, dd_initial