# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from fla.modules.l2norm import l2norm_bwd, l2norm_fwd
from fla.utils import input_guard


@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None
})
@triton.jit(do_not_specialize=['T'])
def fused_recurrent_delta_rule_fwd_kernel(
    q, k, v, u, beta, o,
    h0, ht, scale_0, scale_t,
    cu_seqlens, scale, T,
    B: tl.constexpr, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BK: tl.constexpr, BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr, STORE_FINAL_STATE: tl.constexpr,
    IS_BETA_HEADWISE: tl.constexpr, IS_VARLEN: tl.constexpr,
):
    i_v, i_k, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_h = i_nh // H, i_nh % H

    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        all_t = T
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
        all_t = B * T

    # 指针定义
    p_q = q + (bos * H + i_h) * K + i_k * BK + tl.arange(0, BK)
    p_k = k + (bos * H + i_h) * K + i_k * BK + tl.arange(0, BK)
    p_v = v + (bos * H + i_h) * V + i_v * BV + tl.arange(0, BV)
    p_u = u + (bos * H + i_h) * V + i_v * BV + tl.arange(0, BV)
    p_o = o + ((i_k * all_t + bos) * H + i_h) * V + i_v * BV + tl.arange(0, BV)
    
    if IS_BETA_HEADWISE:
        p_beta = beta + (bos * H + i_h) * V + i_v * BV + tl.arange(0, BV)
    else:
        p_beta = beta + bos * H + i_h

    mask_k = (i_k * BK + tl.arange(0, BK)) < K
    mask_v = (i_v * BV + tl.arange(0, BV)) < V
    mask_h = mask_k[None, :] & mask_v[:, None]

    b_h = tl.zeros([BV, BK], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = h0 + i_n * H * K * V + i_h * K * V + (i_k * BK + tl.arange(0, BK)[None, :]) * V + (i_v * BV + tl.arange(0, BV)[:, None])
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    # 预条件器初始值 D_0
    p_scale = scale_0 + i_n * H * K + i_h * K + i_k * BK + tl.arange(0, BK)
    b_scale = tl.load(p_scale, mask=mask_k, other=0).to(tl.float32)

    for _ in range(0, T):
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32) * scale
        
        # u_t = v_t - S_{t-1} k_t
        b_v_minus = tl.sum(b_h * b_k[None, :], axis=1)
        b_v -= b_v_minus
        tl.store(p_u, b_v.to(p_u.dtype.element_ty), mask=mask_v)

        # D_t = D_{t-1} + k_t^2
        b_scale += b_k * b_k
        
        if IS_BETA_HEADWISE:
            b_beta = tl.load(p_beta, mask=mask_v, other=0).to(tl.float32)
        else:
            b_beta = tl.load(p_beta).to(tl.float32)
        
        # S_t = S_{t-1} + (beta * u_t) k_t^T / D_t
        b_h += (b_v * b_beta)[:, None] * b_k[None, :] / (b_scale[None, :] + 1.0)

        # o_t = S_t q_t
        b_o = tl.sum(b_h * b_q[None, :], axis=1)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        p_q += H*K; p_k += H*K; p_o += H*V; p_v += H*V; p_u += H*V
        p_beta += H * (V if IS_BETA_HEADWISE else 1)

    if STORE_FINAL_STATE:
        p_ht = ht + i_n * H * K * V + i_h * K * V + (i_k * BK + tl.arange(0, BK)[None, :]) * V + (i_v * BV + tl.arange(0, BV)[:, None])
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)
        p_scale_t = scale_t + i_n * H * K + i_h * K + i_k * BK + tl.arange(0, BK)
        tl.store(p_scale_t, b_scale.to(p_scale_t.dtype.element_ty), mask=mask_k)


@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'USE_FINAL_STATE_GRADIENT': lambda args: args['dht'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None
})
@triton.jit(do_not_specialize=['T'])
def fused_recurrent_delta_rule_bwd_kernel(
    q, k, v, beta, h0, dh0, dht, do, dq, dk, dv, db,
    scale_0, d_scale_0, cu_seqlens, scale,
    B: tl.constexpr, T, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BK: tl.constexpr, BV: tl.constexpr, NK: tl.constexpr,
    IS_BETA_HEADWISE: tl.constexpr, USE_INITIAL_STATE: tl.constexpr,
    USE_FINAL_STATE_GRADIENT: tl.constexpr, IS_VARLEN: tl.constexpr,
):
    i_v, i_k, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_h = i_nh // H, i_nh % H
    
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        all_t = T
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
        all_t = B * T

    mask_k = i_k * BK + tl.arange(0, BK) < K
    mask_v = i_v * BV + tl.arange(0, BV) < V

    # 反向指针 (从 T-1 开始)
    p_q = q + (bos * H + i_h) * K + i_k * BK + tl.arange(0, BK) + (T - 1) * H*K
    p_k = k + (bos * H + i_h) * K + i_k * BK + tl.arange(0, BK) + (T - 1) * H*K
    p_v = v + (bos * H + i_h) * V + i_v * BV + tl.arange(0, BV) + (T - 1) * H*V
    p_do = do + (bos * H + i_h) * V + i_v * BV + tl.arange(0, BV) + (T - 1) * H*V
    p_dk = dk + ((i_v * all_t + bos) * H + i_h) * K + i_k * BK + tl.arange(0, BK) + (T - 1) * H*K
    p_dv = dv + ((i_k * all_t + bos) * H + i_h) * V + i_v * BV + tl.arange(0, BV) + (T - 1) * H*V
    
    if IS_BETA_HEADWISE:
        p_beta = beta + (bos + T - 1) * H*V + i_h * V + i_v * BV + tl.arange(0, BV)
        p_dbeta = db + ((i_v * NK + i_k) * all_t + bos + T - 1) * H*V + i_h * V + tl.arange(0, BV)
    else:
        p_beta = beta + (bos + T - 1) * H + i_h
        p_dbeta = db + (i_v * all_t + bos + T - 1) * H + i_h

    # b_dh 是 [BK, BV]，表示 dL/dS_t^T（S_t 的转置的梯度）
    b_dh = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_FINAL_STATE_GRADIENT:
        p_dht = dht + i_n * H * K * V + i_h * K * V + (i_k * BK + tl.arange(0, BK)[:, None]) * V + (i_v * BV + tl.arange(0, BV)[None, :])
        b_dh += tl.load(p_dht, mask=mask_k[:, None] & mask_v[None, :], other=0).to(tl.float32)

    # 重算 D_T（从后往前的累积）
    p_scale_0 = scale_0 + i_n * H * K + i_h * K + i_k * BK + tl.arange(0, BK)
    b_scale = tl.load(p_scale_0, mask=mask_k, other=0).to(tl.float32)
    p_k_fwd = k + (bos * H + i_h) * K + i_k * BK + tl.arange(0, BK)
    for _ in range(T):
        b_k_fwd = tl.load(p_k_fwd, mask=mask_k, other=0).to(tl.float32)
        b_scale += b_k_fwd * b_k_fwd
        p_k_fwd += H*K

    # 用于累积 d(d_t) 的梯度
    b_dd = tl.zeros([BK], dtype=tl.float32)

    for _ in range(T):
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32) * scale
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_u = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_do = tl.load(p_do, mask=mask_v, other=0).to(tl.float32)
        if IS_BETA_HEADWISE:
            b_beta = tl.load(p_beta, mask=mask_v, other=0).to(tl.float32)
        else:
            b_beta = tl.load(p_beta).to(tl.float32)
        
        # S_t^T = S_{t-1}^T + p_t (beta * u_t)^T, 其中 p_t = k_t / d_t
        # o_t = S_t q_t = q_t^T S_t^T
        # 注意：S_t 是 [V, K]，S_t^T 是 [K, V]
        # b_do 是 dL/do_t [BV]，b_q 是 q_t [BK]
        # dL/dS_t^T += q_t (dL/do_t)^T，外积 [BK] x [BV] -> [BK, BV]
        b_dh += b_q[:, None] * b_do[None, :]
        
        # 从 S_t^T 的更新反推
        # S_t^T = S_{t-1}^T + p_t (beta * u_t)^T
        # dL/d(beta * u_t) = S_t^T^T dL/dS_t^T 的内积？不对
        # 实际上：dL/d(beta * u_t) = (dL/dS_t^T)^T @ p_t
        # b_dh: [K, V], p_t: [K],需要计算 b_dh^T @ p_t
        b_p = b_k / (b_scale + 1.0)  # [BK]
        b_beta_u = b_u * b_beta  # [BV]
        
        # dL/d(beta * u_t) [BV] = b_dh^T [V, K] @ p_t [K]
        b_d_beta_u = tl.sum(b_dh * b_p[:, None], axis=0)  # [BV]
        
        # dL/du_t = dL/d(beta * u_t) * beta
        b_du = b_d_beta_u * b_beta  # [BV]
        
        # dL/dbeta_t = dL/d(beta * u_t) * u_t
        b_db = b_d_beta_u * b_u if IS_BETA_HEADWISE else tl.sum(b_d_beta_u * b_u)  # [BV] or scalar
        
        # dL/dp_t = dL/dS_t^T @ (beta * u_t) [K, V] @ [V] -> [K]
        b_dp = tl.sum(b_dh * b_beta_u[None, :], axis=1)  # [BK]
        
        # p_t = k_t / d_t
        # dL/dk_t (来自 p_t) = dL/dp_t / d_t
        b_dk_from_p = b_dp / (b_scale + 1.0)  # [BK]

        # dL/dd_t (来自 p_t) = -dL/dp_t * k_t / d_t^2 = -dL/dp_t * p_t / d_t
        b_dd -= b_dp * b_p / (b_scale + 1.0)  # [BK]
        
        # d_t = d_{t-1} + k_t^2
        # dL/dk_t (来自 d_t) = 2 * k_t * dL/dd_t (累积的)
        b_dk_from_d = 2 * b_k * b_dd  # [BK]
        
        # 总的 dk = dk_from_p + dk_from_d
        b_dk = b_dk_from_p + b_dk_from_d  # [BK]
        
        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), mask=mask_k)
        tl.store(p_dv, b_du.to(p_dv.dtype.element_ty), mask=mask_v)
        if IS_BETA_HEADWISE:
            tl.store(p_dbeta, b_db.to(p_dbeta.dtype.element_ty), mask=mask_v)
        else:
            tl.store(p_dbeta, b_db.to(p_dbeta.dtype.element_ty))

        # S_{t-1}^T = S_t^T - p_t (beta * u_t)^T
        # dL/dS_{t-1}^T = dL/dS_t^T (因为 u_t 不直接依赖于 S_{t-1}^T？不对)
        # 实际上 u_t = v_t - S_{t-1} k_t，所以 S_{t-1} 影响 u_t
        # 但从 dh 的视角，我们需要减去对 S_{t-1}^T 有贡献的部分
        # dL/dS_{t-1}^T = dL/dS_t^T - dL/d(beta * u_t) * (beta * u_t)^T？不对
        # 
        # 重新推导：S_{t-1}^T = S_t^T - p_t (beta * u_t)^T
        # 所以 dL/dS_{t-1}^T = dL/dS_t^T（因为 S_t^T 依赖于 S_{t-1}^T）
        # 但是 u_t 依赖于 S_{t-1}，我们需要考虑 du_t 对 S_{t-1} 的梯度
        # 
        # 实际上，在反向传播中：
        # S_{t-1}^T 影响 u_t (通过 S_{t-1} k_t)
        # u_t 影响 S_t^T
        # 所以 dL/dS_{t-1}^T 应该包含两部分：
        # 1. 直接从 S_t^T 传来的梯度（通过 S_t^T = S_{t-1}^T + ...）
        # 2. 从 u_t 传来的梯度（通过 u_t = v_t - S_{t-1} k_t）
        #
        # 第 1 部分就是当前的 b_dh
        # 第 2 部分：dL/du_t * du_t/dS_{t-1}^T = b_du * (-k_t)^T
        # 所以 dL/dS_{t-1}^T += -b_du [BV] @ k_t^T [BK] -> [BK, BV]？
        # 不对，S_{t-1} 是 [V, K]，S_{t-1}^T 是 [K, V]
        # du_t = -S_{t-1} k_t，所以 du_t/dS_{t-1} = -k_t^T (广播到 [V, K])
        # dL/dS_{t-1} [V, K] += dL/du_t [V, 1] @ k_t [1, K]
        # dL/dS_{t-1}^T [K, V] = (dL/dS_{t-1})^T
        #
        # 所以：dL/dS_{t-1}^T -= k_t [K, 1] @ b_du [1, V]？
        # 不对，应该是：dL/dS_{t-1}^T -= k_t[:, None] * b_du[None, :]？
        # 等等，b_du [BV]，k_t [BK]
        # 外积 k_t [K] x b_du [V] -> [K, V]
        #
        # 实际上，我们需要减去的是：k_t 作为列，b_du 作为行
        b_dh -= b_k[:, None] * b_du[None, :]  # [BK, BV]
        
        # 回退 d_t
        b_scale -= b_k * b_k

        p_q -= H*K; p_k -= H*K; p_v -= H*V; p_do -= H*V; p_dk -= H*K; p_dv -= H*V
        p_dbeta -= H * (V if IS_BETA_HEADWISE else 1); p_beta -= H * (V if IS_BETA_HEADWISE else 1)

    if USE_INITIAL_STATE:
        p_dh0 = dh0 + i_n * H * K * V + i_h * K * V + (i_k * BK + tl.arange(0, BK)[:, None]) * V + (i_v * BV + tl.arange(0, BV)[None, :])
        tl.store(p_dh0, b_dh.to(p_dh0.dtype.element_ty), mask=mask_k[:, None] & mask_v[None, :])

    # Store dL/dD_0 (accumulated preconditioner gradient)
    # b_dd is [BK], one per V-block (i_v). Sum across V-blocks happens outside.
    p_ds0 = d_scale_0 + (i_v * (B if not IS_VARLEN else 1) * H + i_n * H + i_h) * K + i_k * BK + tl.arange(0, BK)
    tl.store(p_ds0, b_dd.to(p_ds0.dtype.element_ty), mask=mask_k)

    tl.debug_barrier()
    
    # Loop 2: 正向计算 dq 并修正 dk
    # 需要重新计算 h (S_t) 状态
    b_h = tl.zeros([BV, BK], dtype=tl.float32)  # S_t [V, K]
    b_scale = tl.load(p_scale_0, mask=mask_k, other=0).to(tl.float32)
    
    # 重置指针到 bos
    p_k = k + (bos * H + i_h) * K + i_k * BK + tl.arange(0, BK)
    p_v = v + (bos * H + i_h) * V + i_v * BV + tl.arange(0, BV)
    p_do = do + (bos * H + i_h) * V + i_v * BV + tl.arange(0, BV)
    p_dq = dq + ((i_v * all_t + bos) * H + i_h) * K + i_k * BK + tl.arange(0, BK)
    p_dk = dk + ((i_v * all_t + bos) * H + i_h) * K + i_k * BK + tl.arange(0, BK)
    p_dv = dv + ((i_k * all_t + bos) * H + i_h) * V + i_v * BV + tl.arange(0, BV)
    
    if IS_BETA_HEADWISE:
        p_beta = beta + (bos * H + i_h) * V + i_v * BV + tl.arange(0, BV)
    else:
        p_beta = beta + bos * H + i_h

    if USE_INITIAL_STATE:
        p_h0 = h0 + i_n * H * K * V + i_h * K * V + (i_k * BK + tl.arange(0, BK)[None, :]) * V + (i_v * BV + tl.arange(0, BV)[:, None])
        b_h += tl.load(p_h0, mask=mask_k[:, None] & mask_v[None, :], other=0).to(tl.float32)

    for _ in range(T):
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_u = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_do = tl.load(p_do, mask=mask_v, other=0).to(tl.float32)
        b_du = tl.load(p_dv, mask=mask_v, other=0).to(tl.float32)
        b_dk = tl.load(p_dk, mask=mask_k, other=0).to(tl.float32)
        
        if IS_BETA_HEADWISE:
            b_beta = tl.load(p_beta, mask=mask_v, other=0).to(tl.float32)
        else:
            b_beta = tl.load(p_beta).to(tl.float32)

        # u_t = v_t - S_{t-1} k_t
        # v_t 的梯度中，来自 u_t 的部分已经计算了
        # 但 dk 还需要加上 -S_{t-1}^T @ du_t 的项
        # S_{t-1} 是 [V, K]，S_{t-1}^T 是 [K, V]
        # du_t 是 [V]，所以 S_{t-1}^T @ du_t 是 [K, V] @ [V] -> [K]
        # 实际上：b_h 是 S_{t-1} [V, K]
        # dL/dk_t (来自 u_t) = -S_{t-1}^T @ dL/du_t = -b_h^T @ b_du
        # b_h [V, K] * b_du [V, 1] -> sum over V -> [K]
        b_dk_from_u = -tl.sum(b_h * b_du[:, None], axis=0)  # [BK]
        b_dk += b_dk_from_u
        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), mask=mask_k)

        # 更新状态用于下一个时间步的 dq 计算
        b_scale += b_k * b_k
        b_h += (b_u * b_beta)[:, None] * b_k[None, :] / (b_scale[None, :] + 1.0)

        # o_t = S_t q_t
        # dL/dq_t = S_t^T @ dL/do_t = b_h^T @ b_do
        # b_h 是 S_t [V, K]，b_h^T 是 [K, V]
        # b_do 是 [V]，所以 b_h^T @ b_do 是 [K]
        # b_h [V, K] * b_do [V, 1] -> sum over V -> [K]
        b_dq = tl.sum(b_h * b_do[:, None], axis=0) * scale  # [BK]
        tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), mask=mask_k)

        p_k += H*K; p_v += H*V; p_do += H*V; p_dq += H*K; p_dk += H*K; p_dv += H*V
        p_beta += H * (V if IS_BETA_HEADWISE else 1)


def fused_recurrent_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    initial_scale: torch.Tensor,
    output_final_state: bool,
    cu_seqlens: Optional[torch.LongTensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *k.shape, v.shape[-1]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BK, BV = triton.next_power_of_2(K), min(triton.next_power_of_2(V), 8)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
    assert NK == 1, "NK > 1 is not supported yet"
    num_stages = 1
    num_warps = 1

    o = q.new_empty(NK, *v.shape)
    if output_final_state:
        final_state = q.new_empty(N, H, K, V, dtype=torch.float32)
        final_scale = q.new_empty(N, H, K, dtype=torch.float32)
    else:
        final_state = None
        final_scale = None

    grid = (NV, NK, N * H)
    u = torch.empty_like(v)
    fused_recurrent_delta_rule_fwd_kernel[grid](
        q,
        k,
        v,
        u,
        beta,
        o,
        initial_state,
        final_state,
        initial_scale,
        final_scale,
        cu_seqlens,
        scale,
        T=T,
        B=B,
        H=H,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        IS_BETA_HEADWISE=beta.ndim == v.ndim,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    o = o.squeeze(0)
    return o, u, final_state, final_scale


def fused_recurrent_delta_rule_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    dht: torch.Tensor,
    do: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    initial_scale: torch.Tensor,
    cu_seqlens: Optional[torch.LongTensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *k.shape, v.shape[-1]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BK, BV = triton.next_power_of_2(K), min(triton.next_power_of_2(V), 32)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
    assert NK == 1, "NK > 1 is not supported yet"
    num_stages = 1
    num_warps = 2

    beta_vector = beta.ndim == v.ndim

    dq = q.new_empty(NV, *q.shape)
    dk = q.new_empty(NV, *k.shape)
    dv = q.new_empty(NK, *v.shape)
    if beta_vector:
        db = q.new_empty(NV, NK, B, T, H, V)
    else:
        db = q.new_empty(NV, B, T, H)
    grid = (NV, NK, N * H)

    if initial_state is not None and initial_state.requires_grad:
        dh0 = torch.empty_like(initial_state, dtype=torch.float32)
    else:
        dh0 = None

    # d_scale_0: gradient of initial_scale, shape [NV, N, H, K] (sum over NV blocks later)
    d_scale_0 = q.new_empty(NV, N, H, K, dtype=torch.float32)

    fused_recurrent_delta_rule_bwd_kernel[grid](
        q,
        k,
        v,
        beta,
        initial_state,
        dh0,
        dht,
        do,
        dq,
        dk,
        dv,
        db,
        initial_scale,
        d_scale_0,
        cu_seqlens,
        scale,
        T=T,
        B=B,
        H=H,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        NK=NK,
        IS_BETA_HEADWISE=beta_vector,
        num_warps=num_warps,
        num_stages=num_stages
    )
    dq = dq.sum(0)
    dk = dk.sum(0)
    dv = dv.sum(0)
    db = db.sum((0, 1)) if beta_vector else db.sum(0)
    d_scale_0 = d_scale_0.sum(0)

    return dq, dk, dv, db, dh0, d_scale_0


class FusedRecurrentFunction(torch.autograd.Function):

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
        if initial_scale is None:
            initial_scale = q.new_zeros(N, H, K, dtype=torch.float32)
        o, u, final_state, final_scale = fused_recurrent_delta_rule_fwd(
            q=q,
            k=k,
            v=v,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            initial_scale=initial_scale,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
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
            q=q,
            k=k,
            v=u,           # u is the stored innovation from forward
            beta=beta,
            dht=dht,
            do=do,
            scale=ctx.scale,
            initial_state=initial_state,
            initial_scale=initial_scale,
            cu_seqlens=ctx.cu_seqlens,
        )

        if ctx.use_qk_l2norm_in_kernel:
            dq = l2norm_bwd(q, q_rstd, dq)
            dk = l2norm_bwd(k, k_rstd, dk)

        # Return order matches forward inputs:
        # q, k, v, beta, scale, initial_state, initial_scale, output_final_state, use_qk_l2norm, cu_seqlens
        return dq.to(q), dk.to(k), dv.to(u), db.to(beta), None, dh0, d_scale_0, None, None, None


@torch.compiler.disable
def fused_recurrent_delta_rule(
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
)-> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, T, H, K]`.
        k (torch.Tensor):
            keys of shape `[B, T, H, K]`.
        v (torch.Tensor):
            values of shape `[B, T, H, V]`.
        beta (torch.Tensor):
            betas of shape `[B, T, H]`.
        scale (Optional[float]):
            Scale factor for the RetNet attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N, H, K, V]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, H, K, V]`. Default: `False`.
        use_qk_l2norm_in_kernel (Optional[bool]):
            Whether to use L2 normalization in the kernel. Default: `False`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, H, V]`.
        final_state (torch.Tensor):
            Final state of shape `[N, H, K, V]` if `output_final_state=True` else `None`.

    Examples::
        >>> import torch
        >>> import torch.nn.functional as F
        >>> from einops import rearrange
        >>> from fla.ops.delta_rule import fused_recurrent_delta_rule
        # inputs with equal lengths
        >>> B, T, H, K, V = 4, 2048, 4, 512, 512
        >>> q = torch.randn(B, T, H, K, device='cuda')
        >>> k = F.normalize(torch.randn(B, T, H, K, device='cuda'), p=2, dim=-1)
        >>> v = torch.randn(B, T, H, V, device='cuda')
        >>> beta = torch.rand(B, T, H, device='cuda').sigmoid()
        >>> h0 = torch.randn(B, H, K, V, device='cuda')
        >>> o, ht = fused_recurrent_delta_rule(
            q, k, v, beta,
            initial_state=h0,
            output_final_state=True
        )
        # for variable-length inputs, the batch size `B` is expected to be 1 and `cu_seqlens` is required
        >>> q, k, v, beta = map(lambda x: rearrange(x, 'b t ... -> 1 (b t) ...'), (q, k, v, beta))
        # for a batch with 4 sequences, `cu_seqlens` with 5 start/end positions are expected
        >>> cu_seqlens = q.new_tensor([0, 2048, 4096, 6144, 8192], dtype=torch.long)
        >>> o, ht = fused_recurrent_delta_rule(
            q, k, v, beta,
            initial_state=h0,
            output_final_state=True,
            cu_seqlens=cu_seqlens
        )
    """
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing."
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}."
            )
    if scale is None:
        scale = k.shape[-1] ** -0.5
    else:
        assert scale > 0, "scale must be positive"
    if beta is None:
        beta = torch.ones_like(q[..., 0])
    if initial_scale is None:
        B, H, K = q.shape[0], q.shape[2], q.shape[3]
        N = B if cu_seqlens is None else len(cu_seqlens) - 1
        initial_scale = q.new_zeros(N, H, K, dtype=torch.float32)
    o, final_state, final_scale = FusedRecurrentFunction.apply(
        q, k, v, beta, scale,
        initial_state, initial_scale,
        output_final_state, use_qk_l2norm_in_kernel, cu_seqlens,
    )
    if output_final_state:
        return o, (final_state, final_scale)
    return o, None
