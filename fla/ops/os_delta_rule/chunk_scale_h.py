
# from typing import Optional, Tuple

# import torch
# import triton
# import triton.language as tl

# from fla.ops.utils import prepare_chunk_indices
# from fla.ops.utils.op import exp
# from fla.utils import check_shared_mem, is_nvidia_hopper

# BKV_LIST = [64, 128] if check_shared_mem() else [32, 64]
# NUM_WARPS = [2, 4] if is_nvidia_hopper else [2, 4, 8]


# @triton.heuristics({
#     'USE_G': lambda args: args['g'] is not None,
#     'USE_G_GAMMA': lambda args: args['g_gamma'] is not None,
#     'IS_VARLEN': lambda args: args['cu_seqlens'] is not None
# })
# @triton.autotune(
#     configs=[
#         triton.Config({'BK': 128, 'BV': 128}, num_warps=8, num_stages=3),
#         triton.Config({'BK': 64, 'BV': 64}, num_warps=4, num_stages=3),
#         triton.Config({'BK': 32, 'BV': 32}, num_warps=2, num_stages=3),
#     ],
#     key=['H', 'K', 'V', 'BT'],
# )
# @triton.jit(do_not_specialize=['T'])
# def prepare_scale_chunk_fwd_kernel(
#     k,
#     h,
#     o,
#     cu_seqlens,
#     chunk_indices,
#     scale,
#     T,
#     H: tl.constexpr,
#     K: tl.constexpr,
#     V: tl.constexpr,
#     BT: tl.constexpr,
#     BK: tl.constexpr,
#     IS_VARLEN: tl.constexpr,
# ):
#     i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
#     i_b, i_h = i_bh // H, i_bh % H

#     if IS_VARLEN:
#         i_tg = i_t
#         i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
#         bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
#         T = eos - bos
#         NT = tl.cdiv(T, BT)
#     else:
#         NT = tl.cdiv(T, BT)
#         i_tg = i_b * NT + i_t
#         bos, eos = i_b * T, i_b * T + T

#     # offset calculation
#     q += (bos * H + i_h) * K
#     k += (bos * H + i_h) * K
#     v += (bos * H + i_h) * V
#     o += (bos * H + i_h) * V
#     h += (i_tg * H + i_h).to(tl.int64) * K*V

#     b_o = tl.zeros([BT, BV], dtype=tl.float32)
#     b_A = tl.zeros([BT, BT], dtype=tl.float32)

#     for i_k in range(tl.cdiv(K, BK)):
#         p_q = tl.make_block_ptr(q, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
#         p_k = tl.make_block_ptr(k, (K, T), (1, H*K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
#         p_h = tl.make_block_ptr(h, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
#         # [BT, BK]
#         b_q = tl.load(p_q, boundary_check=(0, 1))
#         # [BK, BT]
#         b_k = tl.load(p_k, boundary_check=(0, 1))
#         # [BK, BV]
#         b_h = tl.load(p_h, boundary_check=(0, 1))

#         # [BT, BK] @ [BK, BV] -> [BT, BV]
#         b_o += tl.dot(b_q, b_h)
#         # [BT, BK] @ [BK, BT] -> [BT, BT]
#         b_A += tl.dot(b_q, b_k)

#     o_t = i_t * BT + tl.arange(0, BT)
#     m_t = o_t < T
#     m_A = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t)
#     b_A = tl.where(m_A, b_A, 0)

#     p_v = tl.make_block_ptr(v, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
#     p_o = tl.make_block_ptr(o, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))

#     b_v = tl.load(p_v, boundary_check=(0, 1))
#     # to fix mma -> mma layout conversion
#     # already solved by triton v3.2 or higher
#     b_o = b_o * scale + tl.dot(b_A.to(b_v.dtype), b_v) * scale
    
# def prepare_scale_chunk_fwd(
#     q: torch.Tensor,
#     k: torch.Tensor,
#     v: torch.Tensor,
#     h: torch.Tensor,
#     g: Optional[torch.Tensor] = None,
#     g_gamma: Optional[torch.Tensor] = None,
#     scale: Optional[float] = None,
#     cu_seqlens: Optional[torch.LongTensor] = None,
#     chunk_size: int = 64
# ) -> torch.Tensor:
#     B, T, H, K= q.shape
#     BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
#     chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
#     NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)


#     o = torch.empty_like(v)
#     def grid(meta): return (triton.cdiv(V, meta['BV']), NT, B * H)
#     prepare_scale_chunk_fwd_kernel[grid](
#         q=q,
#         k=k,
#         v=v,
#         h=h,
#         g=g,
#         g_gamma=g_gamma,
#         o=o,
#         cu_seqlens=cu_seqlens,
#         chunk_indices=chunk_indices,
#         scale=scale,
#         T=T,
#         H=H,
#         K=K,
#         BT=BT,
#     )
#     return o