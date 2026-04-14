# -*- coding: utf-8 -*-
"""
KDA + OSGM: Reference vs Recurrent vs Chunk Forward Alignment Test
"""

import torch
import torch.nn.functional as F

# ===================================================================
# 1. 纯 PyTorch 实现的 KDA + OSGM (Naive Reference)
# ===================================================================
def naive_recurrent_kda_osgm(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    eta: float = 1.0,
    initial_state: torch.Tensor | None = None,
    d0: torch.Tensor | None = None,  # 新增: initial_d
    use_denominator: bool = False,
    d_min: float | None = 0.0,
    d_max: float | None = 2.0,
):
    dtype = v.dtype
    B, T, H, K, V = *q.shape, v.shape[-1]
    if scale is None:
        scale = K ** -0.5

    q, k, v, g, beta = map(lambda x: x.to(torch.float), [q, k, v, g, beta])
    q = q * scale

    S = k.new_zeros(B, H, K, V).to(q)
    if initial_state is not None:
        S += initial_state.to(torch.float)

    # OSGM 状态初始化
    d_curr = d0.clone() if d0 is not None else k.new_zeros(B, H, K).to(q)

    o = torch.zeros_like(v)
    for i in range(0, T):
        q_i, k_i, v_i, g_i, b_i = q[:, i], k[:, i], v[:, i], g[:, i], beta[:, i]

        # Phase 1: OSGM Scale Prediction
        s = k_i * k_i
        term_A = 1.0 - (d_curr * s).sum(dim=-1, keepdim=True)
        
        if use_denominator:
            grad_d = term_A / (s.sum(dim=-1, keepdim=True) + 1e-5)
        else:
            grad_d = term_A
        
        d_next = d_curr + eta * grad_d * s
        if d_min is not None and d_max is not None:
            d_next = torch.clamp(d_next, min=d_min, max=d_max)

        # Phase 2: KDA Update
        S = S * g_i[..., None].exp()
        
        v_minus = (k_i[..., None] * S).sum(dim=-2)
        u_i = (v_i - v_minus) * b_i[..., None]
        
        k_tilde = k_i * d_curr
        
        S = S + torch.einsum('b h k, b h v -> b h k v', k_tilde, u_i)
        d_curr = d_next

        o[:, i] = torch.einsum('b h k, b h k v -> b h v', q_i, S)
        
    return o.to(dtype), S, d_curr


# ===================================================================
# 2. Triton Kernels 接入
# ===================================================================
# 请确保这三个引入路径正确
from fla.ops.os_kda.fused_recurrent import fused_recurrent_kda
from fla.ops.os_kda.chunk import chunk_kda

def run_recurrent_kernel(q, k, v, g, beta, scale, eta, h0, d0, use_denom, d_min, d_max):
    # 假设底层的 fused_recurrent_kda 已经支持了你之前修改的 OSGM 参数
    res = fused_recurrent_kda(
        q=q, k=k, v=v, g=g, beta=beta, scale=scale, eta=eta,
        initial_state=h0, initial_d=d0,
        output_final_state=True, output_final_d=True,
        use_qk_l2norm_in_kernel=False,
        use_denominator=use_denom, d_min=d_min, d_max=d_max
    )
    # 注意根据你的实际算子返回值调整
    if len(res) == 3:
        o, final_h, final_d = res
    else:
        o, (final_h, final_d) = res
    return o, final_h, final_d


def run_chunk_kernel(q, k, v, g, beta, scale, eta, h0, d0, use_denom, d_min, d_max):
    res = chunk_kda(
        q=q, k=k, v=v, g=g, beta=beta, scale=scale, eta=eta,
        initial_state=h0, initial_d=d0,
        output_final_state=True, output_final_d=True,
        use_qk_l2norm_in_kernel=False,
        use_denominator=use_denom, d_min=d_min, d_max=d_max
    )
    o, final_h, final_d = res
    return o, final_h, final_d


# ===================================================================
# 3. 核心测试函数 (仅测试 FWD)
# ===================================================================
def test_kda_osgm_fwd(initial_d_val=0.5, use_denominator=False):
    torch.manual_seed(42)
    B, T, H, K, V = 2, 64, 2, 32, 32  # 调大 T 以便测试跨 chunk 行为
    scale = K ** -0.5
    eta = 1.0 
    d_min, d_max = 0.0, 2.0

    print(f"\n🚀 测试配置: Initial_D={initial_d_val}, Use_Denom={use_denominator}")

    # 输入数据
    q = torch.randn(B, T, H, K, device='cuda', dtype=torch.float32)
    k = F.normalize(torch.randn(B, T, H, K, device='cuda', dtype=torch.float32), p=2, dim=-1)
    v = torch.randn(B, T, H, V, device='cuda', dtype=torch.float32)
    g = torch.randn(B, T, H, K, device='cuda', dtype=torch.float32) # KDA 独有的 gating
    beta = torch.rand(B, T, H, device='cuda', dtype=torch.float32).sigmoid()
    
    h0 = torch.randn(B, H, K, V, device='cuda', dtype=torch.float32)
    d0 = torch.full((B, H, K), initial_d_val, device='cuda', dtype=torch.float32)
    
    def clone_inputs():
        return (
            q.clone(), k.clone(), v.clone(), g.clone(), beta.clone(),
            h0.clone(), d0.clone()
        )

    # 三路克隆
    q_ref, k_ref, v_ref, g_ref, b_ref, h0_ref, d0_ref = clone_inputs()
    q_rec, k_rec, v_rec, g_rec, b_rec, h0_rec, d0_rec = clone_inputs()
    q_chk, k_chk, v_chk, g_chk, b_chk, h0_chk, d0_chk = clone_inputs()

    # 1. Reference (KDA OSGM)
    o_ref, ht_ref, dt_ref = naive_recurrent_kda_osgm(
        q_ref, k_ref, v_ref, g_ref, b_ref, scale, eta, h0_ref, d0_ref, use_denominator, d_min, d_max
    )
    
    try:
        # 2. Recurrent
        o_rec, ht_rec, dt_rec = run_recurrent_kernel(
            q_rec, k_rec, v_rec, g_rec, b_rec, scale, eta, h0_rec, d0_rec, use_denominator, d_min, d_max
        )
        
        # 3. Chunk
        o_chk, ht_chk, dt_chk = run_chunk_kernel(
            q_chk, k_chk, v_chk, g_chk, b_chk, scale, eta, h0_chk, d0_chk, use_denominator, d_min, d_max
        )
        
        # 误差计算
        print(f"  [Recurrent] O  最大误差: {(o_ref - o_rec).abs().max().item():.2e}")
        print(f"  [Recurrent] Ht 最大误差: {(ht_ref - ht_rec).abs().max().item():.2e}")
        print(f"  [Recurrent] Dt 最大误差: {(dt_ref - dt_rec).abs().max().item():.2e}")
        
        print(f"  [Chunk]     O  最大误差: {(o_ref - o_chk).abs().max().item():.2e}")
        print(f"  [Chunk]     Ht 最大误差: {(ht_ref - ht_chk).abs().max().item():.2e}")
        print(f"  [Chunk]     Dt 最大误差: {(dt_ref - dt_chk).abs().max().item():.2e}")

        # 阈值断言
        assert (o_ref - o_rec).abs().max().item() < 1e-3, "Recurrent 输出 O 不匹配!"
        assert (o_ref - o_chk).abs().max().item() < 1e-3, "Chunk 输出 O 不匹配!"
        assert (ht_ref - ht_chk).abs().max().item() < 1e-3, "Chunk 最终状态 Ht 不匹配!"
        assert (dt_ref - dt_chk).abs().max().item() < 1e-3, "Chunk 最终状态 Dt 不匹配!"
        
        print("  ✅ 前向传播完美对齐！")
        
    except Exception as e:
        import traceback
        print("\n❌ 运行出错:")
        traceback.print_exc()

if __name__ == "__main__":
    # 场景 1: 标准配置（初始 D 为 0）
    test_kda_osgm_fwd(initial_d_val=0.0, use_denominator=False)
    
    # 场景 2: 初始 D 不为 0
    test_kda_osgm_fwd(initial_d_val=0.5, use_denominator=False)
    
    # 场景 3: 开启 Denominator 归一化
    test_kda_osgm_fwd(initial_d_val=0.8, use_denominator=True)
    
    print("\n🎉 所有场景的前向测试均已完成！")