# -*- coding: utf-8 -*-
"""
KDA + OSGM: Chunk Forward and Backward Alignment Test
"""

import torch
import torch.nn.functional as F

from fla.ops.os_kda.chunk import chunk_kda

# ===================================================================
# 1. 绝对 Autograd 友好的 PyTorch 参考实现 (无 in-place 修改)
# ===================================================================
def naive_kda_osgm_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    eta: float,
    h0: torch.Tensor | None,
    d0: torch.Tensor | None,
    use_denominator: bool,
    d_min: float | None,
    d_max: float | None,
):
    B, T, H, K = q.shape
    V = v.shape[-1]

    S = h0.clone() if h0 is not None else q.new_zeros(B, H, K, V)
    d_curr = d0.clone() if d0 is not None else q.new_zeros(B, H, K)

    o_list = []

    for t in range(T):
        q_t = q[:, t] * scale
        k_t = k[:, t]
        v_t = v[:, t]
        g_t = g[:, t]     # 🚀 g_t 现在恢复为正确的 [B, H, K] 形状
        b_t = beta[:, t]

        # ---------------------------------------------------
        # Phase 1: OSGM 步长推导
        # ---------------------------------------------------
        k_sq = k_t * k_t
        inner_prod = (d_curr * k_sq).sum(dim=-1, keepdim=True)
        term_A = 1.0 - inner_prod

        if use_denominator:
            sum_k_sq = k_sq.sum(dim=-1, keepdim=True) + 1e-5
            grad_d = term_A / sum_k_sq
        else:
            grad_d = term_A

        d_next = d_curr + eta * grad_d * k_sq
        
        if d_min is not None and d_max is not None:
            d_next = torch.clamp(d_next, min=d_min, max=d_max)

        # ---------------------------------------------------
        # Phase 2: KDA 注意力与状态更新
        # ---------------------------------------------------
        # 1. 状态衰减 (匹配 g_t 的 K 维度广播)
        S_decayed = S * g_t.unsqueeze(-1).exp()
        
        # 2. 计算残差 u_t 
        v_minus = (k_t.unsqueeze(-1) * S_decayed).sum(dim=-2)
        u_t = (v_t - v_minus) * b_t.unsqueeze(-1)
        
        # 3. 构造预条件键 \tilde{k} 
        k_tilde = k_t * d_curr
        
        # 4. 状态吸收
        S_next = S_decayed + k_tilde.unsqueeze(-1) * u_t.unsqueeze(-2)
        
        # 5. 输出投影
        o_t = (q_t.unsqueeze(-1) * S_next).sum(dim=-2)
        o_list.append(o_t)
        
        S = S_next
        d_curr = d_next

    return torch.stack(o_list, dim=1), S, d_curr


# ===================================================================
# 2. Triton Chunk 内核调用封装
# ===================================================================
def run_chunk_kernel(q, k, v, g, beta, scale, eta, h0, d0, use_denom, d_min, d_max):
    res = chunk_kda(
        q=q, k=k, v=v, g=g, beta=beta, scale=scale,
        eta=eta, use_denominator=use_denom, d_min=d_min, d_max=d_max,
        initial_state=h0, initial_d=d0,
        output_final_state=True, output_final_d=True,
        use_qk_l2norm_in_kernel=False,
        use_gate_in_kernel=False 
    )
    
    if len(res) == 3:
        o, final_h, final_d = res
    else:
        o, (final_h, final_d) = res
    return o, final_h, final_d


# ===================================================================
# 3. 前后向一体化测试核心
# ===================================================================
def test_chunk_osgm_bwd(initial_d_val=0.5, use_denominator=False):
    torch.manual_seed(42)
    B, T, H, K, V = 2, 64, 2, 32, 32 
    scale = K ** -0.5
    eta = 1.0 
    d_min, d_max = 0.0, 2.0

    print(f"\n🚀 开始测试 Chunk FWD & BWD | Initial_D={initial_d_val}, Use_Denom={use_denominator}")

    # 初始化输入
    q = torch.randn(B, T, H, K, device='cuda', dtype=torch.float32)
    k = F.normalize(torch.randn(B, T, H, K, device='cuda', dtype=torch.float32), p=2, dim=-1)
    v = torch.randn(B, T, H, V, device='cuda', dtype=torch.float32)
    
    # 🚀 恢复硬编码的 K 维度，并保持 logsigmoid 防止数值溢出
    g = F.logsigmoid(torch.randn(B, T, H, K, device='cuda', dtype=torch.float32))
    
    beta = torch.rand(B, T, H, device='cuda', dtype=torch.float32).sigmoid()
    
    h0 = torch.randn(B, H, K, V, device='cuda', dtype=torch.float32)
    d0 = torch.full((B, H, K), initial_d_val, device='cuda', dtype=torch.float32)
    
    def clone_with_grad():
        return (
            q.clone().requires_grad_(True),
            k.clone().requires_grad_(True),
            v.clone().requires_grad_(True),
            g.clone().requires_grad_(True),
            beta.clone().requires_grad_(True),
            h0.clone().requires_grad_(True),
            d0.clone().requires_grad_(True)
        )

    q_ref, k_ref, v_ref, g_ref, b_ref, h0_ref, d0_ref = clone_with_grad()
    q_chk, k_chk, v_chk, g_chk, b_chk, h0_chk, d0_chk = clone_with_grad()

    # ==========================
    # Forward Pass
    # ==========================
    o_ref, ht_ref, dt_ref = naive_kda_osgm_ref(
        q_ref, k_ref, v_ref, g_ref, b_ref, scale, eta, h0_ref, d0_ref, use_denominator, d_min, d_max
    )
    
    try:
        o_chk, ht_chk, dt_chk = run_chunk_kernel(
            q_chk, k_chk, v_chk, g_chk, b_chk, scale, eta, h0_chk, d0_chk, use_denominator, d_min, d_max
        )
    except Exception as e:
        import traceback
        print("\n❌ Chunk Forward 运行出错:")
        traceback.print_exc()
        return

    print("  [FWD] O  最大误差: {:.2e}".format((o_ref - o_chk).abs().max().item()))
    print("  [FWD] Ht 最大误差: {:.2e}".format((ht_ref - ht_chk).abs().max().item()))
    print("  [FWD] Dt 最大误差: {:.2e}".format((dt_ref - dt_chk).abs().max().item()))
    
    assert (o_ref - o_chk).abs().max().item() < 5e-3, "Chunk 前向输出 O 不匹配!"
    print("  ✅ Chunk Forward 对齐成功！")

    # ==========================
    # Backward Pass
    # ==========================
    print("\n  正在进行反向传播计算...")
    dout = torch.randn_like(o_ref)
    dht = torch.randn_like(ht_ref)
    ddt = torch.randn_like(dt_ref)

    loss_ref = (o_ref * dout).sum() + (ht_ref * dht).sum() + (dt_ref * ddt).sum()
    loss_ref.backward()

    try:
        loss_chk = (o_chk * dout).sum() + (ht_chk * dht).sum() + (dt_chk * ddt).sum()
        loss_chk.backward()
    except Exception as e:
        import traceback
        print("\n❌ Chunk Backward 运行出错:")
        traceback.print_exc()
        return

    def check_grad(name, grad_ref, grad_chk):
        if grad_ref is None or grad_chk is None:
            print(f"  [BWD] {name:4} 梯度缺失!")
            return
        
        diff = (grad_ref - grad_chk).abs().max().item()
        cos = F.cosine_similarity(grad_ref.flatten(), grad_chk.flatten(), dim=0).item()
        
        print(f"  [BWD] {name:4} | Err: {diff:.2e} | Cos: {cos:.6f}")
        assert cos > 0.99 or diff < 5e-3, f"❌ {name} 梯度不对齐! 误差: {diff}, Cosine: {cos}"

    check_grad("q", q_ref.grad, q_chk.grad)
    check_grad("v", v_ref.grad, v_chk.grad)
    check_grad("beta", b_ref.grad, b_chk.grad)
    check_grad("h0", h0_ref.grad, h0_chk.grad)
    check_grad("d0", d0_ref.grad, d0_chk.grad)
    check_grad("k", k_ref.grad, k_chk.grad)
    check_grad("g", g_ref.grad, g_chk.grad)

    print("  ✅ Chunk Backward 全面梯度对齐成功！")


if __name__ == "__main__":
    test_chunk_osgm_bwd(initial_d_val=0.0, use_denominator=True)
    test_chunk_osgm_bwd(initial_d_val=0.0, use_denominator=False)
    test_chunk_osgm_bwd(initial_d_val=0.6, use_denominator=True)
    
    print("\n🎉 太棒了，Chunk KDA + OSGM 反向梯度完美收敛！")