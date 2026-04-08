import torch
import torch.nn.functional as F

# ===================================================================
# 1. 纯 PyTorch 实现的 OSGM Delta Rule (Reference)
# ===================================================================
def reference_osgm_delta_rule(
    q, k, v, beta, scale, 
    eta=1.0, 
    h0=None, 
    d0=None,  # 新增：支持初始 d 状态
    use_denominator=False, 
    d_min=0.0, 
    d_max=2.0,
    use_l2norm=False
):
    if use_l2norm:
        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)

    B, T, H, K = q.shape
    V = v.shape[-1]
    
    # 状态初始化
    h = h0.clone().transpose(-1, -2) if h0 is not None else torch.zeros(B, H, V, K, device=q.device, dtype=q.dtype)
    d_curr = d0.clone() if d0 is not None else torch.zeros(B, H, K, device=q.device, dtype=q.dtype)
    
    o = torch.zeros_like(v)
    d_history = torch.zeros_like(k)
    
    # Phase 1: OSGM Scale Prediction (Supports Initial d0)
    for t in range(T):
        k_t = k[:, t] 
        d_history[:, t] = d_curr # 存储当前步使用的缩放
        
        k_sq = k_t ** 2
        inner_prod = (d_curr * k_sq).sum(dim=-1, keepdim=True)
        term_A = 1.0 - inner_prod
        
        if use_denominator:
            k_sq_sum = k_sq.sum(dim=-1, keepdim=True) + 1e-5
            grad_d = term_A / k_sq_sum
        else:
            grad_d = term_A
            
        d_curr = d_curr + eta * grad_d * k_sq

        if d_min is not None and d_max is not None:
            d_curr = torch.clamp(d_curr, min=d_min, max=d_max)

    # Phase 2: Standard Delta Rule
    k_osgm = k * d_history 
    
    for t in range(T):
        q_t, k_t, k_osgm_t, v_t = q[:, t] * scale, k[:, t], k_osgm[:, t], v[:, t]
        beta_t = beta[:, t].unsqueeze(-1)  # [B, H, 1]
        
        v_minus = (h @ k_t.unsqueeze(-1)).squeeze(-1)
        u_t = v_t - v_minus
        tilde_u_t = u_t * beta_t
        
        h = h + tilde_u_t.unsqueeze(-1) @ k_osgm_t.unsqueeze(-2)
        o_t = (h @ q_t.unsqueeze(-1)).squeeze(-1)
        o[:, t] = o_t
        
    return o, h.transpose(-1, -2), d_curr


# ===================================================================
# 2. Triton Kernels 接入
# ===================================================================
# 请确保此处路径与您的工程目录一致
from fla.ops.os_delta_rule.fused_recurrent_osgm import fused_recurrent_delta_rule_osgm

def run_recurrent_kernel(q, k, v, beta, scale, eta, h0, d0, use_denom, d_min, d_max, use_l2norm):
    # 注意：recurrent 算子内部也需要支持 initial_state=(h0, d0)
    res = fused_recurrent_delta_rule_osgm(
        q=q, k=k, v=v, beta=beta, scale=scale, eta=eta, 
        initial_state=(h0, d0), output_final_state=True,
        use_qk_l2norm_in_kernel=use_l2norm,
        use_denominator=use_denom, d_min=d_min, d_max=d_max
    )
    # 根据您的算子实际返回值进行解包
    o, (final_h, final_d) = res
    return o, final_h, final_d


# ===================================================================
# 3. 核心测试函数 (仅测试 FWD)
# ===================================================================
def test_recurrent_osgm_fwd(use_l2norm=False, initial_d_val=0.5):
    torch.manual_seed(42)
    B, T, H, K, V = 2, 16, 2, 32, 32 
    scale = K ** -0.5
    eta = 1.0 
    use_denominator = True if not use_l2norm else False # 不做l2norm时建议开启分母
    d_min, d_max = 0.0, 2.0

    print(f"\n🚀 测试配置: L2Norm={use_l2norm}, Initial_D={initial_d_val}, Use_Denom={use_denominator}")

    # 输入数据
    q = torch.randn(B, T, H, K, device='cuda', dtype=torch.float32)
    k = torch.randn(B, T, H, K, device='cuda', dtype=torch.float32)
    v = torch.randn(B, T, H, V, device='cuda', dtype=torch.float32)
    beta = torch.rand(B, T, H, device='cuda', dtype=torch.float32).sigmoid()
    
    h0 = torch.randn(B, H, K, V, device='cuda', dtype=torch.float32)
    d0 = torch.full((B, H, K), initial_d_val, device='cuda', dtype=torch.float32)
    
    def clone_inputs():
        return (
            q.clone(),
            k.clone(),
            v.clone(),
            beta.clone(),
            h0.clone(),
            d0.clone()
        )

    # 两路克隆
    q_ref, k_ref, v_ref, b_ref, h0_ref, d0_ref = clone_inputs()
    q_rec, k_rec, v_rec, b_rec, h0_rec, d0_rec = clone_inputs()

    # 1. Reference
    o_ref, ht_ref, dt_ref = reference_osgm_delta_rule(
        q_ref, k_ref, v_ref, b_ref, scale, eta, h0_ref, d0_ref, use_denominator, d_min, d_max, use_l2norm
    )
    
    # 2. Recurrent
    try:
        o_rec, ht_rec, dt_rec = run_recurrent_kernel(
            q_rec, k_rec, v_rec, b_rec, scale, eta, h0_rec, d0_rec, use_denominator, d_min, d_max, use_l2norm
        )
        
        # 误差计算
        err_o = (o_ref - o_rec).abs().max().item()
        err_ht = (ht_ref - ht_rec).abs().max().item()
        err_dt = (dt_ref - dt_rec).abs().max().item()
        
        print(f"  [FWD] O  最大误差: {err_o:.2e}")
        print(f"  [FWD] Ht 最大误差: {err_ht:.2e}")
        print(f"  [FWD] Dt 最大误差: {err_dt:.2e}")

        # 阈值断言
        assert err_o < 1e-3, f"输出 O 不匹配! 误差: {err_o}"
        assert err_ht < 1e-3, f"最终状态 Ht 不匹配! 误差: {err_ht}"
        assert err_dt < 1e-3, f"最终状态 Dt 不匹配! 误差: {err_dt}"
        print("  ✅ 前向传播完美对齐！")
        
    except Exception as e:
        import traceback
        print("\n❌ 运行出错:")
        traceback.print_exc()

if __name__ == "__main__":
    # 场景 1: 标准配置（带 L2 Norm，初始 D 为 0）
    test_recurrent_osgm_fwd(use_l2norm=True, initial_d_val=0.0)
    
    # 场景 2: 极端配置（不带 L2 Norm，初始 D 较大，验证稳定性）
    test_recurrent_osgm_fwd(use_l2norm=False, initial_d_val=0.8)
    
    # 场景 3: 验证随机初始 D 的连续性
    test_recurrent_osgm_fwd(use_l2norm=True, initial_d_val=1.2)
    
    print("\n🎉 所有场景的前向测试均已完成！")