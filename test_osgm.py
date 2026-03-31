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
from fla.ops.osla_delta_rule.fused_recurrent_osgm import fused_recurrent_delta_rule_osgm
from fla.ops.osla_delta_rule.chunk_osgm import chunk_delta_rule_osgm

def run_recurrent_kernel(q, k, v, beta, scale, eta, h0, d0, use_denom, d_min, d_max, use_l2norm):
    # 注意：recurrent 算子内部也需要支持 initial_state=(h0, d0)
    res = fused_recurrent_delta_rule_osgm(
        q=q, k=k, v=v, beta=beta, scale=scale, eta=eta, 
        initial_state=(h0, d0), output_final_state=True,
        use_qk_l2norm_in_kernel=use_l2norm,
        use_denominator=use_denom, d_min=d_min, d_max=d_max
    )
    o, (final_h, final_d) = res
    return o, final_h, final_d

def run_chunk_kernel(q, k, v, beta, scale, eta, h0, d0, use_denom, d_min, d_max, use_l2norm):
    res = chunk_delta_rule_osgm(
        q=q, k=k, v=v, beta=beta, scale=scale, eta=eta, 
        initial_state=(h0, d0), output_final_state=True,
        use_qk_l2norm_in_kernel=use_l2norm,
        use_denominator=use_denom, d_min=d_min, d_max=d_max
    )
    o, (final_h, final_d) = res
    return o, final_h, final_d


# ===================================================================
# 3. 核心测试函数
# ===================================================================
def test_all_osgm_gradients(use_l2norm=False, initial_d_val=0.5):
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
            q.clone().requires_grad_(True),
            k.clone().requires_grad_(True),
            v.clone().requires_grad_(True),
            beta.clone().requires_grad_(True),
            h0.clone().requires_grad_(True),
            d0.clone().requires_grad_(True)
        )

    # 三路克隆
    q_ref, k_ref, v_ref, b_ref, h0_ref, d0_ref = clone_inputs()
    q_rec, k_rec, v_rec, b_rec, h0_rec, d0_rec = clone_inputs()
    q_chk, k_chk, v_chk, b_chk, h0_chk, d0_chk = clone_inputs()

    # 1. Reference
    o_ref, ht_ref, dt_ref = reference_osgm_delta_rule(
        q_ref, k_ref, v_ref, b_ref, scale, eta, h0_ref, d0_ref, use_denominator, d_min, d_max, use_l2norm
    )
    
    # 2. Recurrent
    o_rec, ht_rec, dt_rec = run_recurrent_kernel(
        q_rec, k_rec, v_rec, b_rec, scale, eta, h0_rec, d0_rec, use_denominator, d_min, d_max, use_l2norm
    )
    
    # 3. Chunk
    o_chk, ht_chk, dt_chk = run_chunk_kernel(
        q_chk, k_chk, v_chk, b_chk, scale, eta, h0_chk, d0_chk, use_denominator, d_min, d_max, use_l2norm
    )
    
    # 正向比对
    print(f"  [FWD] O 误差: Rec={ (o_ref-o_rec).abs().max():.2e}, Chk={ (o_ref-o_chk).abs().max():.2e}")
    print(f"  [FWD] Ht 误差: Rec={ (ht_ref-ht_rec).abs().max():.2e}, Chk={ (ht_ref-ht_chk).abs().max():.2e}")
    print(f"  [FWD] Dt 误差: Rec={ (dt_ref-dt_rec).abs().max():.2e}, Chk={ (dt_ref-dt_chk).abs().max():.2e}")

    # 反向比对
    dout = torch.randn_like(o_ref)
    dht = torch.randn_like(ht_ref) 
    ddt = torch.randn_like(dt_ref)

    loss_ref = (o_ref * dout).sum() + (ht_ref * dht).sum() + (dt_ref * ddt).sum()
    loss_ref.backward()
    
    loss_rec = (o_rec * dout).sum() + (ht_rec * dht).sum() + (dt_rec * ddt).sum()
    loss_rec.backward()

    loss_chk = (o_chk * dout).sum() + (ht_chk * dht).sum() + (dt_chk * ddt).sum()
    loss_chk.backward()

    def check_grad(name, g_ref, g_rec, g_chk):
        c_rec = F.cosine_similarity(g_ref.flatten(), g_rec.flatten(), dim=0).item()
        c_chk = F.cosine_similarity(g_ref.flatten(), g_chk.flatten(), dim=0).item()
        print(f"  [BWD] {name:4} Cosine: Rec={c_rec:.6f}, Chk={c_chk:.6f}")
        assert c_rec > 0.99 and c_chk > 0.99, f"{name} 梯度不对齐！"

    check_grad("q", q_ref.grad, q_rec.grad, q_chk.grad)
    check_grad("k", k_ref.grad, k_rec.grad, k_chk.grad)
    check_grad("v", v_ref.grad, v_rec.grad, v_chk.grad)
    check_grad("h0", h0_ref.grad, h0_rec.grad, h0_chk.grad)
    check_grad("d0", d0_ref.grad, d0_rec.grad, d0_chk.grad)

if __name__ == "__main__":
    # 场景 1: 标准配置（带 L2 Norm，初始 D 为 0）
    test_all_osgm_gradients(use_l2norm=True, initial_d_val=0.0)
    
    # 场景 2: 极端配置（不带 L2 Norm，初始 D 较大，验证稳定性）
    test_all_osgm_gradients(use_l2norm=False, initial_d_val=0.8)
    
    # 场景 3: 验证随机初始 D 的连续性
    test_all_osgm_gradients(use_l2norm=True, initial_d_val=1.2)
    
    print("\n✅ 所有场景三方对齐成功！")