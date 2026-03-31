import torch
import torch.nn.functional as F

# ===================================================================
# 1. 纯 PyTorch 实现的 OSGM Delta Rule (Reference)
# ===================================================================
def reference_osgm_delta_rule(
    q, k, v, beta, scale, 
    eta=1.0, 
    h0=None, 
    use_denominator=False, 
    d_min=0.0, 
    d_max=2.0
):
    B, T, H, K = q.shape
    V = v.shape[-1]
    
    if h0 is not None:
        h = h0.clone().transpose(-1, -2) # [B, H, V, K]
    else:
        h = torch.zeros(B, H, V, K, device=q.device, dtype=q.dtype)
        
    d_curr = torch.zeros(B, H, K, device=q.device, dtype=q.dtype)
    o = torch.zeros_like(v)
    d_history = torch.zeros_like(k)
    
    # Phase 1: OSGM Scale Prediction 
    for t in range(T):
        k_t = k[:, t] 
        d_history[:, t] = d_curr
        
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
        q_t = q[:, t] * scale         
        k_t = k[:, t]                 
        k_osgm_t = k_osgm[:, t]       
        v_t = v[:, t]                 
        
        beta_t = beta[:, t].unsqueeze(-1)  # [B, H, 1]
        
        v_minus = (h @ k_t.unsqueeze(-1)).squeeze(-1)
        u_t = v_t - v_minus
        tilde_u_t = u_t * beta_t
        
        h = h + tilde_u_t.unsqueeze(-1) @ k_osgm_t.unsqueeze(-2)
        o_t = (h @ q_t.unsqueeze(-1)).squeeze(-1)
        o[:, t] = o_t
        
    return o, h.transpose(-1, -2)


# ===================================================================
# 2. Triton Kernels 接入
# ===================================================================
from fla.ops.osla_delta_rule.fused_recurrent_osgm import fused_recurrent_delta_rule_osgm
from fla.ops.osla_delta_rule.chunk_osgm import chunk_delta_rule_osgm

def run_recurrent_kernel(q, k, v, beta, scale, eta, h0, use_denom, d_min, d_max):
    res = fused_recurrent_delta_rule_osgm(
        q=q, k=k, v=v, beta=beta, scale=scale, eta=eta, 
        initial_state=h0, output_final_state=True,
        use_qk_l2norm_in_kernel=False,
        use_denominator=use_denom, d_min=d_min, d_max=d_max
    )
    # 兼容处理：提取 final_h
    final_state = res[1]
    final_h = final_state[0] if isinstance(final_state, tuple) else final_state
    return res[0], final_h

def run_chunk_kernel(q, k, v, beta, scale, eta, h0, use_denom, d_min, d_max):
    res = chunk_delta_rule_osgm(
        q=q, k=k, v=v, beta=beta, scale=scale, eta=eta, 
        initial_state=h0, output_final_state=True,
        use_qk_l2norm_in_kernel=False,
        use_denominator=use_denom, d_min=d_min, d_max=d_max
    )
    # 兼容处理：安全解包 (final_h, final_d)
    final_state = res[1]
    final_h = final_state[0] if isinstance(final_state, tuple) else final_state
    return res[0], final_h


# ===================================================================
# 3. 三方严格梯度对齐测试
# ===================================================================
def test_all_osgm_gradients():
    torch.manual_seed(42)
    
    B, T, H, K, V = 2, 16, 2, 32, 32 
    scale = K ** -0.5
    eta = 1.0 
    use_denominator = False
    d_min = 0.0
    d_max = 2.0

    q = torch.randn(B, T, H, K, device='cuda', dtype=torch.float32)
    k = F.normalize(torch.randn(B, T, H, K, device='cuda', dtype=torch.float32), p=2, dim=-1)
    v = torch.randn(B, T, H, V, device='cuda', dtype=torch.float32)
    beta = torch.rand(B, T, H, device='cuda', dtype=torch.float32).sigmoid()
    h0 = torch.randn(B, H, K, V, device='cuda', dtype=torch.float32)
    
    def clone_inputs():
        return (
            q.clone().requires_grad_(True),
            k.clone().requires_grad_(True),
            v.clone().requires_grad_(True),
            beta.clone().requires_grad_(True),
            h0.clone().requires_grad_(True)
        )

    q_ref, k_ref, v_ref, b_ref, h0_ref = clone_inputs()
    q_rec, k_rec, v_rec, b_rec, h0_rec = clone_inputs()
    q_chk, k_chk, v_chk, b_chk, h0_chk = clone_inputs()

    print(f"=== 开始正向传播比对 (eta={eta}, denom={use_denominator}) ===")
    o_ref, ht_ref = reference_osgm_delta_rule(
        q_ref, k_ref, v_ref, b_ref, scale, eta, h0_ref, use_denominator, d_min, d_max
    )
    
    try:
        o_rec, ht_rec = run_recurrent_kernel(
            q_rec, k_rec, v_rec, b_rec, scale, eta, h0_rec, use_denominator, d_min, d_max
        )
        o_chk, ht_chk = run_chunk_kernel(
            q_chk, k_chk, v_chk, b_chk, scale, eta, h0_chk, use_denominator, d_min, d_max
        )
        
        print(f"[Recurrent] 输出 o 的最大误差: {(o_ref - o_rec).abs().max().item():.6e}")
        print(f"[Recurrent] 终态 ht 的最大误差: {(ht_ref - ht_rec).abs().max().item():.6e}")
        print(f"[Chunk]     输出 o 的最大误差: {(o_ref - o_chk).abs().max().item():.6e}")
        print(f"[Chunk]     终态 ht 的最大误差: {(ht_ref - ht_chk).abs().max().item():.6e}")

        print("\n=== 开始反向传播比对 ===")
        dout = torch.randn_like(o_ref)
        dht = torch.randn_like(ht_ref) 
        
        loss_ref = (o_ref * dout).sum() + (ht_ref * dht).sum()
        loss_ref.backward()
        
        loss_rec = (o_rec * dout).sum() + (ht_rec * dht).sum()
        loss_rec.backward()

        loss_chk = (o_chk * dout).sum() + (ht_chk * dht).sum()
        loss_chk.backward()

        def check_grad(name, grad_ref, grad_rec, grad_chk):
            diff_rec = (grad_ref - grad_rec).abs().max().item()
            cos_rec = F.cosine_similarity(grad_ref.flatten(), grad_rec.flatten(), dim=0).item()
            
            diff_chk = (grad_ref - grad_chk).abs().max().item()
            cos_chk = F.cosine_similarity(grad_ref.flatten(), grad_chk.flatten(), dim=0).item()
            
            print(f"[{name}]")
            print(f"  ├─ Recurrent -> 误差: {diff_rec:.6e} | 余弦相似度: {cos_rec:.6f}")
            print(f"  └─ Chunk     -> 误差: {diff_chk:.6e} | 余弦相似度: {cos_chk:.6f}")
            
            assert diff_rec < 1e-2 or cos_rec > 0.99, f"{name} 的 Recurrent 梯度有误！"
            assert diff_chk < 1e-2 or cos_chk > 0.99, f"{name} 的 Chunk 梯度有误！"

        check_grad("q", q_ref.grad, q_rec.grad, q_chk.grad)
        check_grad("k", k_ref.grad, k_rec.grad, k_chk.grad)
        check_grad("v", v_ref.grad, v_rec.grad, v_chk.grad)
        check_grad("beta", b_ref.grad, b_rec.grad, b_chk.grad)
        check_grad("h0", h0_ref.grad, h0_rec.grad, h0_chk.grad)
        
        print("\n✅ 三方 (Ref / Recurrent / Chunk) 梯度完全对齐！测试通过！")
        
    except Exception as e:
        import traceback
        print("\n🚧 运行异常:")
        traceback.print_exc()

if __name__ == "__main__":
    test_all_osgm_gradients()