import torch
import torch.nn.functional as F


def reference_recurrent_delta_rule(q, k, v, beta, scale, h0=None, d0=None):
    B, T, H, K = q.shape
    V = v.shape[-1]
    
    if h0 is not None:
        h = h0.clone().transpose(-1, -2) # [B, H, V, K]
    else:
        h = torch.zeros(B, H, V, K, device=q.device, dtype=q.dtype)
        
    if d0 is not None:
        d = d0.clone() # [B, H, K]
    else:
        d = torch.zeros(B, H, K, device=q.device, dtype=q.dtype)
        
    o = torch.zeros_like(v)
    
    for t in range(T):
        q_t = q[:, t] * scale         # [B, H, K]
        k_t = k[:, t]                 # [B, H, K]
        v_t = v[:, t]                 # [B, H, V]
        beta_t = beta[:, t]           # [B, H, V]
        
        # 1. u_t = v_t - S_{t-1} k_t
        # h: [B, H, V, K], k_t: [B, H, K] -> [B, H, V]
        v_minus = (h @ k_t.unsqueeze(-1)).squeeze(-1)
        u_t = v_t - v_minus
        
        # 2. D_t = D_{t-1} + k_t^2
        d = d + k_t ** 2
        
        # 3. \tilde{u}_t = \beta_t * u_t
        tilde_u_t = u_t * beta_t
        
        # 4. p_t = k_t / (D_t + eps)
        p_t = k_t / (d + 1e-5)
        
        # 5. S_t = S_{t-1} + \tilde{u}_t p_t^T
        # 外积: [B, H, V, 1] @ [B, H, 1, K] -> [B, H, V, K]
        h = h + tilde_u_t.unsqueeze(-1) @ p_t.unsqueeze(-2)
        
        # 6. o_t = S_t q_t
        o_t = (h @ q_t.unsqueeze(-1)).squeeze(-1)
        o[:, t] = o_t
        
    return o, h.transpose(-1, -2) # 还原为 [B, H, K, V] 返回


def test_gradients():
    torch.manual_seed(42)
    
    B, T, H, K, V = 2, 16, 2, 32, 32 
    scale = K ** -0.5

    q = torch.randn(B, T, H, K, device='cuda', dtype=torch.float32)
    k = F.normalize(torch.randn(B, T, H, K, device='cuda', dtype=torch.float32), p=2, dim=-1)
    v = torch.randn(B, T, H, V, device='cuda', dtype=torch.float32)
    beta = torch.rand(B, T, H, V, device='cuda', dtype=torch.float32).sigmoid()
    h0 = torch.randn(B, H, K, V, device='cuda', dtype=torch.float32)
    
    q_ref = q.clone().requires_grad_(True)
    k_ref = k.clone().requires_grad_(True)
    v_ref = v.clone().requires_grad_(True)
    beta_ref = beta.clone().requires_grad_(True)
    h0_ref = h0.clone().requires_grad_(True)
    
    q_tri = q.clone().requires_grad_(True)
    k_tri = k.clone().requires_grad_(True)
    v_tri = v.clone().requires_grad_(True)
    beta_tri = beta.clone().requires_grad_(True)
    h0_tri = h0.clone().requires_grad_(True)

    o_ref, ht_ref = reference_recurrent_delta_rule(
        q_ref, k_ref, v_ref, beta_ref, scale, h0_ref, d0=None
    )
    
    o_tri, ht_tri = fused_recurrent_delta_rule(
        q_tri, k_tri, v_tri, beta_tri, scale, initial_state=h0_tri, output_final_state=True
    )
    
    print(f"输出 o 的最大误差: {(o_ref - o_tri).abs().max().item():.6f}")
    print(f"终态 ht 的最大误差: {(ht_ref - ht_tri).abs().max().item():.6f}")

    print("\n=== 开始反向传播比对 ===")
    dout = torch.randn_like(o_ref)
    
    loss_ref = (o_ref * dout).sum()
    loss_ref.backward()
    
    loss_tri = (o_tri * dout).sum()
    loss_tri.backward()

    # 比对梯度
    def check_grad(name, grad_ref, grad_tri):
        max_diff = (grad_ref - grad_tri).abs().max().item()
        cos_sim = F.cosine_similarity(grad_ref.flatten(), grad_tri.flatten(), dim=0).item()
        print(f"[{name}] 梯度最大误差: {max_diff:.6f} | 余弦相似度: {cos_sim:.6f}")
        assert max_diff < 1e-2 or cos_sim > 0.99, f"{name} 梯度可能有误！"

    check_grad("q", q_ref.grad, q_tri.grad)
    check_grad("k", k_ref.grad, k_tri.grad)
    check_grad("v", v_ref.grad, v_tri.grad)
    check_grad("beta", beta_ref.grad, beta_tri.grad)
    check_grad("h0", h0_ref.grad, h0_tri.grad)
    
    print("\n✅ 所有梯度验证通过！")

if __name__ == "__main__":
    test_gradients()