import torch

def naive_recurrent_kda_osgm(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    eta: float = 1.0,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
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

    # 预条件步长 d_t 初始化为 0
    d_curr = k.new_zeros(B, H, K).to(q)

    o = torch.zeros_like(v)
    for i in range(0, T):
        # 注意：这里的 g_i 形状是 (B, H, K)
        q_i, k_i, v_i, g_i, b_i = q[:, i], k[:, i], v[:, i], g[:, i], beta[:, i]

        # ==========================================
        # 1. 在线计算 OSGM 的预条件向量 d_t
        # ==========================================
        s = k_i * k_i
        term_A = 1.0 - (d_curr * s).sum(dim=-1, keepdim=True)
        
        if use_denominator:
            grad_d = term_A / (s.sum(dim=-1, keepdim=True) + 1e-5)
        else:
            grad_d = term_A
        
        d_next = d_curr + eta * grad_d * s
        if d_min is not None and d_max is not None:
            d_next = torch.clamp(d_next, min=d_min, max=d_max)

        # ==========================================
        # 2. KDA 状态更新
        # ==========================================
        # (a) 状态衰减: 修正广播维度，g_i[..., None] 变成 (B, H, K, 1)，可覆盖 V 维度
        S = S * g_i[..., None].exp()
        
        # (b) 计算残差 u_t = (v_t - S_{t-1} k_t) * beta
        v_minus = (k_i[..., None] * S).sum(dim=-2)
        u_i = (v_i - v_minus) * b_i[..., None]
        
        # (c) 预条件键向量: \tilde{k}_t = d_t \odot k_t
        k_tilde = k_i * d_curr
        
        # (d) 预条件状态吸收: S = S + \tilde{k}_t \otimes u_t
        S = S + torch.einsum('b h k, b h v -> b h k v', k_tilde, u_i)
        
        # 前推 d_curr
        d_curr = d_next

        # ==========================================
        # 3. 输出提取
        # ==========================================
        o[:, i] = torch.einsum('b h k, b h k v -> b h v', q_i, S)
        
    if not output_final_state:
        S = None
    return o.to(dtype), S