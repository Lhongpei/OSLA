import torch
import math

def naive_recurrent_kda_osgm(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    eta: float = 1.0,
    initial_state: torch.Tensor | None = None,
    initial_d: torch.Tensor | None = None,
    output_final_state: bool = False,
    output_final_d: bool = False,
    use_denominator: bool = False,
    d_min: float | None = 0.0,
    d_max: float | None = 2.0,
    beta_aware: bool = False,
    decay_mode: str = "none",
    decay_gamma: float = 1.0,
    g_decay: torch.Tensor | None = None,
):
    dtype = v.dtype
    B, T, H, K, V = *q.shape, v.shape[-1]
    if scale is None:
        scale = K ** -0.5

    q, k, v, g, beta = map(lambda x: x.to(torch.float), [q, k, v, g, beta])
    if g_decay is not None:
        g_decay = g_decay.to(torch.float)
    q = q * scale

    S = k.new_zeros(B, H, K, V).to(q)
    if initial_state is not None:
        S += initial_state.to(torch.float)

    d_curr = k.new_zeros(B, H, K).to(q)
    if initial_d is not None:
        d_curr += initial_d.to(torch.float)

    o = torch.zeros_like(v)
    for i in range(0, T):
        # 注意：这里的 g_i 形状是 (B, H, K)
        q_i, k_i, v_i, g_i, b_i = q[:, i], k[:, i], v[:, i], g[:, i], beta[:, i]
        g_decay_i = g_decay[:, i] if g_decay is not None else None

        S = S * g_i[..., None].exp()

        v_minus = (k_i[..., None] * S).sum(dim=-2)
        u_i = (v_i - v_minus) * b_i[..., None]

        k_tilde = k_i * d_curr
        S = S + torch.einsum('b h k, b h v -> b h k v', k_tilde, u_i)

        o[:, i] = torch.einsum('b h k, b h k v -> b h v', q_i, S)

        s = k_i * k_i
        inner = (d_curr * s).sum(dim=-1, keepdim=True)
        if beta_aware:
            grad_d = b_i[..., None] * (1.0 - b_i[..., None] * inner)
        else:
            grad_d = 1.0 - inner
        if use_denominator:
            grad_d = grad_d / (s.sum(dim=-1, keepdim=True) + 1e-5)

        if decay_mode == "none":
            d_curr = d_curr + eta * grad_d * s
        elif decay_mode == "constant":
            d_curr = math.exp(math.log(float(decay_gamma))) * d_curr + eta * grad_d * s
        elif decay_mode == "data_dependent":
            if g_decay_i is None:
                raise ValueError("decay_mode='data_dependent' requires g_decay")
            d_curr = g_decay_i.exp()[..., None] * d_curr + eta * grad_d * s
        else:
            raise NotImplementedError(f"Unsupported OS-KDA d decay mode: {decay_mode}")
        if d_min is not None and d_max is not None:
            d_curr = torch.clamp(d_curr, min=d_min, max=d_max)
        
    if not output_final_state:
        S = None

    if output_final_d:
        return o.to(dtype), S, d_curr
    return o.to(dtype), S
