import os

import torch
import torch.nn.functional as F

from fla.models.os_kda import OSKDAConfig, OSKDAForCausalLM
from fla.ops.os_delta_rule.chunk_osgm_phase_dd_decay_beta import compute_osgm_dd_decay_beta_phase1_fwd
from fla.ops.os_kda import chunk_kda


def naive_os_kda_beta_aware(q, k, v, g, beta, scale, eta, h0, d0, d_min, d_max):
    bsz, seqlen, n_heads, k_dim = q.shape
    v_dim = v.shape[-1]
    state = h0.float().clone()
    d = d0.float().clone()
    outs = []

    for t in range(seqlen):
        q_t = q[:, t].float() * scale
        k_t = k[:, t].float()
        v_t = v[:, t].float()
        g_t = g[:, t].float()
        beta_t = beta[:, t].float()

        state_bar = state * g_t[..., None].exp()
        residual = v_t - (k_t[..., None] * state_bar).sum(dim=-2)
        state = state_bar + (d * k_t).unsqueeze(-1) * (beta_t[..., None] * residual).unsqueeze(-2)
        outs.append((q_t.unsqueeze(-1) * state).sum(dim=-2))

        k_sq = k_t * k_t
        inner = (d * k_sq).sum(dim=-1, keepdim=True)
        grad_d = beta_t[..., None] * (1.0 - beta_t[..., None] * inner)
        d = d + eta * grad_d * k_sq
        if d_min is not None and d_max is not None:
            d = d.clamp(d_min, d_max)

    return torch.stack(outs, dim=1).to(v.dtype), state, d


def test_chunk_forward_backward():
    if not torch.cuda.is_available():
        print("CUDA is not available; skipping Triton OS-KDA test.")
        return
    if os.environ.get("OSKDA_RUN_FULL_CUDA", "0") != "1":
        print("Set OSKDA_RUN_FULL_CUDA=1 to run the full chunk forward/backward check.")
        return

    torch.manual_seed(7)
    device = "cuda"
    bsz, seqlen, n_heads, k_dim, v_dim = 1, 16, 1, 16, 16
    scale = k_dim ** -0.5
    eta = 1.0
    d_min, d_max = 0.0, 1e9

    q = torch.randn(bsz, seqlen, n_heads, k_dim, device=device, dtype=torch.float32)
    k = F.normalize(torch.randn_like(q), p=2, dim=-1)
    v = torch.randn(bsz, seqlen, n_heads, v_dim, device=device, dtype=torch.float32)
    g = F.logsigmoid(torch.randn(bsz, seqlen, n_heads, k_dim, device=device, dtype=torch.float32))
    beta = torch.randn(bsz, seqlen, n_heads, device=device, dtype=torch.float32).sigmoid()
    h0 = torch.randn(bsz, n_heads, k_dim, v_dim, device=device, dtype=torch.float32)
    d0 = torch.ones(bsz, n_heads, k_dim, device=device, dtype=torch.float32)

    def clone_with_grad():
        return (
            q.clone().requires_grad_(True),
            k.clone().requires_grad_(True),
            v.clone().requires_grad_(True),
            g.clone().requires_grad_(True),
            beta.clone().requires_grad_(True),
            h0.clone().requires_grad_(True),
            d0.clone().requires_grad_(True),
        )

    q_ref, k_ref, v_ref, g_ref, beta_ref, h0_ref, d0_ref = clone_with_grad()
    q_os, k_os, v_os, g_os, beta_os, h0_os, d0_os = clone_with_grad()

    o_ref, ht_ref, dt_ref = naive_os_kda_beta_aware(
        q_ref, k_ref, v_ref, g_ref, beta_ref, scale, eta, h0_ref, d0_ref, d_min, d_max
    )
    o_os, ht_os, dt_os = chunk_kda(
        q=q_os,
        k=k_os,
        v=v_os,
        g=g_os,
        beta=beta_os,
        scale=scale,
        eta=eta,
        use_denominator=False,
        d_min=d_min,
        d_max=d_max,
        beta_aware=True,
        initial_state=h0_os,
        initial_d=d0_os,
        output_final_state=True,
        output_final_d=True,
        use_qk_l2norm_in_kernel=False,
        use_gate_in_kernel=False,
    )

    print(f"chunk fwd max diff o={float((o_ref - o_os).abs().max()):.3e}")
    print(f"chunk fwd max diff h={float((ht_ref - ht_os).abs().max()):.3e}")
    print(f"chunk fwd max diff d={float((dt_ref - dt_os).abs().max()):.3e}")
    assert torch.allclose(o_ref, o_os, atol=5e-3, rtol=5e-3)
    assert torch.allclose(ht_ref, ht_os, atol=5e-3, rtol=5e-3)
    assert torch.allclose(dt_ref, dt_os, atol=5e-4, rtol=5e-4)

    do = torch.randn_like(o_ref)
    dht = torch.randn_like(ht_ref)
    ddt = torch.randn_like(dt_ref)
    (o_ref * do).sum().add((ht_ref * dht).sum()).add((dt_ref * ddt).sum()).backward()
    (o_os * do).sum().add((ht_os * dht).sum()).add((dt_os * ddt).sum()).backward()

    for name, ref, got in [
        ("q", q_ref.grad, q_os.grad),
        ("k", k_ref.grad, k_os.grad),
        ("v", v_ref.grad, v_os.grad),
        ("g", g_ref.grad, g_os.grad),
        ("beta", beta_ref.grad, beta_os.grad),
        ("h0", h0_ref.grad, h0_os.grad),
        ("d0", d0_ref.grad, d0_os.grad),
    ]:
        diff = float((ref - got).abs().max())
        cos = float(F.cosine_similarity(ref.flatten(), got.flatten(), dim=0))
        print(f"chunk bwd {name:>4s}: diff={diff:.3e}, cos={cos:.6f}")
        min_cos = 0.97 if name == "g" else 0.99
        assert cos > min_cos or diff < 5e-3


def test_model_smoke():
    if not torch.cuda.is_available():
        print("CUDA is not available; skipping model smoke test.")
        return
    if os.environ.get("OSKDA_RUN_FULL_CUDA", "0") != "1":
        print("Set OSKDA_RUN_FULL_CUDA=1 to run the CUDA model smoke test.")
        return

    torch.manual_seed(11)
    config = OSKDAConfig(
        vocab_size=128,
        hidden_size=64,
        num_hidden_layers=1,
        num_heads=2,
        head_dim=16,
        expand_v=1.0,
        use_short_conv=False,
        fuse_norm=False,
        fuse_swiglu=False,
        fuse_cross_entropy=False,
        use_cache=False,
        osgm_beta_aware=True,
    )
    model = OSKDAForCausalLM(config).cuda().train()
    input_ids = torch.randint(0, config.vocab_size, (1, 16), device="cuda")
    out = model(input_ids=input_ids, labels=input_ids)
    assert torch.isfinite(out.loss), out.loss
    out.loss.backward()
    nonzero = sum(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert nonzero > 0
    print(f"model smoke loss={float(out.loss):.4f}, grad_tensors={nonzero}")


def test_registration_and_construction():
    config = OSKDAConfig(
        vocab_size=128,
        hidden_size=64,
        num_hidden_layers=1,
        num_heads=2,
        head_dim=16,
        expand_v=1.0,
        use_short_conv=False,
        fuse_norm=False,
        fuse_swiglu=False,
        fuse_cross_entropy=False,
        use_cache=False,
    )
    model = OSKDAForCausalLM(config)
    layer = model.model.layers[0].attn
    assert layer.use_osgm
    assert layer.osgm_beta_aware
    assert tuple(layer.initial_scale.shape) == (2, 16)
    print("model construction ok: use_osgm=True, initial_scale=(2, 16)")


def test_beta_aware_phase1():
    if not torch.cuda.is_available():
        print("CUDA is not available; skipping beta-aware phase1 test.")
        return

    torch.manual_seed(123)
    bsz, seqlen, n_heads, k_dim = 1, 8, 1, 16
    eta = 1.0
    k = F.normalize(torch.randn(bsz, seqlen, n_heads, k_dim, device="cuda"), p=2, dim=-1)
    beta = torch.randn(bsz, seqlen, n_heads, device="cuda").sigmoid()
    g_decay = torch.zeros(bsz, seqlen, n_heads, device="cuda")
    d0 = torch.ones(bsz, n_heads, k_dim, device="cuda")

    d, final_d = compute_osgm_dd_decay_beta_phase1_fwd(
        k=k,
        g=g_decay,
        beta=beta,
        eta=eta,
        use_denominator=False,
        d_min=0.0,
        d_max=1e9,
        initial_d=d0,
        output_final_state=True,
    )

    ref = []
    d_curr = d0.clone()
    for t in range(seqlen):
        ref.append(d_curr)
        k_sq = k[:, t] * k[:, t]
        inner = (d_curr * k_sq).sum(dim=-1, keepdim=True)
        grad = beta[:, t, :, None] * (1.0 - beta[:, t, :, None] * inner)
        d_curr = (d_curr + eta * grad * k_sq).clamp(0.0, 1e9)
    ref = torch.stack(ref, dim=1)

    print(f"phase1 d diff={float((d - ref).abs().max()):.3e}")
    print(f"phase1 final diff={float((final_d - d_curr).abs().max()):.3e}")
    assert torch.allclose(d, ref, atol=1e-6, rtol=1e-6)
    assert torch.allclose(final_d, d_curr, atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    test_registration_and_construction()
    test_beta_aware_phase1()
    test_chunk_forward_backward()
    test_model_smoke()
