import torch
import torch.nn.functional as F

from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from fla.ops.os_gated_delta_rule import chunk_os_gated_delta_rule
from fla.ops.os_gated_delta_rule.naive import naive_recurrent_os_gated_delta_rule_post_gate_regret


def clone_req(x):
    return x.detach().clone().requires_grad_(x.requires_grad)


def grad_cos(a, b):
    af = a.grad.float().flatten()
    bf = b.grad.float().flatten()
    return F.cosine_similarity(af, bf, dim=0).item()


def check_post_gate(mode, dtype=torch.float32, t=128):
    torch.manual_seed(1234 + (1 if mode == "sep_apf" else 0))
    device = "cuda"
    bsz, heads, kdim, vdim = 2, 3, 32, 32
    q = (torch.randn(bsz, t, heads, kdim, device=device, dtype=dtype) * 0.5).requires_grad_()
    k = (torch.randn(bsz, t, heads, kdim, device=device, dtype=dtype) * 0.5).requires_grad_()
    v = (torch.randn(bsz, t, heads, vdim, device=device, dtype=dtype) * 0.5).requires_grad_()
    beta = torch.rand(bsz, t, heads, device=device, dtype=torch.float32).sigmoid().requires_grad_()
    g = F.logsigmoid(torch.randn(bsz, t, heads, device=device, dtype=torch.float32)).requires_grad_()
    d0 = torch.ones(bsz, heads, kdim, device=device, dtype=torch.float32).requires_grad_()
    h0 = (torch.randn(bsz, heads, kdim, vdim, device=device, dtype=torch.float32) * 0.05).requires_grad_()

    q2, k2, v2, beta2, g2, d02, h02 = [clone_req(x) for x in (q, k, v, beta, g, d0, h0)]

    kwargs = dict(
        scale=kdim ** -0.5,
        eta=0.003,
        initial_state=(h0, d0),
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_denominator=False,
        d_min=0.6666667,
        d_max=1.5,
        post_gate_regret_beta_aware=True,
    )
    ref_kwargs = dict(
        scale=kdim ** -0.5,
        eta=0.003,
        initial_state=h02,
        initial_d=d02,
        output_final_state=True,
        output_final_d=True,
        d_min=0.6666667,
        d_max=1.5,
    )
    if mode == "sep_apf":
        # Separate preconditioner gate initialized near exp(g_decay)=0.999.
        decay_logit = (torch.randn(bsz, t, heads, device=device) * 0.05 + 6.9).requires_grad_()
        decay_logit2 = clone_req(decay_logit)
        g_decay = F.logsigmoid(decay_logit)
        g_decay2 = F.logsigmoid(decay_logit2)
        kwargs.update(decay_mode="data_dependent", g_decay=g_decay)
        ref_kwargs.update(decay_mode="data_dependent", g_decay=g_decay2)
    else:
        decay_logit = decay_logit2 = None
        kwargs.update(decay_mode="none")
        ref_kwargs.update(decay_mode="none")

    out, (hout, dout) = chunk_os_gated_delta_rule(q, k, v, g, beta, **kwargs)
    qn = F.normalize(q2.float(), p=2, dim=-1).to(dtype)
    kn = F.normalize(k2.float(), p=2, dim=-1).to(dtype)
    ref, href, dref = naive_recurrent_os_gated_delta_rule_post_gate_regret(
        qn, kn, v2, g2, beta2, **ref_kwargs
    )

    do = torch.randn_like(out)
    dh = torch.randn_like(hout)
    dd = torch.randn_like(dout)
    (out * do).float().sum().add((hout * dh).sum()).add((dout * dd).sum()).backward()
    (ref * do).float().sum().add((href * dh).sum()).add((dref * dd).sum()).backward()

    out_err = (out.float() - ref.float()).abs().max().item()
    h_err = (hout.float() - href.float()).abs().max().item()
    d_err = (dout.float() - dref.float()).abs().max().item()
    print(f"{mode}: out_max={out_err:.3e} h_max={h_err:.3e} d_max={d_err:.3e}")
    assert out_err < 5e-4
    assert h_err < 2e-3
    assert d_err < 5e-5

    pairs = [("q", q, q2), ("k", k, k2), ("v", v, v2), ("g", g, g2),
             ("beta", beta, beta2), ("d0", d0, d02), ("h0", h0, h02)]
    if decay_logit is not None:
        pairs.append(("decay_logit", decay_logit, decay_logit2))
    for name, a, b in pairs:
        cos = grad_cos(a, b)
        print(f"{mode}: grad {name} cos={cos:.8f}")
        assert cos > 0.999


def check_degenerate_to_gdn():
    torch.manual_seed(4321)
    device = "cuda"
    bsz, t, heads, kdim, vdim = 2, 128, 3, 32, 32
    q = (torch.randn(bsz, t, heads, kdim, device=device) * 0.5).requires_grad_()
    k = (torch.randn(bsz, t, heads, kdim, device=device) * 0.5).requires_grad_()
    v = (torch.randn(bsz, t, heads, vdim, device=device) * 0.5).requires_grad_()
    g = F.logsigmoid(torch.randn(bsz, t, heads, device=device)).requires_grad_()
    beta = torch.rand(bsz, t, heads, device=device).sigmoid().requires_grad_()
    h0 = (torch.randn(bsz, heads, kdim, vdim, device=device) * 0.05).requires_grad_()
    d0 = torch.ones(bsz, heads, kdim, device=device).requires_grad_()
    q2, k2, v2, g2, beta2, h02 = [clone_req(x) for x in (q, k, v, g, beta, h0)]

    out, (hout, dout) = chunk_os_gated_delta_rule(
        q, k, v, g, beta,
        scale=kdim ** -0.5,
        eta=0.0,
        initial_state=(h0, d0),
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_denominator=False,
        d_min=1.0,
        d_max=1.0,
        decay_mode="none",
        post_gate_regret_beta_aware=True,
    )
    ref, href = chunk_gated_delta_rule(
        q2, k2, v2, g2, beta2,
        scale=kdim ** -0.5,
        initial_state=h02,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    out_err = (out.float() - ref.float()).abs().max().item()
    h_err = (hout.float() - href.float()).abs().max().item()
    print(f"d=1/eta=0 -> GDN: out_max={out_err:.3e} h_max={h_err:.3e} d=({dout.min().item():.1f},{dout.max().item():.1f})")
    assert out_err < 1e-4
    assert h_err < 1e-4


if __name__ == "__main__":
    assert torch.cuda.is_available()
    check_degenerate_to_gdn()
    check_post_gate("no_dd")
    check_post_gate("sep_apf")
    print("OS-GDN post-gate kernel checks passed.")
