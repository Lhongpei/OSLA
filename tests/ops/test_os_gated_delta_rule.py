# -*- coding: utf-8 -*-
"""Unit tests for `fla.ops.os_gated_delta_rule.chunk_os_gated_delta_rule`.

Two sanity checks:
  1. At g=0 (no state decay), the new op must match the proven
     `chunk_delta_rule_osgm` (plain OSGM) forward and backward bit-for-bit.
  2. At g≠0, gradients must align with a pure-PyTorch recurrent naive
     reference via cosine similarity > 0.995 per parameter.

Both are run at multiple shapes to catch accumulation-dependent drift.
"""
import pytest
import torch
import torch.nn.functional as F


# Guard: only run when CUDA + triton are available.
pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)


from fla.ops.os_delta_rule.chunk_osgm import chunk_delta_rule_osgm
from fla.ops.os_gated_delta_rule import chunk_os_gated_delta_rule
from fla.ops.os_gated_delta_rule.naive import naive_recurrent_os_gated_delta_rule


DEVICE = "cuda"


def _make(B, T, H, K, V, seed=42, scale=0.1):
    torch.manual_seed(seed)
    q = (torch.randn(B, T, H, K, device=DEVICE) * scale).requires_grad_()
    k = (torch.randn(B, T, H, K, device=DEVICE) * scale).requires_grad_()
    v = (torch.randn(B, T, H, V, device=DEVICE) * scale).requires_grad_()
    beta = torch.rand(B, T, H, device=DEVICE).sigmoid().detach().requires_grad_()
    return q, k, v, beta


def _clone(*tensors):
    return [t.detach().clone().requires_grad_() for t in tensors]


@pytest.mark.parametrize("shape", [(2, 64, 4, 32, 32), (2, 128, 4, 32, 32), (2, 256, 4, 32, 32)])
def test_g_zero_matches_plain_osgm(shape):
    """When g=0 everywhere, the gated kernel must agree with plain OSGM."""
    B, T, H, K, V = shape
    q1, k1, v1, b1 = _make(B, T, H, K, V)
    q2, k2, v2, b2 = _clone(q1, k1, v1, b1)
    g = torch.zeros(B, T, H, device=DEVICE).requires_grad_()

    o_gated, _ = chunk_os_gated_delta_rule(
        q1, k1, v1, g, b1, scale=K ** -0.5, eta=1.0,
        use_denominator=False, d_min=0.0, d_max=1e9,
    )
    o_plain, _ = chunk_delta_rule_osgm(
        q2, k2, v2, b2, scale=K ** -0.5, eta=1.0,
        use_denominator=False, d_min=0.0, d_max=1e9,
    )
    assert (o_gated - o_plain).abs().max().item() < 5e-5

    do = torch.randn_like(o_gated)
    (o_gated * do).sum().backward()
    (o_plain * do).sum().backward()
    # bwd: per-parameter cosine must be essentially 1.0 (kernel drift only)
    for name, a, b in [("q", q1, q2), ("k", k1, k2), ("v", v1, v2), ("beta", b1, b2)]:
        cos = F.cosine_similarity(a.grad.flatten(), b.grad.flatten(), dim=0).item()
        assert cos > 0.9999, f"d{name}: cos={cos}"


@pytest.mark.parametrize("shape", [(2, 64, 4, 32, 32), (2, 128, 4, 32, 32)])
@pytest.mark.parametrize("g_scale", [0.1, 0.5, 1.0])
def test_bwd_matches_naive(shape, g_scale):
    """With g != 0, chunk gradients must cos-align with the recurrent naive."""
    B, T, H, K, V = shape
    q1, k1, v1, b1 = _make(B, T, H, K, V)
    g1 = (-torch.rand(B, T, H, device=DEVICE) * g_scale).requires_grad_()
    q2, k2, v2, b2 = _clone(q1, k1, v1, b1)
    g2 = g1.detach().clone().requires_grad_()

    o_naive, _, _ = naive_recurrent_os_gated_delta_rule(
        q1, k1, v1, g1, b1,
        scale=K ** -0.5, eta=1.0,
        use_denominator=False, d_min=0.0, d_max=1e9,
    )
    o_chunk, _ = chunk_os_gated_delta_rule(
        q2, k2, v2, g2, b2,
        scale=K ** -0.5, eta=1.0,
        use_denominator=False, d_min=0.0, d_max=1e9,
    )

    do = torch.randn_like(o_naive)
    (o_naive * do).sum().backward()
    (o_chunk * do).sum().backward()

    for name, a, b in [("q", q1, q2), ("k", k1, k2), ("v", v1, v2),
                       ("beta", b1, b2), ("g", g1, g2)]:
        cos = F.cosine_similarity(a.grad.flatten(), b.grad.flatten(), dim=0).item()
        assert cos > 0.995, f"d{name}: cos={cos}"


@pytest.mark.parametrize("decay_mode", ["learnable", "constant"])
def test_decay_variant_bwd_matches_naive(decay_mode):
    """OSGM with γ decay (learnable/constant): chunk vs naive autograd.

    γ is a sigmoid over a per-head `gamma_log` parameter. When
    decay_mode='constant', γ is a buffer (no gradient); when 'learnable',
    γ is learnable and we also compare its gradient.
    """
    B, T, H, K, V = 2, 64, 4, 32, 32
    q1, k1, v1, b1 = _make(B, T, H, K, V)
    g1 = (-torch.rand(B, T, H, device=DEVICE) * 0.5).requires_grad_()
    # gamma_log init so γ ≈ 0.999 (the default used in delta_net.py decay init)
    gamma_log = torch.full((H,), 6.9, device=DEVICE)
    if decay_mode == "learnable":
        gamma_log = gamma_log.requires_grad_()
    q2, k2, v2, b2 = _clone(q1, k1, v1, b1)
    g2 = g1.detach().clone().requires_grad_()
    gamma_log2 = gamma_log.detach().clone()
    if decay_mode == "learnable":
        gamma_log2 = gamma_log2.requires_grad_()

    o_naive, _, _ = naive_recurrent_os_gated_delta_rule(
        q1, k1, v1, g1, b1,
        scale=K ** -0.5, eta=1.0,
        use_denominator=False, d_min=0.0, d_max=1e9,
        decay_mode=decay_mode, gamma_log=gamma_log,
    )
    o_chunk, _ = chunk_os_gated_delta_rule(
        q2, k2, v2, g2, b2,
        scale=K ** -0.5, eta=1.0,
        use_denominator=False, d_min=0.0, d_max=1e9,
        decay_mode=decay_mode, gamma_log=gamma_log2,
    )

    do = torch.randn_like(o_naive)
    (o_naive * do).sum().backward()
    (o_chunk * do).sum().backward()

    for name, a, b in [("q", q1, q2), ("k", k1, k2), ("v", v1, v2),
                       ("beta", b1, b2), ("g", g1, g2)]:
        cos = F.cosine_similarity(a.grad.flatten(), b.grad.flatten(), dim=0).item()
        assert cos > 0.99, f"[{decay_mode}] d{name}: cos={cos}"

    if decay_mode == "learnable":
        cos = F.cosine_similarity(
            gamma_log.grad.flatten(), gamma_log2.grad.flatten(), dim=0
        ).item()
        assert cos > 0.99, f"dgamma_log: cos={cos}"


def test_data_dependent_decay_bwd_matches_naive():
    """OSGM with data-dependent γ: chunk vs naive autograd."""
    B, T, H, K, V = 2, 64, 4, 32, 32
    q1, k1, v1, b1 = _make(B, T, H, K, V)
    g1 = (-torch.rand(B, T, H, device=DEVICE) * 0.5).requires_grad_()
    # raw logit going into logsigmoid → g_decay (same convention as delta_net.py)
    a_logit = (torch.randn(B, T, H, device=DEVICE) * 0.3 + 6.9).requires_grad_()
    q2, k2, v2, b2 = _clone(q1, k1, v1, b1)
    g2 = g1.detach().clone().requires_grad_()
    a_logit2 = a_logit.detach().clone().requires_grad_()

    g_dec1 = F.logsigmoid(a_logit)
    g_dec2 = F.logsigmoid(a_logit2)

    o_naive, _, _ = naive_recurrent_os_gated_delta_rule(
        q1, k1, v1, g1, b1,
        scale=K ** -0.5, eta=1.0,
        use_denominator=False, d_min=0.0, d_max=1e9,
        decay_mode="data_dependent", g_decay=g_dec1,
    )
    o_chunk, _ = chunk_os_gated_delta_rule(
        q2, k2, v2, g2, b2,
        scale=K ** -0.5, eta=1.0,
        use_denominator=False, d_min=0.0, d_max=1e9,
        decay_mode="data_dependent", g_decay=g_dec2,
    )

    do = torch.randn_like(o_naive)
    (o_naive * do).sum().backward()
    (o_chunk * do).sum().backward()

    for name, a, b in [("q", q1, q2), ("k", k1, k2), ("v", v1, v2),
                       ("beta", b1, b2), ("g", g1, g2),
                       ("a_logit", a_logit, a_logit2)]:
        cos = F.cosine_similarity(a.grad.flatten(), b.grad.flatten(), dim=0).item()
        assert cos > 0.99, f"d{name}: cos={cos}"
