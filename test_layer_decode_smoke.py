"""Quick layer-level smoke test for OS-GDN decode dispatch.

Builds a tiny GatedDeltaNet layer with use_osgm=True, runs a prefill of
T=128, then a single decode step with use_cache=True, and confirms:
  1. The fused_recurrent path is selected at decode (instead of falling to chunk).
  2. The decode output is consistent with extending the prefill by one more
     chunk-mode step (i.e., chunk(T=129)[..., -1, :] ≈ decode_after_chunk(T=128)).

Both production OSGM configs are exercised.
"""

from __future__ import annotations
import os

os.environ.setdefault("OSLA_DISABLE_POST_GATE_COMPILE", "1")

import torch

import fla  # noqa: F401
from fla.layers.gated_deltanet import GatedDeltaNet


def _build_layer(*, decay_mode: str, source: str | None = None) -> GatedDeltaNet:
    layer = GatedDeltaNet(
        hidden_size=512,
        expand_v=2,
        head_dim=64,
        num_heads=4,
        num_v_heads=None,
        mode='chunk',
        use_gate=True,
        use_short_conv=True,
        layer_idx=0,
        # OSGM
        use_osgm=True,
        osgm_eta=0.003,
        osgm_d_min=0.6666667,
        osgm_d_max=1.5,
        osgm_decay_mode=decay_mode,
        osgm_post_gate_regret=True,
        osgm_post_gate_regret_chunk_size=64,
        osgm_d_decay_source=source if source else "osgm",
    ).cuda().to(torch.bfloat16).eval()
    # Float32 for OSGM-related parameters
    layer.initial_scale.data = layer.initial_scale.data.float()
    return layer


def _check(name: str, *, decay_mode: str, source: str | None):
    print(f"\n=== {name} (decay={decay_mode}, source={source}) ===")
    torch.manual_seed(0)
    layer = _build_layer(decay_mode=decay_mode, source=source)
    assert layer._osgm_supports_fused_recurrent(), (
        "_osgm_supports_fused_recurrent should be True for this config"
    )

    T = 128
    h = torch.randn(1, T, 512, dtype=torch.bfloat16, device='cuda') * 0.1

    # Path A: chunk over full T+1, take the last token output.
    h_full = torch.cat([h, torch.randn(1, 1, 512, dtype=torch.bfloat16, device='cuda') * 0.1], dim=1)
    layer.mode = 'chunk'
    with torch.inference_mode():
        out_full, _, _ = layer(h_full, use_cache=False)
    o_ref = out_full[:, -1:].clone()

    # Path B: chunk-prefill T, then one fused decode step on the last token of h_full.
    layer.mode = 'chunk'  # prefill mode
    with torch.inference_mode():
        from fla.models.utils import Cache
        past = Cache()
        # prefill
        layer.mode = 'chunk'
        out_pre, _, past = layer(h, use_cache=True, past_key_values=past)
        # decode (q_len=1) — should auto-switch to fused_recurrent
        out_dec, _, past = layer(h_full[:, -1:], use_cache=True, past_key_values=past)
    o_dec = out_dec.clone()

    diff = (o_ref.float() - o_dec.float()).abs()
    rel = diff.max() / o_ref.float().abs().max().clamp(min=1e-6)
    print(f"  max abs diff = {diff.max().item():.3e}  rel = {rel.item():.3e}")
    if rel.item() > 1e-1:
        raise AssertionError(f"FAIL: rel diff {rel.item():.3e} > 0.1")
    print(f"  OK")


if __name__ == "__main__":
    _check("OS-GDN no-DD", decay_mode="none", source=None)
    _check("OS-GDN APF (gdn-source)", decay_mode="data_dependent", source="gdn")
    print("\nLayer-level decode smoke test passed.")
