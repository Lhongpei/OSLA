"""Standalone loss-decrease smoke test on random data.

Trains a small OSGDN model for ~50 steps with osgm_post_gate_regret=True
(and a comparison run with =False) and prints the loss curve.

Goal: confirm post-gate-regret training is *stable* and the loss curve
isn't catastrophic. Real comparison against baselines requires the full
flame run.
"""
import time

import torch

from fla.models.os_gated_deltanet import OSGDNConfig, OSGDNForCausalLM


def make_config(use_post_gate_regret: bool) -> OSGDNConfig:
    return OSGDNConfig(
        # Smaller-than-340M but same shapes/aspect for speed.
        hidden_size=512,
        num_heads=4,
        num_v_heads=4,
        num_hidden_layers=6,
        head_dim=128,
        expand_v=2,
        vocab_size=1024,
        max_position_embeddings=512,
        use_short_conv=True,
        use_gate=True,
        use_osgm=True,
        osgm_eta=1.0,
        osgm_use_denominator=False,
        osgm_d_min=0.0,
        osgm_d_max=1e9,
        osgm_decay_mode="data_dependent",
        osgm_post_gate_regret=use_post_gate_regret,
        osgm_post_gate_regret_chunk_size=64,
        # Baseline comparison: gate_aware_hypergradient=True (the best non-§3.4
        # variant per the report).
        gate_aware_hypergradient=not use_post_gate_regret,
        fuse_norm=False,
        fuse_swiglu=False,
        fuse_cross_entropy=False,
        fuse_linear_cross_entropy=False,
    )


def run(use_post_gate_regret: bool, n_steps: int = 30, seed: int = 42):
    torch.manual_seed(seed)
    config = make_config(use_post_gate_regret)
    model = OSGDNForCausalLM(config).to("cuda").bfloat16()
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, eps=1e-15)

    B, T = 4, 512
    rng = torch.Generator(device="cuda").manual_seed(seed)

    losses = []
    t0 = time.time()
    for step in range(n_steps):
        ids = torch.randint(0, config.vocab_size, (B, T), device="cuda", generator=rng)
        out = model(input_ids=ids, labels=ids)
        optimizer.zero_grad()
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(out.loss.item())
        if step < 5 or step % 5 == 0:
            print(f"  step {step:3d}: loss={out.loss.item():.4f}, "
                  f"finite={torch.isfinite(out.loss).item()}")
    dt = time.time() - t0
    print(f"  total time: {dt:.1f}s ({dt/n_steps*1000:.0f} ms/step)")
    print(f"  loss[0]={losses[0]:.3f}, loss[-1]={losses[-1]:.3f}, "
          f"delta={losses[0]-losses[-1]:.3f}")
    print(f"  any NaN/inf: {not all(map(lambda x: x == x and abs(x) < 1e10, losses))}")
    return losses


print("=" * 60)
print("RUN A: osgm_post_gate_regret=True (NEW §3.4 hypergradient)")
print("=" * 60)
losses_new = run(use_post_gate_regret=True, n_steps=30)

print("\n" + "=" * 60)
print("RUN B: gate_aware_hypergradient=True (baseline best per report)")
print("=" * 60)
losses_old = run(use_post_gate_regret=False, n_steps=30)

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  RUN A (post_gate_regret):  {losses_new[0]:.3f} -> {losses_new[-1]:.3f}, "
      f"drop={losses_new[0]-losses_new[-1]:.3f}")
print(f"  RUN B (gate_aware):         {losses_old[0]:.3f} -> {losses_old[-1]:.3f}, "
      f"drop={losses_old[0]-losses_old[-1]:.3f}")

# Sanity: both should drop loss meaningfully on random data
assert losses_new[-1] < losses_new[0], "RUN A loss did not decrease"
assert losses_old[-1] < losses_old[0], "RUN B loss did not decrease"
print("\nSMOKE TEST PASSED: both runs decrease loss")
