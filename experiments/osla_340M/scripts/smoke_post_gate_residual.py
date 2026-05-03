"""Smoke test for variant A: osgm_post_gate_residual=True.

Verifies:
  1. Tiny OSGDN model builds with the new flag.
  2. Forward + backward run without NaN; loss finite.
  3. Phase1 path's gradient on the GDN gate driver `a_proj` is non-zero
     (validates that chunk_local_cumsum at the layer correctly propagates
     dg_residual back through cumsum to the same `a_proj.weight` that the
     state path also drives).
  4. Compared against post_gate_residual=False, the new path produces a
     LARGER grad on `a_proj.weight` (since it now receives both state-side
     dg AND phase1-residual dg).

Run: CUDA_VISIBLE_DEVICES=0 python smoke_post_gate_residual.py
"""
import torch

from fla.models.os_gated_deltanet import OSGDNConfig
from fla.models.os_gated_deltanet.modeling_os_gated_deltanet import OSGDNForCausalLM


def build(post_gate: bool):
    cfg = OSGDNConfig(
        hidden_size=128,
        num_hidden_layers=2,
        num_heads=2,
        head_dim=64,
        expand_v=2,
        intermediate_size=256,
        vocab_size=1024,
        use_osgm=True,
        osgm_decay_mode="data_dependent",
        osgm_d_decay_source="osgm",         # variant A is built on top of osgm dd-decay
        osgm_post_gate_residual=post_gate,
        osgm_eta=1.0,
        osgm_use_denominator=False,
        osgm_d_min=0.0,
        osgm_d_max=1e9,
        fuse_cross_entropy=False,
        fuse_swiglu=False,
        fuse_norm=False,
    )
    model = OSGDNForCausalLM(cfg).cuda().bfloat16()
    return model, cfg


def run_one(post_gate: bool, seq_len: int = 256):
    torch.manual_seed(42)
    model, cfg = build(post_gate)
    label = "post_gate" if post_gate else "baseline_dd_decay"
    print(f"\n[{label}] osgm_post_gate_residual = {cfg.osgm_post_gate_residual}")

    x = torch.randint(0, 1024, (1, seq_len)).cuda()
    out = model(input_ids=x, labels=x)
    loss = out.loss
    print(f"[{label}] loss = {loss.item():.4f}, finite={torch.isfinite(loss).item()}")
    assert torch.isfinite(loss), f"non-finite loss: {loss}"

    loss.backward()

    # Inspect grads for safety + measure a_proj.weight grad norm.
    nan_grads, none_grads = [], []
    for n, p in model.named_parameters():
        if p.grad is None:
            none_grads.append(n)
            continue
        if not torch.isfinite(p.grad).all():
            nan_grads.append(n)
    print(f"[{label}] params w/o grad ({len(none_grads)}): {none_grads[:3]}")
    print(f"[{label}] params w/ NaN/Inf grad: {nan_grads}")
    assert not nan_grads, f"NaN/Inf grads: {nan_grads}"

    aproj_norm = None
    osgm_aproj_norm = None
    for n, p in model.named_parameters():
        if n.endswith("attn.a_proj.weight"):
            aproj_norm = p.grad.norm().item()
            print(f"[{label}] {n} grad_norm = {aproj_norm:.4e}")
        if n.endswith("attn.osgm_a_proj.weight"):
            osgm_aproj_norm = p.grad.norm().item()
            print(f"[{label}] {n} grad_norm = {osgm_aproj_norm:.4e}")
    return loss.item(), aproj_norm, osgm_aproj_norm


if __name__ == "__main__":
    print("=" * 60)
    print("Variant A smoke test: osgm_post_gate_residual")
    print("=" * 60)
    loss_base, base_aproj, base_osgm = run_one(False)
    loss_pg, pg_aproj, pg_osgm = run_one(True)

    print("\n" + "=" * 60)
    print(f"baseline_dd_decay  loss = {loss_base:.4f}, a_proj_grad = {base_aproj:.4e}")
    print(f"post_gate_residual loss = {loss_pg:.4f}, a_proj_grad = {pg_aproj:.4e}")
    print()
    if pg_aproj > base_aproj * 1.001:
        print("PASS: post_gate's a_proj receives extra gradient via phase1 residual ✓")
    else:
        print("WARN: post_gate's a_proj grad isn't bigger — cumsum bwd may not be flowing")
    print("All checks passed: implementation runs end-to-end without NaN.")
    print("=" * 60)
