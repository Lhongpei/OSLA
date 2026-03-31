"""
Profile: Baseline DeltaNet chunk vs OSLA chunk vs OSGM chunk
Kernel-level breakdown via torch.profiler
"""
import torch
import torch.nn.functional as F

from fla.ops.delta_rule import chunk_delta_rule
from fla.ops.osla_delta_rule.chunk_osgm import chunk_delta_rule_osgm
from fla.ops.osla_delta_rule.chunk import chunk_osla_delta_rule
from fla.ops.osla_delta_rule.chunk_osgm_phase import compute_osgm_phase1_fwd
from fla.modules.l2norm import l2norm_fwd


def timed_fwd_bwd(fn, *args, warmup=10, iters=30, **kwargs):
    for a in args:
        if isinstance(a, torch.Tensor) and a.grad is not None:
            a.grad = None
    for _ in range(warmup):
        o = fn(*args, **kwargs)
        o[0].sum().backward(retain_graph=True)
        for a in args:
            if isinstance(a, torch.Tensor) and a.grad is not None:
                a.grad = None
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        for a in args:
            if isinstance(a, torch.Tensor) and a.grad is not None:
                a.grad = None
        starts[i].record()
        o = fn(*args, **kwargs)
        o[0].sum().backward()
        ends[i].record()
    torch.cuda.synchronize()
    return sum(s.elapsed_time(e) for s, e in zip(starts, ends)) / iters


def timed_fwd(fn, *args, warmup=10, iters=30, **kwargs):
    for _ in range(warmup):
        fn(*args, **kwargs)
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn(*args, **kwargs)
        ends[i].record()
    torch.cuda.synchronize()
    return sum(s.elapsed_time(e) for s, e in zip(starts, ends)) / iters


def make_inputs(B, T, H, K, V, device='cuda', dtype=torch.bfloat16, grad=False):
    q = torch.randn(B, T, H, K, device=device, dtype=dtype, requires_grad=grad)
    k = torch.randn(B, T, H, K, device=device, dtype=dtype, requires_grad=grad)
    v = torch.randn(B, T, H, V, device=device, dtype=dtype, requires_grad=grad)
    beta = torch.rand(B, T, H, device=device, dtype=dtype).sigmoid().detach().requires_grad_(grad)
    return q, k, v, beta


def run():
    device = 'cuda'
    dtype = torch.bfloat16
    H, K, V = 8, 128, 128
    scale = K ** -0.5
    eta, d_min, d_max, use_denom = 1.0, 0.0, 2.0, False

    configs = [
        (1, 4096,  "B=1  T=4096"),
        (1, 16384, "B=1  T=16384"),
        (1, 65536, "B=1  T=65536 (training)"),
        (4, 4096,  "B=4  T=4096"),
    ]

    # ====== Part 1: End-to-end FWD comparison ======
    print("=" * 80)
    print("  FORWARD ONLY")
    print("=" * 80)
    print(f"{'Config':<25} {'Baseline':>10} {'OSLA':>10} {'OSGM':>10} {'OSGM/BL':>10}")
    print("-" * 70)

    for B, T, label in configs:
        q, k, v, beta = make_inputs(B, T, H, K, V)
        init_scale = torch.zeros(B, H, K, device=device, dtype=torch.float32)

        t_bl = timed_fwd(chunk_delta_rule, q, k, v, beta, scale,
                         use_qk_l2norm_in_kernel=True)

        t_osla = timed_fwd(chunk_osla_delta_rule, q, k, v, beta, scale,
                           initial_scale=init_scale, use_qk_l2norm_in_kernel=True)

        t_osgm = timed_fwd(chunk_delta_rule_osgm, q, k, v, beta, scale,
                           eta=eta, use_qk_l2norm_in_kernel=True,
                           use_denominator=use_denom, d_min=d_min, d_max=d_max)

        ratio = t_osgm / t_bl
        print(f"{label:<25} {t_bl:>8.2f}ms {t_osla:>8.2f}ms {t_osgm:>8.2f}ms {ratio:>9.2f}x")

        del q, k, v, beta, init_scale
        torch.cuda.empty_cache()

    # ====== Part 2: End-to-end FWD+BWD comparison ======
    print()
    print("=" * 80)
    print("  FORWARD + BACKWARD")
    print("=" * 80)
    print(f"{'Config':<25} {'Baseline':>10} {'OSLA':>10} {'OSGM':>10} {'OSGM/BL':>10}")
    print("-" * 70)

    for B, T, label in configs:
        q1, k1, v1, b1 = make_inputs(B, T, H, K, V, grad=True)
        q2, k2, v2, b2 = make_inputs(B, T, H, K, V, grad=True)
        q3, k3, v3, b3 = make_inputs(B, T, H, K, V, grad=True)
        init_scale = torch.zeros(B, H, K, device=device, dtype=torch.float32)

        t_bl = timed_fwd_bwd(chunk_delta_rule, q1, k1, v1, b1, scale,
                             use_qk_l2norm_in_kernel=True)

        t_osla = timed_fwd_bwd(chunk_osla_delta_rule, q2, k2, v2, b2, scale,
                               initial_scale=init_scale, use_qk_l2norm_in_kernel=True)

        t_osgm = timed_fwd_bwd(chunk_delta_rule_osgm, q3, k3, v3, b3, scale,
                               eta=eta, use_qk_l2norm_in_kernel=True,
                               use_denominator=use_denom, d_min=d_min, d_max=d_max)

        ratio = t_osgm / t_bl
        print(f"{label:<25} {t_bl:>8.2f}ms {t_osla:>8.2f}ms {t_osgm:>8.2f}ms {ratio:>9.2f}x")

        del q1, k1, v1, b1, q2, k2, v2, b2, q3, k3, v3, b3, init_scale
        torch.cuda.empty_cache()

    # ====== Part 3: osgm_phase1 isolated ======
    print()
    print("=" * 80)
    print("  OSGM Phase1 (isolated) — to confirm it's NOT the bottleneck")
    print("=" * 80)
    print(f"{'Config':<25} {'phase1_fwd':>12} {'l2norm':>12}")
    print("-" * 52)

    for B, T, label in configs:
        k = torch.randn(B, T, H, K, device=device, dtype=dtype)
        k_norm, _ = l2norm_fwd(k)

        t_phase1 = timed_fwd(compute_osgm_phase1_fwd, k_norm, eta, use_denom, d_min, d_max)
        t_l2 = timed_fwd(l2norm_fwd, k)

        print(f"{label:<25} {t_phase1:>10.3f}ms {t_l2:>10.3f}ms")

        del k, k_norm
        torch.cuda.empty_cache()

    # ====== Part 4: torch.profiler trace for T=65536 ======
    print()
    print("=" * 80)
    print("  Generating torch.profiler trace for T=65536 ...")
    print("=" * 80)

    B, T = 1, 65536
    q, k, v, beta = make_inputs(B, T, H, K, V, grad=True)

    # warmup
    for _ in range(3):
        o = chunk_delta_rule_osgm(q, k, v, beta, scale, eta=eta,
                                  use_qk_l2norm_in_kernel=True,
                                  use_denominator=use_denom, d_min=d_min, d_max=d_max)
        o[0].sum().backward()
        for a in [q, k, v, beta]:
            if a.grad is not None:
                a.grad = None

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        o = chunk_delta_rule_osgm(q, k, v, beta, scale, eta=eta,
                                  use_qk_l2norm_in_kernel=True,
                                  use_denominator=use_denom, d_min=d_min, d_max=d_max)
        o[0].sum().backward()

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25))


if __name__ == "__main__":
    run()
