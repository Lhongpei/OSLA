"""
Correctness + performance test for optimized phase1 kernel.
"""
import torch
from fla.ops.os_delta_rule.chunk_osgm_phase import (
    compute_osgm_phase1_fwd,
    compute_osgm_phase1_bwd,
)
from fla.ops.os_delta_rule.chunk_osgm_phase_v2 import (
    compute_osgm_phase1_fwd_v2,
    compute_osgm_phase1_bwd_v2,
)
from fla.modules.l2norm import l2norm_fwd


def timed(fn, *args, warmup=10, iters=30, **kwargs):
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


def test_correctness():
    print("=" * 60)
    print("  Correctness Test")
    print("=" * 60)
    device = 'cuda'

    for B, T, H, K in [(1, 4096, 8, 128), (1, 65536, 8, 128), (2, 8192, 8, 128)]:
        for use_denom in [False, True]:
            k = torch.randn(B, T, H, K, device=device, dtype=torch.bfloat16)
            k_norm, _ = l2norm_fwd(k)
            eta, d_min, d_max = 1.0, 0.0, 2.0

            # Forward
            d_ref = compute_osgm_phase1_fwd(k_norm, eta, use_denom, d_min, d_max)
            d_v2 = compute_osgm_phase1_fwd_v2(k_norm, eta, use_denom, d_min, d_max)

            fwd_diff = (d_ref.float() - d_v2.float()).abs().max().item()
            fwd_ok = fwd_diff < 1e-5

            # Backward
            dd_in = torch.randn_like(k_norm)
            dk_ref = compute_osgm_phase1_bwd(k_norm, d_ref, dd_in, eta, use_denom, d_min, d_max)
            dk_v2 = compute_osgm_phase1_bwd_v2(k_norm, d_ref, dd_in, eta, use_denom, d_min, d_max)

            bwd_diff = (dk_ref.float() - dk_v2.float()).abs().max().item()
            bwd_ok = bwd_diff < 1e-5

            status = "PASS" if (fwd_ok and bwd_ok) else "FAIL"
            print(f"  B={B} T={T:>5} denom={str(use_denom):>5} | "
                  f"fwd_diff={fwd_diff:.2e} bwd_diff={bwd_diff:.2e} [{status}]")

            del k, k_norm, d_ref, d_v2, dd_in, dk_ref, dk_v2
            torch.cuda.empty_cache()


def test_performance():
    print()
    print("=" * 60)
    print("  Performance Test (phase1 fwd only)")
    print("=" * 60)
    print(f"{'Config':<25} {'Original':>12} {'Chunked':>12} {'Speedup':>10}")
    print("-" * 62)

    device = 'cuda'
    eta, d_min, d_max = 1.0, 0.0, 2.0
    use_denom = False

    for B, T, H, K in [
        (1, 4096, 8, 128),
        (1, 16384, 8, 128),
        (1, 65536, 8, 128),
        (4, 4096, 8, 128),
    ]:
        k = torch.randn(B, T, H, K, device=device, dtype=torch.bfloat16)
        k_norm, _ = l2norm_fwd(k)

        t_orig = timed(compute_osgm_phase1_fwd, k_norm, eta, use_denom, d_min, d_max)
        t_v2 = timed(compute_osgm_phase1_fwd_v2, k_norm, eta, use_denom, d_min, d_max)

        speedup = t_orig / t_v2
        label = f"B={B} T={T}"
        print(f"{label:<25} {t_orig:>10.3f}ms {t_v2:>10.3f}ms {speedup:>9.2f}x")

        del k, k_norm
        torch.cuda.empty_cache()

    print()
    print("=" * 60)
    print("  Performance Test (phase1 bwd only)")
    print("=" * 60)
    print(f"{'Config':<25} {'Original':>12} {'Chunked':>12} {'Speedup':>10}")
    print("-" * 62)

    for B, T, H, K in [
        (1, 4096, 8, 128),
        (1, 16384, 8, 128),
        (1, 65536, 8, 128),
        (4, 4096, 8, 128),
    ]:
        k = torch.randn(B, T, H, K, device=device, dtype=torch.bfloat16)
        k_norm, _ = l2norm_fwd(k)
        d_out = compute_osgm_phase1_fwd(k_norm, eta, use_denom, d_min, d_max)
        dd_in = torch.randn_like(k_norm)

        t_orig = timed(compute_osgm_phase1_bwd, k_norm, d_out, dd_in, eta, use_denom, d_min, d_max)
        t_v2 = timed(compute_osgm_phase1_bwd_v2, k_norm, d_out, dd_in, eta, use_denom, d_min, d_max)

        speedup = t_orig / t_v2
        label = f"B={B} T={T}"
        print(f"{label:<25} {t_orig:>10.3f}ms {t_v2:>10.3f}ms {speedup:>9.2f}x")

        del k, k_norm, d_out, dd_in
        torch.cuda.empty_cache()


if __name__ == "__main__":
    test_correctness()
    test_performance()
