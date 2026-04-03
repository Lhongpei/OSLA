"""Test original phase1 kernel with different num_warps settings."""
import torch
import triton
import triton.language as tl
from fla.modules.l2norm import l2norm_fwd
from fla.ops.osla_delta_rule.chunk_osgm_phase import (
    osgm_phase1_fwd_kernel,
    osgm_phase1_bwd_kernel,
)


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


def launch_fwd(k, eta, use_denom, d_min, d_max, num_warps):
    B, T, H, K = k.shape
    d_out = torch.empty_like(k)
    BK = triton.next_power_of_2(K)
    osgm_phase1_fwd_kernel[(B * H,)](
        k, d_out, eta, d_min, d_max,
        k.stride(0), k.stride(1), k.stride(2),
        T, H, K, BK,
        USE_DENOMINATOR=use_denom, USE_PROJECTION=True,
        num_warps=num_warps,
    )
    return d_out


def launch_bwd(k, d_out, dd_in, eta, use_denom, d_min, d_max, num_warps):
    B, T, H, K = k.shape
    dk_out = torch.empty_like(k)
    BK = triton.next_power_of_2(K)
    osgm_phase1_bwd_kernel[(B * H,)](
        k, d_out, dd_in, dk_out, eta, d_min, d_max,
        k.stride(0), k.stride(1), k.stride(2),
        T, H, K, BK,
        USE_DENOMINATOR=use_denom, USE_PROJECTION=True,
        num_warps=num_warps,
    )
    return dk_out


def main():
    device = 'cuda'
    B, T, H, K = 1, 65536, 8, 128
    eta, d_min, d_max = 1.0, 0.0, 2.0
    use_denom = False

    k = torch.randn(B, T, H, K, device=device, dtype=torch.bfloat16)
    k_norm, _ = l2norm_fwd(k)
    dd_in = torch.randn_like(k_norm)

    print(f"B={B} T={T} H={H} K={K}")
    print()
    print(f"{'num_warps':<12} {'FWD (ms)':>10} {'BWD (ms)':>10}")
    print("-" * 35)

    for nw in [1, 2, 4, 8, 16]:
        try:
            t_fwd = timed(launch_fwd, k_norm, eta, use_denom, d_min, d_max, nw)
            d_out = launch_fwd(k_norm, eta, use_denom, d_min, d_max, nw)
            t_bwd = timed(launch_bwd, k_norm, d_out, dd_in, eta, use_denom, d_min, d_max, nw)
            print(f"{nw:<12} {t_fwd:>10.3f} {t_bwd:>10.3f}")
        except Exception as e:
            print(f"{nw:<12} FAILED: {e}")

    # Also test with T as non-constexpr by using do_not_specialize
    print()
    print("Default (no num_warps specified):")
    from fla.ops.osla_delta_rule.chunk_osgm_phase import compute_osgm_phase1_fwd, compute_osgm_phase1_bwd
    t_fwd = timed(compute_osgm_phase1_fwd, k_norm, eta, use_denom, d_min, d_max)
    d_out = compute_osgm_phase1_fwd(k_norm, eta, use_denom, d_min, d_max)
    t_bwd = timed(compute_osgm_phase1_bwd, k_norm, d_out, dd_in, eta, use_denom, d_min, d_max)
    print(f"{'default':<12} {t_fwd:>10.3f} {t_bwd:>10.3f}")


if __name__ == "__main__":
    main()
