"""
Profile: Baseline DeltaNet chunk vs OSLA chunk vs OSGM chunk
Tests both non-varlen and varlen (cu_seqlens) modes to match training config.
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


def make_cu_seqlens(total_len, context_len, device='cuda'):
    """Create cu_seqlens for varlen packing: [0, ctx, 2*ctx, ..., total_len]"""
    n_seqs = total_len // context_len
    cu_seqlens = torch.arange(0, n_seqs + 1, device=device, dtype=torch.long) * context_len
    return cu_seqlens


def run():
    device = 'cuda'
    dtype = torch.bfloat16
    H, K, V = 8, 128, 128
    scale = K ** -0.5
    eta, d_min, d_max, use_denom = 1.0, 0.0, 2.0, False

    # Non-varlen configs
    configs_plain = [
        (1, 4096,  None, "B=1  T=4096"),
        (1, 16384, None, "B=1  T=16384"),
        (1, 65536, None, "B=1  T=65536"),
        (4, 4096,  None, "B=4  T=4096"),
    ]

    # Varlen configs (matches training: B=1, T=65536, context_len=4096)
    configs_varlen = [
        (1, 65536, 4096,  "varlen T=65536 ctx=4096 (16 seqs)"),
        (1, 65536, 2048,  "varlen T=65536 ctx=2048 (32 seqs)"),
        (1, 65536, 8192,  "varlen T=65536 ctx=8192 (8 seqs)"),
    ]

    all_configs = configs_plain + configs_varlen

    # ====== Part 1: End-to-end FWD comparison ======
    print("=" * 85)
    print("  FORWARD ONLY")
    print("=" * 85)
    print(f"{'Config':<40} {'Baseline':>10} {'OSLA':>10} {'OSGM':>10} {'OSGM/BL':>8}")
    print("-" * 82)

    for B, T, ctx_len, label in all_configs:
        q, k, v, beta = make_inputs(B, T, H, K, V)
        cu = make_cu_seqlens(T, ctx_len, device) if ctx_len else None
        N = len(cu) - 1 if cu is not None else B
        init_scale = torch.zeros(N, H, K, device=device, dtype=torch.float32)

        t_bl = timed_fwd(chunk_delta_rule, q, k, v, beta, scale,
                         use_qk_l2norm_in_kernel=True, cu_seqlens=cu)

        t_osla = timed_fwd(chunk_osla_delta_rule, q, k, v, beta, scale,
                           initial_scale=init_scale, use_qk_l2norm_in_kernel=True,
                           cu_seqlens=cu)

        t_osgm = timed_fwd(chunk_delta_rule_osgm, q, k, v, beta, scale,
                           eta=eta, use_qk_l2norm_in_kernel=True,
                           use_denominator=use_denom, d_min=d_min, d_max=d_max,
                           cu_seqlens=cu)

        ratio = t_osgm / t_bl
        print(f"{label:<40} {t_bl:>8.2f}ms {t_osla:>8.2f}ms {t_osgm:>8.2f}ms {ratio:>7.2f}x")

        del q, k, v, beta, init_scale
        torch.cuda.empty_cache()

    # ====== Part 2: End-to-end FWD+BWD comparison ======
    print()
    print("=" * 85)
    print("  FORWARD + BACKWARD")
    print("=" * 85)
    print(f"{'Config':<40} {'Baseline':>10} {'OSLA':>10} {'OSGM':>10} {'OSGM/BL':>8}")
    print("-" * 82)

    for B, T, ctx_len, label in all_configs:
        q1, k1, v1, b1 = make_inputs(B, T, H, K, V, grad=True)
        q2, k2, v2, b2 = make_inputs(B, T, H, K, V, grad=True)
        q3, k3, v3, b3 = make_inputs(B, T, H, K, V, grad=True)
        cu = make_cu_seqlens(T, ctx_len, device) if ctx_len else None
        N = len(cu) - 1 if cu is not None else B
        init_scale = torch.zeros(N, H, K, device=device, dtype=torch.float32)

        t_bl = timed_fwd_bwd(chunk_delta_rule, q1, k1, v1, b1, scale,
                             use_qk_l2norm_in_kernel=True, cu_seqlens=cu)

        t_osla = timed_fwd_bwd(chunk_osla_delta_rule, q2, k2, v2, b2, scale,
                               initial_scale=init_scale, use_qk_l2norm_in_kernel=True,
                               cu_seqlens=cu)

        t_osgm = timed_fwd_bwd(chunk_delta_rule_osgm, q3, k3, v3, b3, scale,
                               eta=eta, use_qk_l2norm_in_kernel=True,
                               use_denominator=use_denom, d_min=d_min, d_max=d_max,
                               cu_seqlens=cu)

        ratio = t_osgm / t_bl
        print(f"{label:<40} {t_bl:>8.2f}ms {t_osla:>8.2f}ms {t_osgm:>8.2f}ms {ratio:>7.2f}x")

        del q1, k1, v1, b1, q2, k2, v2, b2, q3, k3, v3, b3, init_scale
        torch.cuda.empty_cache()

    # ====== Part 3: osgm_phase1 isolated ======
    print()
    print("=" * 85)
    print("  OSGM Phase1 (isolated)")
    print("=" * 85)
    print(f"{'Config':<40} {'phase1_fwd':>12} {'l2norm':>12}")
    print("-" * 67)

    for B, T, ctx_len, label in all_configs:
        k = torch.randn(B, T, H, K, device=device, dtype=dtype)
        k_norm, _ = l2norm_fwd(k)
        cu = make_cu_seqlens(T, ctx_len, device) if ctx_len else None
        t_phase1 = timed_fwd(compute_osgm_phase1_fwd, k_norm, eta, use_denom, d_min, d_max, cu)
        t_l2 = timed_fwd(l2norm_fwd, k)

        print(f"{label:<40} {t_phase1:>10.3f}ms {t_l2:>10.3f}ms")

        del k, k_norm
        torch.cuda.empty_cache()


if __name__ == "__main__":
    run()
