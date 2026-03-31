import torch
import torch.nn.functional as F
import time

# 导入你的两个 Triton 算子
from fla.ops.osla_delta_rule.fused_recurrent_osgm import fused_recurrent_delta_rule_osgm
from fla.ops.osla_delta_rule.chunk_osgm import chunk_delta_rule_osgm

def benchmark_fn(fn, *args, num_warmup=10, num_iters=50, backward=False, **kwargs):
    """标准的 PyTorch CUDA 精确测速函数"""
    # 预热 (Warmup) 消除初始化和内核编译带来的时间抖动
    for _ in range(num_warmup):
        out = fn(*args, **kwargs)
        if backward:
            loss = out[0].sum()
            loss.backward(retain_graph=True)
            
            # 清理梯度
            for arg in args:
                if isinstance(arg, torch.Tensor) and arg.grad is not None:
                    arg.grad = None
                    
    torch.cuda.synchronize()
    
    # 记录时间
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
    
    for i in range(num_iters):
        # 每次反向传播前清空梯度
        if backward:
            for arg in args:
                if isinstance(arg, torch.Tensor) and arg.grad is not None:
                    arg.grad = None

        start_events[i].record()
        
        out = fn(*args, **kwargs)
        if backward:
            loss = out[0].sum()
            loss.backward()
            
        end_events[i].record()
        
    torch.cuda.synchronize()
    
    # 计算平均耗时 (毫秒)
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    avg_time = sum(times) / num_iters
    return avg_time

def run_benchmark():
    # 为了测试真实的硬件性能，必须使用 bf16 (Triton 会在 bf16 下启用 Tensor Cores)
    dtype = torch.bfloat16
    device = 'cuda'
    
    # 固定参数 (模拟真实的大模型配置)
    B = 4           # Batch Size
    H = 8           # Attention Heads
    K = 128         # Head Dimension (Key/Query)
    V = 128         # Head Dimension (Value)
    scale = K ** -0.5
    eta = 1.0 
    use_denom = False
    
    # 扫描不同的序列长度
    seq_lengths = [1024, 2048, 4096, 8192]
    
    print(f"=== OSGM Delta Rule 性能基准测试 ===")
    print(f"环境配置: B={B}, H={H}, K={K}, V={V}, Dtype={dtype}\n")
    
    print(f"{'Seq Length (T)':<15} | {'Recurrent FWD':<15} | {'Chunk FWD':<15} | {'Speedup':<10}")
    print("-" * 65)
    
    for T in seq_lengths:
        q = torch.randn(B, T, H, K, device=device, dtype=dtype)
        k = F.normalize(torch.randn(B, T, H, K, device=device, dtype=dtype), p=2, dim=-1)
        v = torch.randn(B, T, H, V, device=device, dtype=dtype)
        beta = torch.rand(B, T, H, device=device, dtype=dtype).sigmoid()
        
        # 测速 Forward
        time_rec_fwd = benchmark_fn(
            fused_recurrent_delta_rule_osgm, 
            q, k, v, beta, scale, eta=eta, use_qk_l2norm_in_kernel=False, use_denominator=use_denom
        )
        time_chk_fwd = benchmark_fn(
            chunk_delta_rule_osgm, 
            q, k, v, beta, scale, eta=eta, use_qk_l2norm_in_kernel=False, use_denominator=use_denom
        )
        
        speedup = time_rec_fwd / time_chk_fwd
        print(f"{T:<15} | {time_rec_fwd:>12.2f} ms | {time_chk_fwd:>12.2f} ms | {speedup:>8.2f}x")

    print("\n" + "="*65 + "\n")
    
    print(f"{'Seq Length (T)':<15} | {'Recurrent FWD+BWD':<17} | {'Chunk FWD+BWD':<17} | {'Speedup':<10}")
    print("-" * 69)
    
    for T in seq_lengths:
        q = torch.randn(B, T, H, K, device=device, dtype=dtype, requires_grad=True)
        k = F.normalize(torch.randn(B, T, H, K, device=device, dtype=dtype), p=2, dim=-1).detach().requires_grad_(True)
        v = torch.randn(B, T, H, V, device=device, dtype=dtype, requires_grad=True)
        beta = torch.rand(B, T, H, device=device, dtype=dtype).sigmoid().detach().requires_grad_(True)
        
        # 测速 Forward + Backward
        time_rec_bwd = benchmark_fn(
            fused_recurrent_delta_rule_osgm, 
            q, k, v, beta, scale, eta=eta, use_qk_l2norm_in_kernel=False, use_denominator=use_denom, backward=True
        )
        time_chk_bwd = benchmark_fn(
            chunk_delta_rule_osgm, 
            q, k, v, beta, scale, eta=eta, use_qk_l2norm_in_kernel=False, use_denominator=use_denom, backward=True
        )
        
        speedup = time_rec_bwd / time_chk_bwd
        print(f"{T:<15} | {time_rec_bwd:>14.2f} ms | {time_chk_bwd:>14.2f} ms | {speedup:>8.2f}x")

if __name__ == "__main__":
    run_benchmark()