"""
Benchmark: Metal GPU ternary matmul vs naive MLX operations.
Measures speed and confirms GPU is being used.
"""

import time
import mlx.core as mx
import numpy as np

from bitnet.kernels import pack_ternary_weights, ternary_matmul, ternary_matmul_naive


def benchmark_ternary_matmul(
    out_features: int = 4096,
    in_features: int = 4096,
    batch_size: int = 1,
    iterations: int = 50,
    warmup: int = 5,
):
    """Benchmark the Metal GPU ternary matmul kernel."""
    print(f"\n{'='*60}")
    print(f"Ternary MatMul Benchmark")
    print(f"Matrix: ({out_features}, {in_features}) @ ({in_features},)")
    print(f"Batch size: {batch_size}")
    print(f"Iterations: {iterations} (+ {warmup} warmup)")
    print(f"{'='*60}\n")
    
    # Create random ternary weights
    rng = np.random.default_rng(42)
    w_ternary = rng.choice([-1, 0, 1], size=(out_features, in_features)).astype(np.float32)
    packed = pack_ternary_weights(mx.array(w_ternary))
    scale = mx.array([0.5], dtype=mx.float32)
    
    # Create random activations
    if batch_size == 1:
        x = mx.random.normal((in_features,), dtype=mx.float16)
    else:
        x = mx.random.normal((batch_size, in_features), dtype=mx.float16)
    
    # --- Warmup ---
    for _ in range(warmup):
        out = ternary_matmul(packed, x, scale)
        mx.eval(out)
    
    # --- Metal GPU Kernel ---
    print("Running Metal GPU kernel benchmark...")
    start = time.perf_counter()
    for _ in range(iterations):
        out_gpu = ternary_matmul(packed, x, scale)
        mx.eval(out_gpu)
    gpu_time = (time.perf_counter() - start) / iterations
    
    # --- Naive (unpack + standard matmul) ---
    print("Running naive MLX matmul benchmark...")
    for _ in range(warmup):
        out = ternary_matmul_naive(packed, x, scale, in_features)
        mx.eval(out)
    
    start = time.perf_counter()
    for _ in range(iterations):
        out_naive = ternary_matmul_naive(packed, x, scale, in_features)
        mx.eval(out_naive)
    naive_time = (time.perf_counter() - start) / iterations
    
    # --- Standard float16 matmul for reference ---
    print("Running standard float16 matmul benchmark...")
    w_float = mx.array(w_ternary, dtype=mx.float16)
    for _ in range(warmup):
        if batch_size == 1:
            out = w_float @ x
        else:
            out = x @ w_float.T
        mx.eval(out)
    
    start = time.perf_counter()
    for _ in range(iterations):
        if batch_size == 1:
            out_std = w_float @ x
        else:
            out_std = x @ w_float.T
        mx.eval(out_std)
    std_time = (time.perf_counter() - start) / iterations
    
    # --- Results ---
    print(f"\n{'='*60}")
    print(f"Results (average per iteration):")
    print(f"{'='*60}")
    print(f"  Metal GPU ternary kernel: {gpu_time*1000:.3f} ms")
    print(f"  Naive (unpack + matmul):  {naive_time*1000:.3f} ms")
    print(f"  Standard float16 matmul:  {std_time*1000:.3f} ms")
    print(f"")
    print(f"  GPU vs Naive speedup:     {naive_time/gpu_time:.2f}x")
    print(f"  GPU vs Float16 speedup:   {std_time/gpu_time:.2f}x")
    print(f"")
    
    # Memory comparison
    packed_bytes = packed.nbytes
    float_bytes = w_float.nbytes
    print(f"Memory:")
    print(f"  Packed ternary (2-bit):   {packed_bytes/1e6:.2f} MB")
    print(f"  Standard float16:         {float_bytes/1e6:.2f} MB")
    print(f"  Compression:              {float_bytes/packed_bytes:.1f}x smaller")
    print(f"{'='*60}")
    
    # Verify correctness
    mx.eval(out_gpu, out_naive)
    if out_gpu.ndim > 1:
        diff = mx.max(mx.abs(out_gpu.astype(mx.float32) - out_naive.astype(mx.float32))).item()
    else:
        diff = mx.max(mx.abs(out_gpu.astype(mx.float32) - out_naive.astype(mx.float32))).item()
    print(f"  Max diff (GPU vs naive):  {diff:.6f}")
    print(f"  Correct: {'✅ YES' if diff < 0.1 else '❌ NO'}")
    
    return {
        "gpu_ms": gpu_time * 1000,
        "naive_ms": naive_time * 1000,
        "std_ms": std_time * 1000,
        "speedup_vs_naive": naive_time / gpu_time,
        "compression": float_bytes / packed_bytes,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark BitNet Metal kernels")
    parser.add_argument("--size", type=int, default=4096, help="Matrix size")
    parser.add_argument("--batch", type=int, default=1, help="Batch size")
    parser.add_argument("--iterations", type=int, default=50, help="Number of iterations")
    args = parser.parse_args()
    
    # Print system info
    print(f"MLX version: {mx.__version__}")
    print(f"Metal available: {mx.metal.is_available()}")
    print(f"Device: {mx.default_device()}")
    
    # Run benchmarks at different sizes
    sizes = [args.size] if args.size != 4096 else [1024, 2048, 4096]
    
    for size in sizes:
        benchmark_ternary_matmul(
            out_features=size,
            in_features=size,
            batch_size=args.batch,
            iterations=args.iterations,
        )
