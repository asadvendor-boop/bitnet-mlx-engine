"""
Kernel tuning: test different configurations to find the sweet spot.
"""
import time
import mlx.core as mx
import numpy as np

from bitnet.kernels import pack_ternary_weights


def make_kernel(rows_per_thread, use_half_acc=False):
    """Generate a ternary matvec kernel with configurable parameters."""
    
    # Generate the row processing code
    acc_decls = ", ".join([f"acc{i} = 0.0f" for i in range(rows_per_thread)])
    
    load_lines = [f"uint p0 = packed_weights[(base_row + 0) * packed_cols + pc];"]
    for i in range(1, rows_per_thread):
        load_lines.append(
            f"uint p{i} = (rows > {i}) ? packed_weights[(base_row + {i}) * packed_cols + pc] : 0;"
        )
    loads = "\n        ".join(load_lines)
    
    decode_lines = []
    for i in range(rows_per_thread):
        decode_lines.append(
            f"uint bv{i} = (p{i} >> shift) & 0x3; "
            f"acc{i} += float(int(bv{i} & 1u) - int(bv{i} >> 1u)) * val;"
        )
    decodes = "\n            ".join(decode_lines)
    
    store_lines = [f"out[base_row + 0] = static_cast<T>(acc0 * s);"]
    for i in range(1, rows_per_thread):
        store_lines.append(
            f"if (rows > {i}) out[base_row + {i}] = static_cast<T>(acc{i} * s);"
        )
    stores = "\n    ".join(store_lines)
    
    source = f"""
    uint tid = thread_position_in_grid.x;
    uint out_features = packed_weights_shape[0];
    uint packed_cols = packed_weights_shape[1];
    uint in_feat = x_shape[0];
    
    uint base_row = tid * {rows_per_thread};
    if (base_row >= out_features) return;
    uint rows = min({rows_per_thread}u, out_features - base_row);
    
    float {acc_decls};
    
    for (uint pc = 0; pc < packed_cols; pc++) {{
        {loads}
        
        uint base_col = pc * 16;
        
        for (uint i = 0; i < 16; i++) {{
            uint col = base_col + i;
            if (col >= in_feat) break;
            
            float val = static_cast<float>(x[col]);
            uint shift = i * 2;
            
            {decodes}
        }}
    }}
    
    float s = static_cast<float>(scale[0]);
    {stores}
"""
    
    kernel = mx.fast.metal_kernel(
        name=f"ternary_matvec_r{rows_per_thread}",
        input_names=["packed_weights", "x", "scale"],
        output_names=["out"],
        source=source,
    )
    return kernel, rows_per_thread


def bench_kernel(kernel, rows_per_thread, packed, x, scale, out_features, iterations=200, warmup=10):
    """Benchmark a specific kernel configuration."""
    num_threads = (out_features + rows_per_thread - 1) // rows_per_thread
    
    for _ in range(warmup):
        out = kernel(
            inputs=[packed, x, scale],
            template=[("T", x.dtype)],
            grid=(num_threads, 1, 1),
            threadgroup=(min(256, num_threads), 1, 1),
            output_shapes=[(out_features,)],
            output_dtypes=[x.dtype],
        )[0]
        mx.eval(out)
    
    start = time.perf_counter()
    for _ in range(iterations):
        out = kernel(
            inputs=[packed, x, scale],
            template=[("T", x.dtype)],
            grid=(num_threads, 1, 1),
            threadgroup=(min(256, num_threads), 1, 1),
            output_shapes=[(out_features,)],
            output_dtypes=[x.dtype],
        )[0]
        mx.eval(out)
    elapsed = (time.perf_counter() - start) / iterations
    return elapsed, out


if __name__ == "__main__":
    N = 4096
    rng = np.random.default_rng(42)
    w = rng.choice([-1, 0, 1], size=(N, N)).astype(np.float32)
    packed = pack_ternary_weights(mx.array(w))
    x = mx.random.normal((N,), dtype=mx.float16)
    scale = mx.array([0.5], dtype=mx.float32)
    
    print(f"Matrix: ({N}, {N}) @ ({N},)")
    print(f"MLX {mx.__version__} | Device: {mx.default_device()}")
    print(f"{'='*60}")
    
    # Also benchmark standard float16 matmul
    w_float = mx.array(w, dtype=mx.float16)
    for _ in range(10):
        out = w_float @ x
        mx.eval(out)
    start = time.perf_counter()
    for _ in range(200):
        out = w_float @ x
        mx.eval(out)
    f16_time = (time.perf_counter() - start) / 200
    print(f"  {'Float16 matmul':<30} {f16_time*1000:.3f} ms (baseline)")
    print(f"{'='*60}")
    
    # Test different rows-per-thread
    for rpt in [1, 2, 4, 6, 8, 12, 16]:
        try:
            kernel, rpt_actual = make_kernel(rpt)
            elapsed, out = bench_kernel(kernel, rpt_actual, packed, x, scale, N)
            ratio = f16_time / elapsed
            print(f"  rows_per_thread={rpt:>2} {elapsed*1000:.3f} ms  ({ratio:.2f}x vs f16)  "
                  f"threads={N//rpt:>5}")
        except Exception as e:
            print(f"  rows_per_thread={rpt:>2} FAILED: {e}")
    
    # Also test different threadgroup sizes with best RPT
    print(f"\n{'='*60}")
    print(f"Threadgroup size sweep (rows_per_thread=4):")
    kernel, rpt = make_kernel(4)
    num_threads = N // 4
    for tg_size in [32, 64, 128, 256, 512, 1024]:
        try:
            for _ in range(10):
                out = kernel(
                    inputs=[packed, x, scale],
                    template=[("T", x.dtype)],
                    grid=(num_threads, 1, 1),
                    threadgroup=(min(tg_size, num_threads), 1, 1),
                    output_shapes=[(N,)],
                    output_dtypes=[x.dtype],
                )[0]
                mx.eval(out)
            
            start = time.perf_counter()
            for _ in range(200):
                out = kernel(
                    inputs=[packed, x, scale],
                    template=[("T", x.dtype)],
                    grid=(num_threads, 1, 1),
                    threadgroup=(min(tg_size, num_threads), 1, 1),
                    output_shapes=[(N,)],
                    output_dtypes=[x.dtype],
                )[0]
                mx.eval(out)
            elapsed = (time.perf_counter() - start) / 200
            ratio = f16_time / elapsed
            print(f"  threadgroup={tg_size:<5} {elapsed*1000:.3f} ms  ({ratio:.2f}x vs f16)")
        except Exception as e:
            print(f"  threadgroup={tg_size:<5} FAILED: {e}")
