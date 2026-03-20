"""
Export BitNet decode step + Build C++ runner.
Strategy: Export the model, do prefill in Python, decode in C++ via imported graph.

Actually simpler: since MLX's import_function works in PYTHON too,
let's first test if the imported function is faster than the normal model call.
If the pre-compiled graph removes Python model traversal overhead, we win.
"""

import mlx.core as mx
from mlx_lm import load, generate
import time
import os

print("Loading model via mlx-lm...")
model, tokenizer = load("models/bitnet-2b")
mx.set_memory_limit(8 * 1024**3)
mx.set_cache_limit(4 * 1024**3)

# Export the forward pass (no cache, single step)
print("Exporting model forward pass...")

# Remove old export
if os.path.exists("bitnet_decode.mlxfn"):
    import shutil
    shutil.rmtree("bitnet_decode.mlxfn", ignore_errors=True)

def forward_fn(input_ids):
    result = model(input_ids)
    if isinstance(result, tuple):
        return result[0]
    return result

# Export with shapeless=True for variable sequence lengths
mx.export_function(
    "bitnet_decode.mlxfn",
    forward_fn,
    mx.array([[42]]),  # single token trace
    shapeless=True,
)
print("Export done!")

# Import the compiled function
print("Importing compiled function...")
compiled_fn = mx.import_function("bitnet_decode.mlxfn")

# Verify correctness
print("Verifying...")
test_in = mx.array([[1, 2, 3, 4, 5]])
out_orig = forward_fn(test_in)
out_compiled = compiled_fn(test_in)
mx.eval(out_orig, out_compiled[0])
import numpy as np
diff = np.abs(np.array(out_orig) - np.array(out_compiled[0])).max()
print(f"  Max diff: {diff:.6f}")
if diff < 0.01:
    print("  ✓ Outputs match!")

# Now benchmark: compiled function (no Python model traversal) vs regular model
print()
print("="*60)
print("Benchmark: Compiled export vs normal model")
print("="*60)

# Warmup
for _ in range(5):
    r = compiled_fn(mx.array([[42]]))
    mx.eval(r[0])
for _ in range(5):
    r = forward_fn(mx.array([[42]]))
    mx.eval(r)

ITERS = 200
token = mx.array([[42]])

# Compiled export
start = time.time()
for _ in range(ITERS):
    r = compiled_fn(token)
    mx.eval(r[0])
compiled_time = time.time() - start
compiled_tps = ITERS / compiled_time

# Normal model
start = time.time()
for _ in range(ITERS):
    r = forward_fn(token)
    mx.eval(r)
normal_time = time.time() - start
normal_tps = ITERS / normal_time

print(f"Compiled export: {ITERS} calls in {compiled_time:.2f}s = {compiled_tps:.1f} calls/s")
print(f"Normal model:    {ITERS} calls in {normal_time:.2f}s = {normal_tps:.1f} calls/s")
print(f"Speedup: {compiled_tps/normal_tps:.2f}x")
print()
print("NOTE: Each call is a FULL forward pass (no KV cache)")
print(f"  This corresponds to ~{compiled_tps:.0f} tok/s decode speed")
print(f"  Previous best (mlx-lm with KV cache): 90 tok/s")
