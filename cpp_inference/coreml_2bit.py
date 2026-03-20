"""
CoreML with 2-bit Palettization — Proper ANE Benchmark
Instead of dequantizing to float16, use CoreML's native 2-bit palettization
which maps 4 values (perfect for ternary {-1, 0, 1} + padding).
"""
import torch
import torch.nn as nn
import numpy as np
import mlx.core as mx
import coremltools as ct
from coremltools.optimize.coreml import (
    OpPalettizerConfig, OptimizationConfig, palettize_weights
)
import time

HIDDEN = 2560
NUM_HEADS = 20
NUM_KV_HEADS = 5
INTERMEDIATE = 6912
HEAD_DIM = 128
TEST_LAYERS = 2

def mlx_to_np(arr):
    return np.array(arr.astype(mx.float16))

def unpack_ternary_mlx(packed_w, weight_scale, out_features, in_features):
    packed = np.array(packed_w)
    scale = float(np.array(weight_scale.astype(mx.float32)).item())
    packed_rows = packed.shape[0]
    result = np.zeros((out_features, in_features), dtype=np.float16)
    for r in range(packed_rows):
        for c in range(in_features):
            byte = int(packed[r, c])
            for j in range(4):
                v = ((byte >> (2*j)) & 3) - 1
                idx = r + j * packed_rows
                if idx < out_features:
                    result[idx, c] = v * scale
    return result

class BitNetSimple(nn.Module):
    """Simplified model for CoreML conversion."""
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        for i in range(TEST_LAYERS):
            # RMS norm
            var = torch.mean(x * x, dim=-1, keepdim=True)
            x_norm = (x * torch.rsqrt(var + 1e-5)) * getattr(self, f'in_{i}')
            
            # Projections
            q = x_norm @ getattr(self, f'q_{i}').t()
            k = x_norm @ getattr(self, f'k_{i}').t()
            v = x_norm @ getattr(self, f'v_{i}').t()
            
            # Skip actual attention for CoreML compat, just test matmul throughput
            attn_out = q
            var2 = torch.mean(attn_out * attn_out, dim=-1, keepdim=True)
            attn_out = (attn_out * torch.rsqrt(var2 + 1e-5)) * getattr(self, f'asn_{i}')
            attn_proj = attn_out @ getattr(self, f'o_{i}').t()
            x = x + attn_proj
            
            # Post-attn norm + FFN
            var3 = torch.mean(x * x, dim=-1, keepdim=True)
            x_norm2 = (x * torch.rsqrt(var3 + 1e-5)) * getattr(self, f'pan_{i}')
            gate = torch.relu(x_norm2 @ getattr(self, f'gate_{i}').t()) ** 2
            up = x_norm2 @ getattr(self, f'up_{i}').t()
            hidden = gate * up
            var4 = torch.mean(hidden * hidden, dim=-1, keepdim=True)
            hidden = (hidden * torch.rsqrt(var4 + 1e-5)) * getattr(self, f'fsn_{i}')
            x = x + (hidden @ getattr(self, f'down_{i}').t())
        
        var5 = torch.mean(x * x, dim=-1, keepdim=True)
        x = (x * torch.rsqrt(var5 + 1e-5)) * self.final_norm
        return x @ self.embed.t()

# Load weights
print("Loading weights...")
wmap = mx.load("models/bitnet-2b/model.safetensors")
model = BitNetSimple()
model.register_buffer('embed', torch.from_numpy(mlx_to_np(wmap["model.embed_tokens.weight"])))
model.register_buffer('final_norm', torch.from_numpy(mlx_to_np(wmap["model.norm.weight"])))

for i in range(TEST_LAYERS):
    print(f"  Layer {i}...")
    p = f"model.layers.{i}"
    model.register_buffer(f'in_{i}', torch.from_numpy(mlx_to_np(wmap[f"{p}.input_layernorm.weight"])))
    model.register_buffer(f'pan_{i}', torch.from_numpy(mlx_to_np(wmap[f"{p}.post_attention_layernorm.weight"])))
    model.register_buffer(f'asn_{i}', torch.from_numpy(mlx_to_np(wmap[f"{p}.self_attn.attn_sub_norm.weight"])))
    model.register_buffer(f'fsn_{i}', torch.from_numpy(mlx_to_np(wmap[f"{p}.mlp.ffn_sub_norm.weight"])))
    
    for name, key, inf, outf in [
        ('q', 'self_attn.q_proj', HIDDEN, NUM_HEADS * HEAD_DIM),
        ('k', 'self_attn.k_proj', HIDDEN, NUM_KV_HEADS * HEAD_DIM),
        ('v', 'self_attn.v_proj', HIDDEN, NUM_KV_HEADS * HEAD_DIM),
        ('o', 'self_attn.o_proj', NUM_HEADS * HEAD_DIM, HIDDEN),
        ('gate', 'mlp.gate_proj', HIDDEN, INTERMEDIATE),
        ('up', 'mlp.up_proj', HIDDEN, INTERMEDIATE),
        ('down', 'mlp.down_proj', INTERMEDIATE, HIDDEN),
    ]:
        packed = wmap[f"{p}.{key}.weight"]
        scale = wmap[f"{p}.{key}.weight_scale"]
        w = unpack_ternary_mlx(packed, scale, outf, inf)
        model.register_buffer(f'{name}_{i}', torch.from_numpy(w))

# Trace
print("Tracing...")
model.eval()
dummy = torch.randn(1, 1, HIDDEN, dtype=torch.float16)
with torch.no_grad():
    traced = torch.jit.trace(model, dummy)

# Convert to CoreML (float16 first)
print("Converting to CoreML (float16 baseline)...")
mlmodel_fp16 = ct.convert(
    traced,
    inputs=[ct.TensorType(name="x", shape=(1, 1, HIDDEN), dtype=np.float16)],
    outputs=[ct.TensorType(name="logits", dtype=np.float16)],
    compute_units=ct.ComputeUnit.ALL,
    minimum_deployment_target=ct.target.macOS15,
    convert_to="mlprogram",
)
mlmodel_fp16.save("models/bitnet_fp16.mlpackage")

# Apply 2-bit palettization
print("Applying 2-bit palettization...")
op_config = OpPalettizerConfig(mode="kmeans", nbits=2)
config = OptimizationConfig(global_config=op_config)
mlmodel_2bit = palettize_weights(mlmodel_fp16, config)
mlmodel_2bit.save("models/bitnet_2bit.mlpackage")
print("Saved both models!")

# Benchmark all variants
results = []
for label, path, unit in [
    ("FP16 + ALL", "models/bitnet_fp16.mlpackage", ct.ComputeUnit.ALL),
    ("FP16 + ANE", "models/bitnet_fp16.mlpackage", ct.ComputeUnit.CPU_AND_NE),
    ("FP16 + GPU", "models/bitnet_fp16.mlpackage", ct.ComputeUnit.CPU_AND_GPU),
    ("2-bit + ALL", "models/bitnet_2bit.mlpackage", ct.ComputeUnit.ALL),
    ("2-bit + ANE", "models/bitnet_2bit.mlpackage", ct.ComputeUnit.CPU_AND_NE),
    ("2-bit + GPU", "models/bitnet_2bit.mlpackage", ct.ComputeUnit.CPU_AND_GPU),
]:
    print(f"\n=== {label} ===")
    m = ct.models.MLModel(path, compute_units=unit)
    
    inp = {"x": np.random.randn(1, 1, HIDDEN).astype(np.float16)}
    for _ in range(30): m.predict(inp)
    
    t0 = time.time()
    N = 300
    for _ in range(N): m.predict(inp)
    elapsed = time.time() - t0
    ms = elapsed / N * 1000
    est = ms * 30 / TEST_LAYERS
    tok_s = 1000 / est
    
    print(f"  {ms:.2f} ms/step ({TEST_LAYERS} layers)")
    print(f"  Est. full model: {est:.1f} ms → {tok_s:.0f} tok/s")
    results.append((label, ms, est, tok_s))

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
for label, ms, est, tok_s in results:
    print(f"  {label:20s}: {ms:.2f} ms/step → est. {tok_s:.0f} tok/s")
print(f"\n  MLX GPU (reference):  ~0.75 ms/step → 89 tok/s")
