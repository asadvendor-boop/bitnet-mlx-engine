"""
CoreML ANE Benchmark — Simplified model (no SDPA, no GQA expand)
Just benchmark the matmul-heavy parts to see if ANE can beat GPU.
"""
import torch
import torch.nn as nn
import numpy as np
import mlx.core as mx
import coremltools as ct
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
    """Simplified model: skip attention, just test matmul+norm+ffn path on ANE."""
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        for i in range(TEST_LAYERS):
            # RMS norm
            var = torch.mean(x * x, dim=-1, keepdim=True)
            x_norm = (x * torch.rsqrt(var + 1e-5)) * getattr(self, f'in_{i}')
            
            # QKV projections (just test the matmul speed, skip attention for simplicity)
            q = x_norm @ getattr(self, f'q_{i}').t()
            k = x_norm @ getattr(self, f'k_{i}').t()
            v = x_norm @ getattr(self, f'v_{i}').t()
            
            # Simple proxy for attention (no actual attention, just project back)
            # This tests the matmul throughput on ANE
            attn_out = q  # skip actual attention for CoreML compatibility
            
            # attn sub norm
            var2 = torch.mean(attn_out * attn_out, dim=-1, keepdim=True)
            attn_out = (attn_out * torch.rsqrt(var2 + 1e-5)) * getattr(self, f'asn_{i}')
            
            # O projection
            attn_proj = attn_out @ getattr(self, f'o_{i}').t()
            x = x + attn_proj
            
            # Post attention norm
            var3 = torch.mean(x * x, dim=-1, keepdim=True)
            x_norm2 = (x * torch.rsqrt(var3 + 1e-5)) * getattr(self, f'pan_{i}')
            
            # FFN: gate * up → norm → down
            gate = x_norm2 @ getattr(self, f'gate_{i}').t()
            gate = torch.relu(gate) ** 2
            up = x_norm2 @ getattr(self, f'up_{i}').t()
            hidden = gate * up
            
            var4 = torch.mean(hidden * hidden, dim=-1, keepdim=True)
            hidden = (hidden * torch.rsqrt(var4 + 1e-5)) * getattr(self, f'fsn_{i}')
            
            x = x + (hidden @ getattr(self, f'down_{i}').t())
        
        # Final norm + lm_head
        var5 = torch.mean(x * x, dim=-1, keepdim=True)
        x = (x * torch.rsqrt(var5 + 1e-5)) * self.final_norm
        return x @ self.embed.t()

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

print("Tracing...")
model.eval()
dummy = torch.randn(1, 1, HIDDEN, dtype=torch.float16)
with torch.no_grad():
    traced = torch.jit.trace(model, dummy)

print("Converting to CoreML...")
mlmodel = ct.convert(
    traced,
    inputs=[ct.TensorType(name="x", shape=(1, 1, HIDDEN), dtype=np.float16)],
    outputs=[ct.TensorType(name="logits", dtype=np.float16)],
    compute_units=ct.ComputeUnit.ALL,
    minimum_deployment_target=ct.target.macOS15,
    convert_to="mlprogram",
)
mlmodel.save("models/bitnet_ane_test.mlpackage")
print("Saved!")

# Benchmark
for label, unit in [
    ("ALL (ANE+GPU+CPU)", ct.ComputeUnit.ALL),
    ("ANE+CPU only", ct.ComputeUnit.CPU_AND_NE),
    ("GPU+CPU only", ct.ComputeUnit.CPU_AND_GPU),
]:
    print(f"\n=== {label} ===")
    m = ct.models.MLModel("models/bitnet_ane_test.mlpackage", compute_units=unit)
    inp = {"x": np.random.randn(1, 1, HIDDEN).astype(np.float16)}
    
    for _ in range(20): m.predict(inp)
    
    t0 = time.time()
    N = 200
    for _ in range(N): m.predict(inp)
    elapsed = time.time() - t0
    ms = elapsed / N * 1000
    
    print(f"  {ms:.2f} ms/step ({TEST_LAYERS} layers)")
    estimated_full = ms * 30 / TEST_LAYERS
    print(f"  Estimated full 30-layer: {estimated_full:.1f} ms → {1000/estimated_full:.0f} tok/s")
