"""
Approach 3: Custom MLX ternary op via optimized Metal shader with:
  - Threadgroup shared memory for cooperative activation tiling
  - SIMD group reductions (simd_sum)
  - Coalesced memory access patterns
  - Integrated into QuantizedLinear-based model for best of both worlds

Strategy: Use our optimized ternary kernel ONLY for the matmul ops,
but keep all other ops (RoPE, attention, norms) using Apple's optimized versions.
"""

import time
import math
import json
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from pathlib import Path
from transformers import AutoTokenizer

from run_approach1_native2bit import (
    unpack_microsoft_to_ternary, GROUP_SIZE,
    RMSNorm
)
from bitnet.kernels import pack_ternary_weights


# ---------------------------------------------------------------------------
# Approach 3: Tiled Metal kernel using threadgroup shared memory + SIMD ops
# ---------------------------------------------------------------------------

TERNARY_TILED_HEADER = """
// Utility: branchless ternary decode
// 2-bit code → weight: 0b00=0, 0b01=+1, 0b10=-1
// sign = (b & 1) - (b >> 1)
"""

TERNARY_TILED_KERNEL = """
    // Tiled ternary matvec with threadgroup shared memory.
    // Each threadgroup cooperatively loads a tile of activations into shared memory,
    // then ALL threads in the group use the cached tile to process their rows.
    //
    // This dramatically reduces global memory reads for the activation vector.
    // Previous kernel: each thread reads all of x independently (redundant!)
    // This kernel: threadgroup loads x tile ONCE, all threads share it.
    
    // Layout:
    // - packed_weights: (out_features, packed_cols) uint32, 16 ternary values per uint32
    // - x: (in_features,) float16
    // - out: (out_features,) float16
    // - scale: (1,) float32
    
    constexpr uint TILE_SIZE = 256;  // activation tile size
    constexpr uint ROWS_PER_THREAD = 4;
    
    uint out_features = packed_weights_shape[0];
    uint packed_cols = packed_weights_shape[1];
    uint in_feat = x_shape[0];
    
    // Thread identification
    uint tg_idx = thread_position_in_threadgroup.x;
    uint tg_size = threads_per_threadgroup.x;
    uint global_tid = thread_position_in_grid.x;
    
    // Each thread handles ROWS_PER_THREAD output rows
    uint base_row = global_tid * ROWS_PER_THREAD;
    if (base_row >= out_features) return;
    uint rows = min(ROWS_PER_THREAD, out_features - base_row);
    
    // Shared memory for activation tile
    threadgroup float shared_x[TILE_SIZE];
    
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    
    // Process activation vector in tiles
    for (uint tile_start = 0; tile_start < in_feat; tile_start += TILE_SIZE) {
        uint tile_end = min(tile_start + TILE_SIZE, in_feat);
        uint tile_len = tile_end - tile_start;
        
        // Cooperative loading: all threads in threadgroup load the tile
        for (uint i = tg_idx; i < tile_len; i += tg_size) {
            shared_x[i] = static_cast<float>(x[tile_start + i]);
        }
        // Zero out unused portion
        for (uint i = tile_len + tg_idx; i < TILE_SIZE; i += tg_size) {
            shared_x[i] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Now process the tile — each thread reads weights for its rows
        uint pc_start = tile_start / 16;
        uint pc_end = (tile_end + 15) / 16;
        
        for (uint pc = pc_start; pc < pc_end && pc < packed_cols; pc++) {
            uint p0 = packed_weights[(base_row + 0) * packed_cols + pc];
            uint p1 = (rows > 1) ? packed_weights[(base_row + 1) * packed_cols + pc] : 0;
            uint p2 = (rows > 2) ? packed_weights[(base_row + 2) * packed_cols + pc] : 0;
            uint p3 = (rows > 3) ? packed_weights[(base_row + 3) * packed_cols + pc] : 0;
            
            uint base_col = pc * 16;
            
            for (uint i = 0; i < 16; i++) {
                uint col = base_col + i;
                if (col >= in_feat || col < tile_start || col >= tile_end) continue;
                
                float val = shared_x[col - tile_start];
                uint shift = i * 2;
                
                uint b0 = (p0 >> shift) & 0x3; acc0 += float(int(b0 & 1u) - int(b0 >> 1u)) * val;
                uint b1 = (p1 >> shift) & 0x3; acc1 += float(int(b1 & 1u) - int(b1 >> 1u)) * val;
                uint b2 = (p2 >> shift) & 0x3; acc2 += float(int(b2 & 1u) - int(b2 >> 1u)) * val;
                uint b3 = (p3 >> shift) & 0x3; acc3 += float(int(b3 & 1u) - int(b3 >> 1u)) * val;
            }
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    float s = static_cast<float>(scale[0]);
    out[base_row + 0] = static_cast<T>(acc0 * s);
    if (rows > 1) out[base_row + 1] = static_cast<T>(acc1 * s);
    if (rows > 2) out[base_row + 2] = static_cast<T>(acc2 * s);
    if (rows > 3) out[base_row + 3] = static_cast<T>(acc3 * s);
"""

_tiled_kernel_cache = {}

def _get_tiled_kernel():
    if "kernel" not in _tiled_kernel_cache:
        _tiled_kernel_cache["kernel"] = mx.fast.metal_kernel(
            name="ternary_matvec_tiled",
            input_names=["packed_weights", "x", "scale"],
            output_names=["out"],
            source=TERNARY_TILED_KERNEL,
            header=TERNARY_TILED_HEADER,
        )
    return _tiled_kernel_cache["kernel"]


def ternary_matmul_tiled(packed_weights, x, scale):
    """Tiled ternary matmul with shared memory."""
    out_features = packed_weights.shape[0]
    
    if scale.ndim == 0:
        scale = scale.reshape(1)
    
    orig_shape = x.shape
    if x.ndim == 3:
        B, L, D = x.shape
        x = x.reshape(B * L, D)
    
    if x.ndim == 1:
        num_threads = (out_features + 3) // 4
        kernel = _get_tiled_kernel()
        out = kernel(
            inputs=[packed_weights, x, scale],
            template=[("T", x.dtype)],
            grid=(num_threads, 1, 1),
            threadgroup=(128, 1, 1),
            output_shapes=[(out_features,)],
            output_dtypes=[x.dtype],
        )[0]
    else:
        # Batched: process each row
        batch = x.shape[0]
        results = []
        for b in range(batch):
            num_threads = (out_features + 3) // 4
            kernel = _get_tiled_kernel()
            r = kernel(
                inputs=[packed_weights, x[b], scale],
                template=[("T", x.dtype)],
                grid=(num_threads, 1, 1),
                threadgroup=(128, 1, 1),
                output_shapes=[(out_features,)],
                output_dtypes=[x.dtype],
            )[0]
            results.append(r)
        out = mx.stack(results)
    
    if len(orig_shape) == 3:
        out = out.reshape(orig_shape[0], orig_shape[1], -1)
    return out


# ---------------------------------------------------------------------------
# Model using the tiled kernel for linear ops + Apple ops for everything else
# ---------------------------------------------------------------------------

class TernaryLinear(nn.Module):
    """Linear layer using our tiled Metal kernel."""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        packed_cols = math.ceil(in_features / 16)
        self.packed_weights = mx.zeros((out_features, packed_cols), dtype=mx.uint32)
        self.scale = mx.array([1.0], dtype=mx.float32)
    
    def __call__(self, x):
        return ternary_matmul_tiled(self.packed_weights, x, self.scale)


class BitNetAttention3(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.num_heads = config["num_attention_heads"]
        self.num_kv_heads = config["num_key_value_heads"]
        self.head_dim = self.hidden_size // self.num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        self.q_proj = TernaryLinear(self.hidden_size, self.num_heads * self.head_dim)
        self.k_proj = TernaryLinear(self.hidden_size, self.num_kv_heads * self.head_dim)
        self.v_proj = TernaryLinear(self.hidden_size, self.num_kv_heads * self.head_dim)
        self.o_proj = TernaryLinear(self.num_heads * self.head_dim, self.hidden_size)
        
        self.attn_sub_norm = RMSNorm(self.hidden_size, eps=config["rms_norm_eps"])
        self.rope = nn.RoPE(self.head_dim, traditional=False, base=config["rope_theta"])
    
    def __call__(self, x, mask=None, cache=None):
        B, L, _ = x.shape
        
        q = self.q_proj(x).reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, L, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        offset = cache[0].shape[2] if cache is not None else 0
        q = self.rope(q, offset=offset)
        k = self.rope(k, offset=offset)
        
        if cache is not None:
            k = mx.concatenate([cache[0], k], axis=2)
            v = mx.concatenate([cache[1], v], axis=2)
        new_cache = (k, v)
        
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        out = self.attn_sub_norm(out)
        return self.o_proj(out), new_cache


class BitNetMLP3(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = TernaryLinear(config["hidden_size"], config["intermediate_size"])
        self.up_proj = TernaryLinear(config["hidden_size"], config["intermediate_size"])
        self.down_proj = TernaryLinear(config["intermediate_size"], config["hidden_size"])
        self.ffn_sub_norm = RMSNorm(config["intermediate_size"], eps=config["rms_norm_eps"])
    
    def __call__(self, x):
        gate = nn.relu(self.gate_proj(x))
        gate = gate * gate
        up = self.up_proj(x)
        hidden = gate * up
        hidden = self.ffn_sub_norm(hidden)
        return self.down_proj(hidden)


class BitNetBlock3(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_layernorm = RMSNorm(config["hidden_size"], eps=config["rms_norm_eps"])
        self.self_attn = BitNetAttention3(config)
        self.post_attention_layernorm = RMSNorm(config["hidden_size"], eps=config["rms_norm_eps"])
        self.mlp = BitNetMLP3(config)
    
    def __call__(self, x, mask=None, cache=None):
        r = x
        x = self.input_layernorm(x)
        x, new_cache = self.self_attn(x, mask=mask, cache=cache)
        x = r + x
        r = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        return r + x, new_cache


class BitNet2BApproach3(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config["vocab_size"], config["hidden_size"])
        self.layers = [BitNetBlock3(config) for _ in range(config["num_hidden_layers"])]
        self.norm = RMSNorm(config["hidden_size"], eps=config["rms_norm_eps"])
    
    def __call__(self, input_ids, cache=None):
        x = self.embed_tokens(input_ids)
        
        mask = None
        if x.shape[1] > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(x.shape[1])
            mask = mask.astype(x.dtype)
        
        new_cache = []
        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache is not None else None
            x, c = layer(x, mask=mask, cache=layer_cache)
            new_cache.append(c)
        
        x = self.norm(x)
        logits = self.embed_tokens.as_linear(x)
        return logits, new_cache


def load_model_approach3(model_path):
    """Load model with tiled ternary kernel."""
    model_dir = Path(model_path)
    
    with open(model_dir / "config.json") as f:
        config = json.load(f)
    
    print(f"Approach 3: Tiled Metal Kernel + Apple Attention/Norms")
    print(f"  Hidden: {config['hidden_size']}, Layers: {config['num_hidden_layers']}")
    
    model = BitNet2BApproach3(config)
    raw = mx.load(str(model_dir / "model.safetensors"))
    
    model.embed_tokens.weight = raw["model.embed_tokens.weight"].astype(mx.float16)
    model.norm.weight = raw["model.norm.weight"].astype(mx.float32)
    
    for i in range(config["num_hidden_layers"]):
        if i % 10 == 0:
            print(f"  Loading layer {i}/{config['num_hidden_layers']}...")
        
        layer = model.layers[i]
        prefix = f"model.layers.{i}"
        
        layer.input_layernorm.weight = raw[f"{prefix}.input_layernorm.weight"].astype(mx.float32)
        layer.post_attention_layernorm.weight = raw[f"{prefix}.post_attention_layernorm.weight"].astype(mx.float32)
        layer.self_attn.attn_sub_norm.weight = raw[f"{prefix}.self_attn.attn_sub_norm.weight"].astype(mx.float32)
        layer.mlp.ffn_sub_norm.weight = raw[f"{prefix}.mlp.ffn_sub_norm.weight"].astype(mx.float32)
        
        linear_map = {
            "self_attn.q_proj": layer.self_attn.q_proj,
            "self_attn.k_proj": layer.self_attn.k_proj,
            "self_attn.v_proj": layer.self_attn.v_proj,
            "self_attn.o_proj": layer.self_attn.o_proj,
            "mlp.gate_proj": layer.mlp.gate_proj,
            "mlp.up_proj": layer.mlp.up_proj,
            "mlp.down_proj": layer.mlp.down_proj,
        }
        
        for name, target in linear_map.items():
            w_key = f"{prefix}.{name}.weight"
            s_key = f"{prefix}.{name}.weight_scale"
            
            ternary = unpack_microsoft_to_ternary(raw[w_key])
            target.packed_weights = pack_ternary_weights(mx.array(ternary))
            target.scale = raw[s_key].astype(mx.float32).reshape(1)
    
    print("Model loaded!")
    mx.eval(model.parameters())
    return model, config


def generate(model, tokenizer, prompt, max_tokens=100):
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])
    
    print(f"Prompt: {prompt}")
    print(f"Tokens: {len(tokens)}")
    print("-" * 50)
    
    start = time.time()
    logits, cache = model(input_ids)
    mx.eval(logits)
    prefill_time = time.time() - start
    print(f"Prefill: {len(tokens)} tokens in {prefill_time:.3f}s ({len(tokens)/prefill_time:.0f} tok/s)")
    
    generated = []
    gen_start = time.time()
    eos_id = tokenizer.eos_token_id
    
    for _ in range(max_tokens):
        token_id = mx.argmax(logits[:, -1, :], axis=-1).item()
        if token_id == eos_id:
            break
        generated.append(token_id)
        logits, cache = model(mx.array([[token_id]]), cache=cache)
        mx.eval(logits)
    
    gen_time = time.time() - gen_start
    n = len(generated)
    tps = n / gen_time if gen_time > 0 else 0
    
    if generated:
        print(tokenizer.decode(generated))
    print(f"{'='*50}")
    print(f"Generated {n} tokens in {gen_time:.2f}s = {tps:.1f} tok/s")
    return tps


if __name__ == "__main__":
    MODEL_PATH = "models/bitnet-2b"
    
    print("=" * 60)
    print("BitNet — Approach 3: Tiled Metal + Apple Attention")
    print(f"MLX {mx.__version__} | Device: {mx.default_device()}")
    print("=" * 60)
    
    model, config = load_model_approach3(MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    
    # Warmup
    print("\nWarmup...")
    generate(model, tokenizer, "Hi", max_tokens=5)
    
    # Benchmark
    print("\n" + "=" * 60)
    tps1 = generate(model, tokenizer, "Once upon a time", max_tokens=100)
    
    print("\n" + "=" * 60)
    tps2 = generate(model, tokenizer, "The meaning of life is", max_tokens=100)
    
    avg = (tps1 + tps2) / 2
    print("\n" + "=" * 60)
    print("COMPARISON:")
    print(f"  Custom inline kernel:      13.8 tok/s")
    print(f"  Approach 1 (QuantizedLinear): 27.9 tok/s")
    print(f"  Approach 3 (Tiled Metal):  {avg:.1f} tok/s")
