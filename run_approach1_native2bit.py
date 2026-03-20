"""
Approach 1: BitNet-2B using MLX's native QuantizedLinear (2-bit).
Uses Apple's hyper-optimized quantized matmul kernels.
"""

import time
import math
import json
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from pathlib import Path
from transformers import AutoTokenizer


# ---------------------------------------------------------------------------
# Weight conversion: Microsoft uint8 → MLX QuantizedLinear 2-bit
# ---------------------------------------------------------------------------

GROUP_SIZE = 64  # MLX quantization group size


def unpack_microsoft_to_ternary(packed_uint8):
    """Unpack Microsoft's uint8 to ternary float {-1, 0, +1}."""
    p = np.array(packed_uint8)
    packed_rows, in_features = p.shape
    real_rows = packed_rows * 4
    result = np.zeros((real_rows, in_features), dtype=np.float32)
    for i in range(4):
        start = i * packed_rows
        end = start + packed_rows
        bits = (p >> (2 * i)) & 0x3
        result[start:end] = bits.astype(np.float32) - 1.0
    return result


def ternary_to_mlx_quantized(ternary_weights, group_size=GROUP_SIZE):
    """
    Convert ternary float weights to MLX's QuantizedLinear format.
    
    Encoding: -1→0, 0→1, +1→2 (i.e., value + 1)
    Scale=1.0, Bias=-1.0 per group → exact reconstruction.
    """
    out_features, in_features = ternary_weights.shape
    
    # Encode ternary to uint: -1→0, 0→1, +1→2
    encoded = (ternary_weights + 1.0).astype(np.uint32)
    
    # Pack 16 values per uint32 (2 bits each)
    packed_cols = in_features * 2 // 32  # 2 bits per value, 32 bits per uint32
    packed = np.zeros((out_features, packed_cols), dtype=np.uint32)
    
    for j in range(in_features):
        packed_idx = j // 16
        bit_pos = (j % 16) * 2
        packed[:, packed_idx] |= encoded[:, j] << bit_pos
    
    # Scales and biases: scale=1.0, bias=-1.0 for every group
    num_groups = in_features // group_size
    scales = np.ones((out_features, num_groups), dtype=np.float16)
    biases = np.full((out_features, num_groups), -1.0, dtype=np.float16)
    
    return mx.array(packed), mx.array(scales), mx.array(biases)


# ---------------------------------------------------------------------------
# Model using nn.QuantizedLinear
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dims, eps=1e-5):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps
    
    def __call__(self, x):
        return mx.fast.rms_norm(x, self.weight, self.eps)


class BitNetAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.num_heads = config["num_attention_heads"]
        self.num_kv_heads = config["num_key_value_heads"]
        self.head_dim = self.hidden_size // self.num_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.rope_theta = config["rope_theta"]
        
        self.q_proj = nn.QuantizedLinear(self.hidden_size, self.num_heads * self.head_dim,
                                         bias=False, group_size=GROUP_SIZE, bits=2)
        self.k_proj = nn.QuantizedLinear(self.hidden_size, self.num_kv_heads * self.head_dim,
                                         bias=False, group_size=GROUP_SIZE, bits=2)
        self.v_proj = nn.QuantizedLinear(self.hidden_size, self.num_kv_heads * self.head_dim,
                                         bias=False, group_size=GROUP_SIZE, bits=2)
        self.o_proj = nn.QuantizedLinear(self.num_heads * self.head_dim, self.hidden_size,
                                         bias=False, group_size=GROUP_SIZE, bits=2)
        
        self.attn_sub_norm = RMSNorm(self.hidden_size, eps=config["rms_norm_eps"])
        self.rope = nn.RoPE(self.head_dim, traditional=False, base=self.rope_theta)
    
    def __call__(self, x, mask=None, cache=None):
        B, L, _ = x.shape
        
        q = self.q_proj(x).reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, L, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        offset = cache[0].shape[2] if cache is not None else 0
        q = self.rope(q, offset=offset)
        k = self.rope(k, offset=offset)
        
        if cache is not None:
            k = mx.concatenate([cache[0], k], axis=2)  # (B, heads, seq, dim)
            v = mx.concatenate([cache[1], v], axis=2)
        new_cache = (k, v)
        
        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.scale, mask=mask
        )
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        out = self.attn_sub_norm(out)
        return self.o_proj(out), new_cache


class BitNetMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.QuantizedLinear(config["hidden_size"], config["intermediate_size"],
                                            bias=False, group_size=GROUP_SIZE, bits=2)
        self.up_proj = nn.QuantizedLinear(config["hidden_size"], config["intermediate_size"],
                                          bias=False, group_size=GROUP_SIZE, bits=2)
        self.down_proj = nn.QuantizedLinear(config["intermediate_size"], config["hidden_size"],
                                            bias=False, group_size=GROUP_SIZE, bits=2)
        self.ffn_sub_norm = RMSNorm(config["intermediate_size"], eps=config["rms_norm_eps"])
    
    def __call__(self, x):
        gate = nn.relu(self.gate_proj(x))
        gate = gate * gate  # relu2
        up = self.up_proj(x)
        hidden = gate * up
        hidden = self.ffn_sub_norm(hidden)
        return self.down_proj(hidden)


class BitNetBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_layernorm = RMSNorm(config["hidden_size"], eps=config["rms_norm_eps"])
        self.self_attn = BitNetAttention(config)
        self.post_attention_layernorm = RMSNorm(config["hidden_size"], eps=config["rms_norm_eps"])
        self.mlp = BitNetMLP(config)
    
    def __call__(self, x, mask=None, cache=None):
        r = x
        x = self.input_layernorm(x)
        x, new_cache = self.self_attn(x, mask=mask, cache=cache)
        x = r + x
        r = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        return r + x, new_cache


class BitNet2BQuantized(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config["vocab_size"], config["hidden_size"])
        self.layers = [BitNetBlock(config) for _ in range(config["num_hidden_layers"])]
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


def load_model(model_path):
    """Load Microsoft's BitNet-2B using MLX QuantizedLinear."""
    model_dir = Path(model_path)
    
    with open(model_dir / "config.json") as f:
        config = json.load(f)
    
    print(f"Approach 1: MLX Native 2-bit QuantizedLinear")
    print(f"  Hidden: {config['hidden_size']}, Layers: {config['num_hidden_layers']}")
    
    model = BitNet2BQuantized(config)
    
    raw = mx.load(str(model_dir / "model.safetensors"))
    
    # Embedding + final norm
    model.embed_tokens.weight = raw["model.embed_tokens.weight"].astype(mx.float16)
    model.norm.weight = raw["model.norm.weight"].astype(mx.float32)
    
    head_dim = config["hidden_size"] // config["num_attention_heads"]
    
    for i in range(config["num_hidden_layers"]):
        if i % 10 == 0:
            print(f"  Loading layer {i}/{config['num_hidden_layers']}...")
        
        layer = model.layers[i]
        prefix = f"model.layers.{i}"
        
        # Norms
        layer.input_layernorm.weight = raw[f"{prefix}.input_layernorm.weight"].astype(mx.float32)
        layer.post_attention_layernorm.weight = raw[f"{prefix}.post_attention_layernorm.weight"].astype(mx.float32)
        layer.self_attn.attn_sub_norm.weight = raw[f"{prefix}.self_attn.attn_sub_norm.weight"].astype(mx.float32)
        layer.mlp.ffn_sub_norm.weight = raw[f"{prefix}.mlp.ffn_sub_norm.weight"].astype(mx.float32)
        
        # Linear layers
        linear_map = {
            "self_attn.q_proj": layer.self_attn.q_proj,
            "self_attn.k_proj": layer.self_attn.k_proj,
            "self_attn.v_proj": layer.self_attn.v_proj,
            "self_attn.o_proj": layer.self_attn.o_proj,
            "mlp.gate_proj": layer.mlp.gate_proj,
            "mlp.up_proj": layer.mlp.up_proj,
            "mlp.down_proj": layer.mlp.down_proj,
        }
        
        for name, ql_layer in linear_map.items():
            w_key = f"{prefix}.{name}.weight"
            s_key = f"{prefix}.{name}.weight_scale"
            
            ternary = unpack_microsoft_to_ternary(raw[w_key])
            packed, scales, biases = ternary_to_mlx_quantized(ternary, group_size=GROUP_SIZE)
            
            # Apply Microsoft's weight scale to our scales
            ms_scale = raw[s_key].astype(mx.float32).item()
            scales = scales * ms_scale
            biases = biases * ms_scale
            
            ql_layer.weight = packed
            ql_layer.scales = scales
            ql_layer.biases = biases
    
    print("Model loaded!")
    mx.eval(model.parameters())
    return model, config


def generate(model, tokenizer, prompt, max_tokens=100, temperature=0.0):
    """Generate text with KV cache."""
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])
    
    print(f"Prompt: {prompt}")
    print(f"Tokens: {len(tokens)}")
    print("-" * 50)
    
    # Prefill
    start = time.time()
    logits, cache = model(input_ids)
    mx.eval(logits)
    prefill_time = time.time() - start
    tps_prefill = len(tokens) / prefill_time
    print(f"Prefill: {len(tokens)} tokens in {prefill_time:.2f}s ({tps_prefill:.0f} tok/s)")
    
    # Decode
    generated = []
    gen_start = time.time()
    
    for _ in range(max_tokens):
        next_token = mx.argmax(logits[:, -1, :], axis=-1) if temperature == 0 else \
            mx.random.categorical(logits[:, -1, :] / temperature)
        
        token_id = next_token.item()
        if token_id == tokenizer.eos_token_id:
            break
        
        generated.append(token_id)
        print(tokenizer.decode([token_id]), end="", flush=True)
        
        logits, cache = model(next_token.reshape(1, 1), cache=cache)
        mx.eval(logits)
    
    gen_time = time.time() - gen_start
    n = len(generated)
    tps = n / gen_time if gen_time > 0 else 0
    
    print(f"\n{'='*50}")
    print(f"Generated {n} tokens in {gen_time:.2f}s = {tps:.1f} tok/s")
    return tps


if __name__ == "__main__":
    MODEL_PATH = "models/bitnet-2b"
    
    print("=" * 60)
    print("MLX BitNet Engine — Approach 1: Native 2-bit QuantizedLinear")
    print(f"MLX {mx.__version__} | Device: {mx.default_device()}")
    print("=" * 60)
    
    model, config = load_model(MODEL_PATH)
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    
    # Warmup
    print("\nWarmup run...")
    generate(model, tokenizer, "Hello", max_tokens=5, temperature=0.0)
    
    # Real benchmarks
    print("\n" + "=" * 60)
    tps1 = generate(model, tokenizer, "Once upon a time", max_tokens=100, temperature=0.0)
    
    print("\n" + "=" * 60)
    tps2 = generate(model, tokenizer, "The meaning of life is", max_tokens=100, temperature=0.0)
    
    print("\n" + "=" * 60)
    print(f"AVERAGE: {(tps1 + tps2) / 2:.1f} tok/s")
    print(f"Previous (custom kernel): 13.8 tok/s")
