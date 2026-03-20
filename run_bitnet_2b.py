"""
End-to-end test: Load Microsoft's BitNet-b1.58-2B-4T and generate text.
Handles Microsoft's custom packed uint8 weight format.
"""

import time
import math
import json
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from pathlib import Path
from transformers import AutoTokenizer

from bitnet.kernels import pack_ternary_weights, ternary_matmul
from bitnet.layers import RMSNorm


# ---------------------------------------------------------------------------
# Microsoft BitNet architecture (differs from standard Qwen/LLaMA)
# - Uses relu2 (squared ReLU) instead of SwiGLU
# - Has sub-norms after attention and FFN projections
# - Weights stored as packed uint8 (4 ternary values per byte)
# ---------------------------------------------------------------------------


def unpack_microsoft_weights(packed_uint8: mx.array) -> mx.array:
    """
    Unpack Microsoft's bitnet uint8 packing to ternary float.
    
    From HF transformers: values are encoded as (ternary + 1), so:
      0b00 (0) → -1
      0b01 (1) →  0  
      0b10 (2) → +1
    
    4 values packed per byte along dim 0.
    Layout is contiguous blocks:
      rows [0, packed_rows) → bits 0-1 of each byte
      rows [packed_rows, 2*packed_rows) → bits 2-3
      rows [2*packed_rows, 3*packed_rows) → bits 4-5
      rows [3*packed_rows, 4*packed_rows) → bits 6-7
    """
    p = np.array(packed_uint8)  # (packed_rows, in_features) uint8
    packed_rows, in_features = p.shape
    real_rows = packed_rows * 4
    
    result = np.zeros((real_rows, in_features), dtype=np.float32)
    
    for i in range(4):
        start = i * packed_rows
        end = start + packed_rows
        bits = (p >> (2 * i)) & 0x3
        # Subtract 1: 0→-1, 1→0, 2→+1
        result[start:end] = bits.astype(np.float32) - 1.0
    
    return mx.array(result)


def repack_for_our_kernel(ternary_weights: mx.array) -> mx.array:
    """
    Take ternary float weights (out, in) and pack into our uint32 format.
    Our format: 16 ternary values per uint32, packed along dim 1 (in_features).
    """
    return pack_ternary_weights(ternary_weights)


class BitLinear2B(nn.Module):
    """BitLinear layer adapted for Microsoft's BitNet-2B format."""
    
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        packed_cols = math.ceil(in_features / 16)
        self.packed_weights = mx.zeros((out_features, packed_cols), dtype=mx.uint32)
        self.scale = mx.array([1.0], dtype=mx.float32)
    
    def __call__(self, x):
        # Kernel handles 1D or 2D input. Flatten 3D (B,L,D) -> 2D (B*L, D)
        orig_shape = x.shape
        if x.ndim == 3:
            B, L, D = x.shape
            x = x.reshape(B * L, D)
        out = ternary_matmul(self.packed_weights, x, self.scale)
        if len(orig_shape) == 3:
            out = out.reshape(orig_shape[0], orig_shape[1], -1)
        return out


class RotaryEmbedding:
    def __init__(self, head_dim, theta=500000.0):
        freqs = 1.0 / (theta ** (mx.arange(0, head_dim, 2, dtype=mx.float32) / head_dim))
        self.freqs = freqs
    
    def __call__(self, x, offset=0):
        seq_len = x.shape[1]
        positions = mx.arange(offset, offset + seq_len, dtype=mx.float32)
        freqs = mx.outer(positions, self.freqs)
        cos = mx.cos(freqs)
        sin = mx.sin(freqs)
        head_dim = x.shape[-1]
        x1, x2 = x[..., :head_dim//2], x[..., head_dim//2:]
        cos = cos[None, :, None, :]
        sin = sin[None, :, None, :]
        return mx.concatenate([x1*cos - x2*sin, x2*cos + x1*sin], axis=-1)


class BitNetAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim, rope_theta):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_kv_groups = num_heads // num_kv_heads
        self.scale = 1.0 / math.sqrt(head_dim)
        
        self.q_proj = BitLinear2B(hidden_size, num_heads * head_dim)
        self.k_proj = BitLinear2B(hidden_size, num_kv_heads * head_dim)
        self.v_proj = BitLinear2B(hidden_size, num_kv_heads * head_dim)
        self.o_proj = BitLinear2B(num_heads * head_dim, hidden_size)
        
        # Microsoft's BitNet has an extra sub-norm after attention QKV
        self.attn_sub_norm = RMSNorm(hidden_size)
        self.rope = RotaryEmbedding(head_dim, theta=rope_theta)
    
    def __call__(self, x, mask=None, cache=None):
        B, L, _ = x.shape
        
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        q = q.reshape(B, L, self.num_heads, self.head_dim)
        k = k.reshape(B, L, self.num_kv_heads, self.head_dim)
        v = v.reshape(B, L, self.num_kv_heads, self.head_dim)
        
        offset = cache[0].shape[1] if cache is not None else 0
        q = self.rope(q, offset=offset)
        k = self.rope(k, offset=offset)
        
        if cache is not None:
            k = mx.concatenate([cache[0], k], axis=1)
            v = mx.concatenate([cache[1], v], axis=1)
        new_cache = (k, v)
        
        if self.num_kv_groups > 1:
            k = mx.repeat(k, self.num_kv_groups, axis=2)
            v = mx.repeat(v, self.num_kv_groups, axis=2)
        
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        
        attn = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        if mask is not None:
            attn = attn + mask
        attn = mx.softmax(attn, axis=-1)
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, L, -1)
        
        out = self.attn_sub_norm(out)
        out = self.o_proj(out)
        return out, new_cache


class BitNetMLP(nn.Module):
    """MLP with relu2 (squared ReLU) activation."""
    
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = BitLinear2B(hidden_size, intermediate_size)
        self.up_proj = BitLinear2B(hidden_size, intermediate_size)
        self.down_proj = BitLinear2B(intermediate_size, hidden_size)
        self.ffn_sub_norm = RMSNorm(intermediate_size)
    
    def __call__(self, x):
        # relu2 = relu(x)^2
        gate = nn.relu(self.gate_proj(x))
        gate = gate * gate  # squared
        up = self.up_proj(x)
        hidden = gate * up
        hidden = self.ffn_sub_norm(hidden)
        return self.down_proj(hidden)


class BitNetBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        head_dim = config["hidden_size"] // config["num_attention_heads"]
        self.input_layernorm = RMSNorm(config["hidden_size"], eps=config["rms_norm_eps"])
        self.self_attn = BitNetAttention(
            config["hidden_size"], config["num_attention_heads"],
            config["num_key_value_heads"], head_dim, config["rope_theta"]
        )
        self.post_attention_layernorm = RMSNorm(config["hidden_size"], eps=config["rms_norm_eps"])
        self.mlp = BitNetMLP(config["hidden_size"], config["intermediate_size"])
    
    def __call__(self, x, mask=None, cache=None):
        residual = x
        x = self.input_layernorm(x)
        x, new_cache = self.self_attn(x, mask=mask, cache=cache)
        x = residual + x
        
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x
        return x, new_cache


class BitNet2BModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config["vocab_size"], config["hidden_size"])
        self.layers = [BitNetBlock(config) for _ in range(config["num_hidden_layers"])]
        self.norm = RMSNorm(config["hidden_size"], eps=config["rms_norm_eps"])
    
    def __call__(self, input_ids, cache=None):
        x = self.embed_tokens(input_ids)
        B, L = input_ids.shape
        
        if L > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(L).astype(x.dtype)
            if cache is not None and cache[0] is not None:
                cache_len = cache[0][0].shape[1]
                mask = nn.MultiHeadAttention.create_additive_causal_mask(cache_len + L)
                mask = mask[-L:, :].astype(x.dtype)
        else:
            mask = None
        
        new_cache = []
        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache is not None else None
            x, c = layer(x, mask=mask, cache=layer_cache)
            new_cache.append(c)
        
        x = self.norm(x)
        logits = x @ self.embed_tokens.weight.T  # tied embeddings
        return logits, new_cache


def load_bitnet_2b(model_path):
    """Load Microsoft's BitNet-b1.58-2B-4T model."""
    model_dir = Path(model_path)
    
    with open(model_dir / "config.json") as f:
        config = json.load(f)
    
    print(f"Loading BitNet-b1.58-2B-4T:")
    print(f"  Hidden: {config['hidden_size']}, Layers: {config['num_hidden_layers']}")
    print(f"  Heads: {config['num_attention_heads']}, KV heads: {config['num_key_value_heads']}")
    
    model = BitNet2BModel(config)
    
    # Load weights
    print("Loading safetensors...")
    raw = mx.load(str(model_dir / "model.safetensors"))
    
    # Embedding
    model.embed_tokens.weight = raw["model.embed_tokens.weight"].astype(mx.float16)
    model.norm.weight = raw["model.norm.weight"].astype(mx.float32)
    
    head_dim = config["hidden_size"] // config["num_attention_heads"]
    
    for i in range(config["num_hidden_layers"]):
        if i % 5 == 0:
            print(f"  Loading layer {i}/{config['num_hidden_layers']}...")
        
        layer = model.layers[i]
        prefix = f"model.layers.{i}"
        
        # Layer norms
        layer.input_layernorm.weight = raw[f"{prefix}.input_layernorm.weight"].astype(mx.float32)
        layer.post_attention_layernorm.weight = raw[f"{prefix}.post_attention_layernorm.weight"].astype(mx.float32)
        
        # Sub-norms
        layer.self_attn.attn_sub_norm.weight = raw[f"{prefix}.self_attn.attn_sub_norm.weight"].astype(mx.float32)
        layer.mlp.ffn_sub_norm.weight = raw[f"{prefix}.mlp.ffn_sub_norm.weight"].astype(mx.float32)
        
        # Linear layers — unpack Microsoft's uint8 and repack to our uint32
        linear_map = {
            "self_attn.q_proj": (layer.self_attn.q_proj, config["num_attention_heads"] * head_dim, config["hidden_size"]),
            "self_attn.k_proj": (layer.self_attn.k_proj, config["num_key_value_heads"] * head_dim, config["hidden_size"]),
            "self_attn.v_proj": (layer.self_attn.v_proj, config["num_key_value_heads"] * head_dim, config["hidden_size"]),
            "self_attn.o_proj": (layer.self_attn.o_proj, config["hidden_size"], config["num_attention_heads"] * head_dim),
            "mlp.gate_proj": (layer.mlp.gate_proj, config["intermediate_size"], config["hidden_size"]),
            "mlp.up_proj": (layer.mlp.up_proj, config["intermediate_size"], config["hidden_size"]),
            "mlp.down_proj": (layer.mlp.down_proj, config["hidden_size"], config["intermediate_size"]),
        }
        
        for name, (target_layer, out_feat, in_feat) in linear_map.items():
            w_key = f"{prefix}.{name}.weight"
            s_key = f"{prefix}.{name}.weight_scale"
            
            packed_uint8 = raw[w_key]
            scale_val = raw[s_key].astype(mx.float32)
            
            # Unpack Microsoft's format to ternary, then repack for our kernel
            ternary = unpack_microsoft_weights(packed_uint8)
            target_layer.packed_weights = repack_for_our_kernel(ternary)
            target_layer.scale = scale_val.reshape(1)
    
    print("Model loaded!")
    return model, config


def generate(model, tokenizer, prompt, max_tokens=100, temperature=0.7):
    """Generate text with the loaded model."""
    input_ids = mx.array([tokenizer.encode(prompt)])
    
    print(f"Prompt: {prompt}")
    print(f"Prompt tokens: {input_ids.shape[1]}")
    print("-" * 50)
    
    # Prefill
    start = time.time()
    logits, cache = model(input_ids)
    mx.eval(logits)
    prefill_time = time.time() - start
    print(f"Prefill: {input_ids.shape[1]} tokens in {prefill_time:.2f}s")
    
    # Decode
    generated = []
    gen_start = time.time()
    
    for step in range(max_tokens):
        next_logits = logits[:, -1, :]
        
        if temperature == 0:
            next_token = mx.argmax(next_logits, axis=-1)
        else:
            next_logits = next_logits / temperature
            probs = mx.softmax(next_logits, axis=-1)
            next_token = mx.random.categorical(mx.log(probs + 1e-10))
        
        token_id = next_token.item()
        
        if token_id == tokenizer.eos_token_id:
            break
        
        generated.append(token_id)
        text = tokenizer.decode([token_id])
        print(text, end="", flush=True)
        
        # Next step with KV cache
        token_input = next_token.reshape(1, 1)
        logits, cache = model(token_input, cache=cache)
        mx.eval(logits)
    
    gen_time = time.time() - gen_start
    n_gen = len(generated)
    tps = n_gen / gen_time if gen_time > 0 else 0
    
    print(f"\n{'='*50}")
    print(f"Generated {n_gen} tokens in {gen_time:.2f}s = {tps:.1f} tok/s")
    print(f"Prefill: {prefill_time:.2f}s | Decode: {gen_time:.2f}s")
    
    return tokenizer.decode(generated)


if __name__ == "__main__":
    MODEL_PATH = "models/bitnet-2b"
    
    print("=" * 60)
    print("MLX BitNet Engine — End-to-End Test")
    print(f"MLX {mx.__version__} | Device: {mx.default_device()}")
    print("=" * 60)
    
    # Load model
    model, config = load_bitnet_2b(MODEL_PATH)
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    
    # Generate
    print("\n" + "=" * 60)
    generate(model, tokenizer, "Once upon a time", max_tokens=50, temperature=0.7)
    
    print("\n" + "=" * 60)
    generate(model, tokenizer, "The meaning of life is", max_tokens=50, temperature=0.7)
