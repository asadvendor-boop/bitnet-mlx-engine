"""
Transformer model for BitNet inference.
Supports Qwen / LLaMA style decoder-only architectures.
"""

import math
import mlx.core as mx
import mlx.nn as nn
from dataclasses import dataclass
from typing import Optional, Tuple

from .layers import BitLinear, RMSNorm


@dataclass
class BitNetConfig:
    """Model configuration."""
    vocab_size: int = 32000
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 32  # GQA support
    head_dim: int = 128
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    tie_word_embeddings: bool = False
    # BitNet specific
    use_bitlinear: bool = True
    
    @property
    def num_key_value_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads


class RotaryEmbedding:
    """Rotary Position Embedding (RoPE)."""
    
    def __init__(self, head_dim: int, max_seq_len: int = 32768, theta: float = 10000.0):
        self.head_dim = head_dim
        freqs = 1.0 / (theta ** (mx.arange(0, head_dim, 2, dtype=mx.float32) / head_dim))
        self.freqs = freqs
    
    def __call__(self, x: mx.array, offset: int = 0) -> mx.array:
        seq_len = x.shape[1]
        positions = mx.arange(offset, offset + seq_len, dtype=mx.float32)
        freqs = mx.outer(positions, self.freqs)
        cos = mx.cos(freqs)
        sin = mx.sin(freqs)
        return self._apply_rotary(x, cos, sin)
    
    def _apply_rotary(self, x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        # x: (batch, seq, heads, head_dim)
        head_dim = x.shape[-1]
        x1 = x[..., :head_dim // 2]
        x2 = x[..., head_dim // 2:]
        
        # Reshape cos/sin for broadcasting: (1, seq, 1, head_dim//2)
        cos = cos[None, :, None, :]
        sin = sin[None, :, None, :]
        
        out1 = x1 * cos - x2 * sin
        out2 = x2 * cos + x1 * sin
        return mx.concatenate([out1, out2], axis=-1)


class BitNetAttention(nn.Module):
    """Multi-head attention with BitLinear projections and KV-cache."""
    
    def __init__(self, config: BitNetConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = config.num_key_value_groups
        
        LinearLayer = BitLinear if config.use_bitlinear else nn.Linear
        
        self.q_proj = LinearLayer(config.hidden_size, self.num_heads * self.head_dim)
        self.k_proj = LinearLayer(config.hidden_size, self.num_kv_heads * self.head_dim)
        self.v_proj = LinearLayer(config.hidden_size, self.num_kv_heads * self.head_dim)
        self.o_proj = LinearLayer(self.num_heads * self.head_dim, config.hidden_size)
        
        self.rope = RotaryEmbedding(self.head_dim, theta=config.rope_theta)
        self.scale = 1.0 / math.sqrt(self.head_dim)
    
    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> Tuple[mx.array, Optional[Tuple[mx.array, mx.array]]]:
        B, L, _ = x.shape
        
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape to (B, L, num_heads, head_dim)
        q = q.reshape(B, L, self.num_heads, self.head_dim)
        k = k.reshape(B, L, self.num_kv_heads, self.head_dim)
        v = v.reshape(B, L, self.num_kv_heads, self.head_dim)
        
        # Apply RoPE
        offset = 0
        if cache is not None:
            offset = cache[0].shape[1]
        
        q = self.rope(q, offset=offset)
        k = self.rope(k, offset=offset)
        
        # Update KV cache
        if cache is not None:
            k = mx.concatenate([cache[0], k], axis=1)
            v = mx.concatenate([cache[1], v], axis=1)
        new_cache = (k, v)
        
        # GQA: repeat KV heads
        if self.num_kv_groups > 1:
            k = mx.repeat(k, self.num_kv_groups, axis=2)
            v = mx.repeat(v, self.num_kv_groups, axis=2)
        
        # Transpose for attention: (B, num_heads, L, head_dim)
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        
        # Compute attention
        attn = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        
        if mask is not None:
            attn = attn + mask
        
        attn = mx.softmax(attn, axis=-1)
        out = attn @ v
        
        # Reshape back: (B, L, hidden_size)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        out = self.o_proj(out)
        
        return out, new_cache


class BitNetMLP(nn.Module):
    """Feed-forward network with BitLinear (SwiGLU variant)."""
    
    def __init__(self, config: BitNetConfig):
        super().__init__()
        LinearLayer = BitLinear if config.use_bitlinear else nn.Linear
        
        self.gate_proj = LinearLayer(config.hidden_size, config.intermediate_size)
        self.up_proj = LinearLayer(config.hidden_size, config.intermediate_size)
        self.down_proj = LinearLayer(config.intermediate_size, config.hidden_size)
    
    def __call__(self, x: mx.array) -> mx.array:
        gate = nn.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class BitNetBlock(nn.Module):
    """Single transformer block."""
    
    def __init__(self, config: BitNetConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = BitNetAttention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = BitNetMLP(config)
    
    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> Tuple[mx.array, Optional[Tuple[mx.array, mx.array]]]:
        # Self-attention with residual
        residual = x
        x = self.input_layernorm(x)
        x, new_cache = self.self_attn(x, mask=mask, cache=cache)
        x = residual + x
        
        # MLP with residual
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x
        
        return x, new_cache


class BitNetModel(nn.Module):
    """Full BitNet transformer model."""
    
    def __init__(self, config: BitNetConfig):
        super().__init__()
        self.config = config
        
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [BitNetBlock(config) for _ in range(config.num_hidden_layers)]
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        else:
            self.lm_head = None
    
    def __call__(
        self,
        input_ids: mx.array,
        cache: Optional[list] = None,
    ) -> Tuple[mx.array, list]:
        x = self.embed_tokens(input_ids)
        
        # Create causal mask
        B, L = input_ids.shape
        if L > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(L)
            mask = mask.astype(x.dtype)
        else:
            mask = None
        
        # If we have cache, adjust mask for the full sequence length
        if cache is not None and cache[0] is not None and L > 1:
            cache_len = cache[0][0].shape[1]
            total_len = cache_len + L
            mask = nn.MultiHeadAttention.create_additive_causal_mask(total_len)
            mask = mask[-L:, :]
            mask = mask.astype(x.dtype)
        
        new_cache = []
        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache is not None else None
            x, c = layer(x, mask=mask, cache=layer_cache)
            new_cache.append(c)
        
        x = self.norm(x)
        
        # LM head
        if self.lm_head is not None:
            logits = self.lm_head(x)
        else:
            logits = x @ self.embed_tokens.weight.T
        
        return logits, new_cache
