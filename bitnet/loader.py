"""
Model weight loader for BitNet.
Loads from HuggingFace safetensors and packs weights into ternary format.
"""

import json
import os
from pathlib import Path
from typing import Dict, Optional

import mlx.core as mx
import mlx.nn as nn

from .model import BitNetConfig, BitNetModel
from .layers import BitLinear
from .kernels import pack_ternary_weights


def load_config(model_path: str) -> BitNetConfig:
    """Load model config from config.json."""
    config_path = Path(model_path) / "config.json"
    with open(config_path) as f:
        cfg = json.load(f)
    
    return BitNetConfig(
        vocab_size=cfg.get("vocab_size", 32000),
        hidden_size=cfg.get("hidden_size", 4096),
        intermediate_size=cfg.get("intermediate_size", 11008),
        num_hidden_layers=cfg.get("num_hidden_layers", 32),
        num_attention_heads=cfg.get("num_attention_heads", 32),
        num_key_value_heads=cfg.get("num_key_value_heads", cfg.get("num_attention_heads", 32)),
        head_dim=cfg.get("head_dim", cfg.get("hidden_size", 4096) // cfg.get("num_attention_heads", 32)),
        max_position_embeddings=cfg.get("max_position_embeddings", 32768),
        rms_norm_eps=cfg.get("rms_norm_eps", 1e-6),
        rope_theta=cfg.get("rope_theta", 10000.0),
        tie_word_embeddings=cfg.get("tie_word_embeddings", False),
    )


def load_safetensors_weights(model_path: str) -> Dict[str, mx.array]:
    """Load all weights from safetensors files in a directory."""
    from safetensors import safe_open
    
    weights = {}
    model_dir = Path(model_path)
    
    # Find all safetensors files
    shard_files = sorted(model_dir.glob("*.safetensors"))
    if not shard_files:
        raise FileNotFoundError(f"No .safetensors files found in {model_path}")
    
    for shard_file in shard_files:
        with safe_open(str(shard_file), framework="numpy") as f:
            for key in f.keys():
                weights[key] = mx.array(f.get_tensor(key))
    
    return weights


def _is_ternary(w: mx.array, threshold: float = 0.01) -> bool:
    """Check if weights are approximately ternary {-1, 0, +1}."""
    unique = mx.unique(mx.round(w))
    mx.eval(unique)
    # Allow for float noise
    unique_vals = set(round(v.item(), 1) for v in unique)
    return unique_vals.issubset({-1.0, 0.0, 1.0})


def _quantize_to_ternary(w: mx.array) -> tuple:
    """
    Quantize float weights to ternary {-1, 0, +1}.
    Returns (packed_weights, scale).
    
    Method: AbsMean quantization (from BitNet paper)
    scale = mean(|w|)
    w_ternary = round(clip(w / scale, -1, 1))
    """
    w_float = w.astype(mx.float32)
    scale = mx.mean(mx.abs(w_float))
    mx.eval(scale)
    
    w_normalized = w_float / (scale + 1e-8)
    w_ternary = mx.clip(mx.round(w_normalized), -1, 1)
    
    packed = pack_ternary_weights(w_ternary)
    return packed, scale.reshape(1), w.shape[1]


# Weight name mapping from HuggingFace to our model
WEIGHT_MAP = {
    "model.embed_tokens.weight": "embed_tokens.weight",
    "model.norm.weight": "norm.weight",
    "lm_head.weight": "lm_head.weight",
}


def _get_layer_weight_map(layer_idx: int) -> dict:
    """Generate weight name mapping for a transformer layer."""
    prefix_hf = f"model.layers.{layer_idx}"
    prefix_ours = f"layers.{layer_idx}"
    
    return {
        f"{prefix_hf}.input_layernorm.weight": f"{prefix_ours}.input_layernorm.weight",
        f"{prefix_hf}.post_attention_layernorm.weight": f"{prefix_ours}.post_attention_layernorm.weight",
        f"{prefix_hf}.self_attn.q_proj.weight": f"{prefix_ours}.self_attn.q_proj",
        f"{prefix_hf}.self_attn.k_proj.weight": f"{prefix_ours}.self_attn.k_proj",
        f"{prefix_hf}.self_attn.v_proj.weight": f"{prefix_ours}.self_attn.v_proj",
        f"{prefix_hf}.self_attn.o_proj.weight": f"{prefix_ours}.self_attn.o_proj",
        f"{prefix_hf}.mlp.gate_proj.weight": f"{prefix_ours}.mlp.gate_proj",
        f"{prefix_hf}.mlp.up_proj.weight": f"{prefix_ours}.mlp.up_proj",
        f"{prefix_hf}.mlp.down_proj.weight": f"{prefix_ours}.mlp.down_proj",
    }


def load_model(
    model_path: str,
    quantize: bool = True,
    dtype: mx.Dtype = mx.float16,
    verbose: bool = True,
) -> BitNetModel:
    """
    Load a BitNet model from a HuggingFace directory.
    
    If the weights are already ternary (-1, 0, +1), they are packed directly.
    If the weights are float, they are quantized to ternary using AbsMean method.
    
    Args:
        model_path: Local path to HuggingFace model directory
        quantize: Whether to quantize float weights to ternary
        dtype: Activation dtype (float16 recommended for speed)
        verbose: Print loading progress
    
    Returns:
        BitNetModel with packed ternary weights
    """
    if verbose:
        print(f"Loading config from {model_path}...")
    config = load_config(model_path)
    config.use_bitlinear = True
    
    if verbose:
        print(f"Model: {config.num_hidden_layers} layers, "
              f"{config.hidden_size} hidden, "
              f"{config.num_attention_heads} heads, "
              f"vocab={config.vocab_size}")
    
    # Create model
    model = BitNetModel(config)
    
    if verbose:
        print(f"Loading weights from safetensors...")
    raw_weights = load_safetensors_weights(model_path)
    
    if verbose:
        print(f"Loaded {len(raw_weights)} weight tensors")
        total_params = sum(w.size for w in raw_weights.values())
        print(f"Total parameters: {total_params:,}")
    
    # Load embedding and norm weights directly
    if "model.embed_tokens.weight" in raw_weights:
        model.embed_tokens.weight = raw_weights["model.embed_tokens.weight"].astype(dtype)
    if "model.norm.weight" in raw_weights:
        model.norm.weight = raw_weights["model.norm.weight"].astype(mx.float32)
    if "lm_head.weight" in raw_weights:
        if model.lm_head is not None:
            model.lm_head.weight = raw_weights["lm_head.weight"].astype(dtype)
    
    # Load each transformer layer
    for layer_idx in range(config.num_hidden_layers):
        if verbose and layer_idx % 4 == 0:
            print(f"  Loading layer {layer_idx}/{config.num_hidden_layers}...")
        
        wmap = _get_layer_weight_map(layer_idx)
        layer = model.layers[layer_idx]
        
        for hf_name, our_prefix in wmap.items():
            if hf_name not in raw_weights:
                continue
            
            w = raw_weights[hf_name]
            
            # RMSNorm weights
            if "layernorm" in hf_name:
                if "input_layernorm" in hf_name:
                    layer.input_layernorm.weight = w.astype(mx.float32)
                else:
                    layer.post_attention_layernorm.weight = w.astype(mx.float32)
                continue
            
            # Linear layer weights -> BitLinear
            parts = our_prefix.split(".")
            target = layer
            for p in parts[1:-1]:  # Skip "layers.{idx}" prefix
                target = getattr(target, p)
            proj_name = parts[-1]
            bit_layer = getattr(target, proj_name)
            
            if isinstance(bit_layer, BitLinear):
                already_ternary = _is_ternary(w)
                
                if already_ternary:
                    # Weights are already ternary, just pack
                    bit_layer.packed_weights = pack_ternary_weights(w)
                    bit_layer.scale = mx.array([1.0], dtype=mx.float32)
                elif quantize:
                    # Quantize to ternary
                    packed, scale, in_feat = _quantize_to_ternary(w)
                    bit_layer.packed_weights = packed
                    bit_layer.scale = scale
                else:
                    raise ValueError(
                        f"Weight {hf_name} is not ternary and quantize=False"
                    )
            
            # Free the raw weight
            del raw_weights[hf_name]
    
    if verbose:
        # Estimate model size
        total_bytes = 0
        for layer in model.layers:
            for name, param in layer.parameters().items():
                if hasattr(param, 'nbytes'):
                    total_bytes += param.nbytes
        print(f"Packed model size: ~{total_bytes / 1e9:.2f} GB")
        print("Model loaded successfully!")
    
    return model
