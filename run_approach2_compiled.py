"""
Approach 2: BitNet-2B with mx.compile() graph fusion + optimized decode loop.
This is how mlx-lm achieves high throughput — fuses GPU operations.
"""

import time
import math
import json
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from pathlib import Path
from transformers import AutoTokenizer

# Reuse Approach 1's model and loading code
from run_approach1_native2bit import (
    load_model, RMSNorm, BitNetAttention, BitNetMLP, 
    BitNetBlock, BitNet2BQuantized, GROUP_SIZE
)


def generate_compiled(model, tokenizer, prompt, max_tokens=100):
    """Generate with mx.compile() for graph fusion."""
    tokens = tokenizer.encode(prompt)
    
    print(f"Prompt: {prompt}")
    print(f"Tokens: {len(tokens)}")
    print("-" * 50)
    
    # Compile the model forward pass
    @mx.compile
    def compiled_forward(input_ids, *cache_flat):
        # Reconstruct cache from flat list
        if len(cache_flat) > 0:
            cache = [(cache_flat[i*2], cache_flat[i*2+1]) for i in range(len(model.layers))]
        else:
            cache = None
        
        logits, new_cache = model(input_ids, cache=cache)
        
        # Flatten cache for compiled function (can't return nested structures)
        flat_cache = []
        for k, v in new_cache:
            flat_cache.extend([k, v])
        
        return logits, *flat_cache
    
    # Prefill (uncompiled for first run)
    input_ids = mx.array([tokens])
    start = time.time()
    logits, cache = model(input_ids)
    mx.eval(logits)
    prefill_time = time.time() - start
    tps_prefill = len(tokens) / prefill_time
    print(f"Prefill: {len(tokens)} tokens in {prefill_time:.3f}s ({tps_prefill:.0f} tok/s)")
    
    # Flatten cache for compiled function
    cache_flat = []
    for k, v in cache:
        cache_flat.extend([k, v])
    
    # Decode with compiled forward
    generated = []
    gen_start = time.time()
    
    for step in range(max_tokens):
        next_token = mx.argmax(logits[:, -1, :], axis=-1)
        token_id = next_token.item()
        if token_id == tokenizer.eos_token_id:
            break
        
        generated.append(token_id)
        print(tokenizer.decode([token_id]), end="", flush=True)
        
        results = compiled_forward(next_token.reshape(1, 1), *cache_flat)
        logits = results[0]
        cache_flat = list(results[1:])
        mx.eval(logits)
    
    gen_time = time.time() - gen_start
    n = len(generated)
    tps = n / gen_time if gen_time > 0 else 0
    
    print(f"\n{'='*50}")
    print(f"Generated {n} tokens in {gen_time:.2f}s = {tps:.1f} tok/s")
    return tps


def generate_simple_optimized(model, tokenizer, prompt, max_tokens=100):
    """Generate with minimal Python overhead — single eval per step."""
    tokens = tokenizer.encode(prompt)
    
    print(f"Prompt: {prompt}")
    print(f"Tokens: {len(tokens)}")
    print("-" * 50)
    
    input_ids = mx.array([tokens])
    
    # Prefill
    start = time.time()
    logits, cache = model(input_ids)
    mx.eval(logits)
    prefill_time = time.time() - start
    print(f"Prefill: {len(tokens)} tokens in {prefill_time:.3f}s ({len(tokens)/prefill_time:.0f} tok/s)")
    
    # Decode
    generated = []
    gen_start = time.time()
    
    # Minimal Python in the loop
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
    
    # Print all at once to avoid print overhead in loop
    if generated:
        print(tokenizer.decode(generated))
    
    print(f"{'='*50}")
    print(f"Generated {n} tokens in {gen_time:.2f}s = {tps:.1f} tok/s")
    return tps


if __name__ == "__main__":
    MODEL_PATH = "models/bitnet-2b"
    
    print("=" * 60)
    print("BitNet — Approach 2: Graph Compilation + Optimized Decode")
    print(f"MLX {mx.__version__} | Device: {mx.default_device()}")
    print("=" * 60)
    
    model, config = load_model(MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    
    # Method A: Minimal Python overhead (batch print)
    print("\n" + "=" * 60)
    print("Method A: Minimized Python overhead")
    print("=" * 60)
    
    # Warmup
    generate_simple_optimized(model, tokenizer, "Hi", max_tokens=5)
    
    tps_a1 = generate_simple_optimized(model, tokenizer, "Once upon a time", max_tokens=100)
    tps_a2 = generate_simple_optimized(model, tokenizer, "The meaning of life is", max_tokens=100)
    avg_a = (tps_a1 + tps_a2) / 2
    
    # Method B: mx.compile() graph fusion
    print("\n" + "=" * 60)
    print("Method B: mx.compile() graph fusion")
    print("=" * 60)
    
    # Warmup
    generate_compiled(model, tokenizer, "Hi", max_tokens=5)
    
    tps_b1 = generate_compiled(model, tokenizer, "Once upon a time", max_tokens=100)
    tps_b2 = generate_compiled(model, tokenizer, "The meaning of life is", max_tokens=100)
    avg_b = (tps_b1 + tps_b2) / 2
    
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print(f"  Baseline (custom kernel): 13.8 tok/s")
    print(f"  Approach 1 (QuantizedLinear): 27.9 tok/s")
    print(f"  Approach 2A (optimized loop): {avg_a:.1f} tok/s")
    print(f"  Approach 2B (mx.compile):     {avg_b:.1f} tok/s")
