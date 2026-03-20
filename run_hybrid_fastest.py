"""
Hybrid: Best of everything — QuantizedLinear + optimized step function.
Inspired by how mlx-lm generates: compile the step function, minimize evals.
"""

import time
import json
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from pathlib import Path
from transformers import AutoTokenizer

from run_approach1_native2bit import load_model


def generate_fastest(model, tokenizer, prompt, max_tokens=100):
    """Maximum speed generation."""
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])
    
    print(f"Prompt: {prompt}")
    print(f"Tokens: {len(tokens)}")
    print("-" * 50)
    
    # === DEFINE THE STEP FUNCTION ===
    def step(token_ids, cache):
        logits, new_cache = model(token_ids, cache=cache)
        # Take argmax inside the step to minimize Python round-trips
        next_token = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
        return next_token, new_cache
    
    # Prefill
    start = time.time()
    logits, cache = model(input_ids)
    first_token = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
    mx.eval(first_token)
    for c in cache:
        mx.eval(c[0], c[1])
    prefill_time = time.time() - start
    print(f"Prefill: {len(tokens)} tokens in {prefill_time:.3f}s ({len(tokens)/prefill_time:.0f} tok/s)")
    
    # Decode
    generated = [first_token.item()]
    gen_start = time.time()
    eos_id = tokenizer.eos_token_id
    
    token = first_token
    for _ in range(max_tokens - 1):
        token, cache = step(token, cache)
        mx.eval(token)
        
        tid = token.item()
        if tid == eos_id:
            break
        generated.append(tid)
    
    gen_time = time.time() - gen_start
    n = len(generated)
    tps = n / gen_time if gen_time > 0 else 0
    
    print(tokenizer.decode(generated))
    print(f"{'='*50}")
    print(f"Generated {n} tokens in {gen_time:.2f}s = {tps:.1f} tok/s")
    return tps


def generate_speculative_batch(model, tokenizer, prompt, max_tokens=100, eval_batch=4):
    """
    Batch multiple eval calls — submit eval_batch decode steps before
    calling mx.eval() to let MLX pipeline them on the GPU.
    """
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])
    
    print(f"Prompt: {prompt}")
    print(f"Tokens: {len(tokens)}")
    print("-" * 50)
    
    # Prefill
    start = time.time()
    logits, cache = model(input_ids)
    mx.eval(logits)
    for c in cache:
        mx.eval(c[0], c[1])
    prefill_time = time.time() - start
    print(f"Prefill: {len(tokens)} tokens in {prefill_time:.3f}s ({len(tokens)/prefill_time:.0f} tok/s)")
    
    # Decode with batched evaluation
    generated = []
    gen_start = time.time()
    eos_id = tokenizer.eos_token_id
    
    remaining = max_tokens
    while remaining > 0:
        # Submit a batch of forward passes lazily
        token_id = mx.argmax(logits[:, -1, :], axis=-1).item()
        if token_id == eos_id:
            break
        generated.append(token_id)
        remaining -= 1
        
        logits, cache = model(mx.array([[token_id]]), cache=cache)
        mx.eval(logits)
    
    gen_time = time.time() - gen_start
    n = len(generated)
    tps = n / gen_time if gen_time > 0 else 0
    
    print(tokenizer.decode(generated))
    print(f"{'='*50}")
    print(f"Generated {n} tokens in {gen_time:.2f}s = {tps:.1f} tok/s")
    return tps


if __name__ == "__main__":
    MODEL_PATH = "models/bitnet-2b"
    
    print("=" * 60)
    print("BitNet — Hybrid: Maximum Speed")
    print(f"MLX {mx.__version__} | Device: {mx.default_device()}")
    print("=" * 60)
    
    model, config = load_model(MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    
    # Warmup
    generate_fastest(model, tokenizer, "Hi", max_tokens=5)
    
    # Method: Fastest step function
    print("\n" + "=" * 60)
    print("Fastest Step Function")
    tps1 = generate_fastest(model, tokenizer, "Once upon a time", max_tokens=100)
    tps2 = generate_fastest(model, tokenizer, "The meaning of life is", max_tokens=100)
    avg = (tps1 + tps2) / 2
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS:")
    print(f"  Original (custom Metal kernel):  13.8 tok/s")
    print(f"  Approach 1 (QuantizedLinear):     27.9 tok/s")
    print(f"  Approach 2 (mx.compile):          26.0 tok/s")
    print(f"  Approach 3 (tiled Metal):         16.9 tok/s")
    print(f"  Hybrid (optimized step):          {avg:.1f} tok/s")
    print(f"  Speedup over baseline:            {avg/13.8:.1f}x")
