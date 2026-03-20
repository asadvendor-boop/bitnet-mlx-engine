"""
Text generation pipeline with KV-cache, sampling, and streaming.
"""

import time
from typing import Optional, Generator

import mlx.core as mx

from .model import BitNetModel


def top_p_sampling(logits: mx.array, temperature: float, top_p: float) -> mx.array:
    """Nucleus (top-p) sampling."""
    if temperature == 0:
        return mx.argmax(logits, axis=-1)
    
    logits = logits / temperature
    probs = mx.softmax(logits, axis=-1)
    
    # Sort probabilities descending
    sorted_indices = mx.argsort(-probs, axis=-1)
    sorted_probs = mx.take_along_axis(probs, sorted_indices, axis=-1)
    
    # Cumulative sum
    cumulative_probs = mx.cumsum(sorted_probs, axis=-1)
    
    # Create mask for tokens to keep (cumulative prob <= top_p)
    mask = cumulative_probs - sorted_probs <= top_p
    sorted_probs = sorted_probs * mask
    
    # Re-normalize
    sorted_probs = sorted_probs / mx.sum(sorted_probs, axis=-1, keepdims=True)
    
    # Sample from filtered distribution
    token = mx.random.categorical(mx.log(sorted_probs + 1e-10))
    
    # Map back to original indices
    return mx.take_along_axis(sorted_indices, token.reshape(-1, 1), axis=-1).squeeze(-1)


def generate(
    model: BitNetModel,
    tokenizer,
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    verbose: bool = True,
) -> str:
    """
    Generate text from a prompt.
    
    Args:
        model: BitNetModel
        tokenizer: HuggingFace tokenizer
        prompt: Input text
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0 = greedy)
        top_p: Nucleus sampling threshold
        verbose: Print progress info
    
    Returns:
        Generated text string
    """
    # Tokenize
    input_ids = mx.array([tokenizer.encode(prompt)])
    
    if verbose:
        print(f"Prompt tokens: {input_ids.shape[1]}")
        print(f"Generating up to {max_tokens} tokens...")
        print("-" * 50)
        print(prompt, end="", flush=True)
    
    # Prefill: process the full prompt
    start_time = time.time()
    logits, cache = model(input_ids)
    mx.eval(logits)
    prefill_time = time.time() - start_time
    
    if verbose:
        tps = input_ids.shape[1] / prefill_time
        print(f"\n[Prefill: {input_ids.shape[1]} tokens in {prefill_time:.2f}s = {tps:.1f} tok/s]",
              end="", flush=True)
    
    # Get the next token
    next_logits = logits[:, -1, :]
    next_token = top_p_sampling(next_logits, temperature, top_p)
    
    generated_tokens = [next_token.item()]
    
    # Decode loop
    gen_start = time.time()
    eos_id = tokenizer.eos_token_id
    
    for i in range(max_tokens - 1):
        # Forward pass with single token + KV cache
        token_input = next_token.reshape(1, 1)
        logits, cache = model(token_input, cache=cache)
        mx.eval(logits)
        
        next_logits = logits[:, -1, :]
        next_token = top_p_sampling(next_logits, temperature, top_p)
        
        token_id = next_token.item()
        
        if token_id == eos_id:
            break
        
        generated_tokens.append(token_id)
        
        # Stream output
        if verbose:
            text = tokenizer.decode([token_id])
            print(text, end="", flush=True)
    
    gen_time = time.time() - gen_start
    
    # Decode full output
    output_text = tokenizer.decode(generated_tokens)
    
    if verbose:
        n_gen = len(generated_tokens)
        tps = n_gen / gen_time if gen_time > 0 else 0
        print(f"\n\n{'=' * 50}")
        print(f"Generated {n_gen} tokens in {gen_time:.2f}s = {tps:.1f} tok/s")
        print(f"Prefill: {prefill_time:.2f}s | Generation: {gen_time:.2f}s")
    
    return output_text


def generate_stream(
    model: BitNetModel,
    tokenizer,
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> Generator[str, None, None]:
    """
    Streaming generator version — yields token strings as they're generated.
    """
    input_ids = mx.array([tokenizer.encode(prompt)])
    
    logits, cache = model(input_ids)
    mx.eval(logits)
    
    next_logits = logits[:, -1, :]
    next_token = top_p_sampling(next_logits, temperature, top_p)
    
    eos_id = tokenizer.eos_token_id
    
    for i in range(max_tokens):
        token_id = next_token.item()
        
        if token_id == eos_id:
            break
        
        yield tokenizer.decode([token_id])
        
        token_input = next_token.reshape(1, 1)
        logits, cache = model(token_input, cache=cache)
        mx.eval(logits)
        
        next_logits = logits[:, -1, :]
        next_token = top_p_sampling(next_logits, temperature, top_p)
