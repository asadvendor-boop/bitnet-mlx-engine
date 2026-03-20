"""
Generate training data for the draft model.
Runs BitNet-2B on text, collects full-sequence logits for distillation.
"""
import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import generate_step
import numpy as np
import time
import os
import json

MODEL_PATH = "models/bitnet-2b"
OUTPUT_DIR = "data/teacher"
CONTEXT_LEN = 128        # Tokens per sequence
NUM_SEQUENCES = 500      # Number of sequences
TOP_K = 32               # Save top-32 logits per position

os.makedirs(OUTPUT_DIR, exist_ok=True)

PROMPTS = [
    "The history of artificial intelligence began in",
    "In mathematics, a prime number is defined as",
    "Machine learning is a subset of artificial intelligence",
    "The solar system consists of eight planets",
    "Python is a high-level programming language",
    "The theory of relativity proposed by Einstein",
    "In economics, supply and demand refers to",
    "Shakespeare wrote many famous plays including",
    "Climate change is primarily caused by",
    "The internet was originally developed as",
    "Quantum computing uses quantum bits or qubits",
    "Deep learning neural networks consist of",
    "The periodic table organizes chemical elements",
    "The Renaissance was a period of cultural",
    "Photosynthesis is the process by which plants",
    "DNA stands for deoxyribonucleic acid and",
    "Natural language processing enables computers",
    "Blockchain technology provides a decentralized",
    "Algebra is a branch of mathematics that",
    "Transformers architecture revolutionized natural",
    "The United States Declaration of Independence",
    "Water is composed of two hydrogen atoms",
    "The speed of light in a vacuum is",
    "Artificial neural networks are inspired by",
    "The global population has grown significantly",
    "Operating systems manage computer hardware and",
    "The human brain contains approximately one hundred",
    "Gravity is a fundamental force that attracts",
    "The printing press was invented by Johannes",
    "Renewable energy sources include solar wind",
    "The Great Barrier Reef is the largest",
    "Calculus was independently developed by Newton",
    "The immune system protects the body from",
    "Social media platforms have transformed how",
    "The Milky Way galaxy contains billions of",
    "Object-oriented programming organizes software design",
    "The French Revolution fundamentally changed",
    "Semiconductors are materials that have electrical",
    "The Amazon River is the largest river by",
    "Cryptography is the practice of securing",
]

print("Loading BitNet-2B teacher model...")
model, tokenizer = load(MODEL_PATH)
mx.set_memory_limit(8 * 1024**3)
mx.set_cache_limit(4 * 1024**3)
print(f"Vocab: {tokenizer.vocab_size}")

all_sequences = []    # List of (seq_len,) int arrays
all_top_logits = []   # List of (seq_len-1, TOP_K) float16 arrays
all_top_indices = []  # List of (seq_len-1, TOP_K) int32 arrays
total_tokens = 0
t0 = time.time()

for seq_idx in range(NUM_SEQUENCES):
    prompt = PROMPTS[seq_idx % len(PROMPTS)]
    tokens = tokenizer.encode(prompt)
    
    # Generate continuation using generate_step
    generated = list(tokens)
    prompt_arr = mx.array(tokens)
    
    for tok, _ in generate_step(prompt_arr, model, max_tokens=CONTEXT_LEN - len(tokens)):
        if isinstance(tok, mx.array):
            mx.eval(tok)
            tok_id = tok.item()
        else:
            tok_id = int(tok)
        generated.append(tok_id)
        if tok_id == tokenizer.eos_token_id:
            break
    
    if len(generated) < 10:
        continue  # Skip very short sequences
    
    # Forward pass on entire sequence to get all logits at once
    seq = generated[:CONTEXT_LEN]
    x = mx.array([seq])
    logits = model(x)  # (1, seq_len, vocab)
    mx.eval(logits)
    
    seq_len = len(seq)
    
    # Extract top-K logits for each position (predict next token)
    seq_top_logits = np.zeros((seq_len - 1, TOP_K), dtype=np.float16)
    seq_top_indices = np.zeros((seq_len - 1, TOP_K), dtype=np.int32)
    
    for pos in range(seq_len - 1):
        pos_logits = logits[0, pos]
        # Cast to float32 first (bfloat16 → numpy doesn't work directly)
        pos_logits_f32 = pos_logits.astype(mx.float32)
        top_idx = mx.argpartition(pos_logits_f32, kth=pos_logits_f32.shape[0] - TOP_K)[-TOP_K:]
        top_vals = pos_logits_f32[top_idx]
        mx.eval(top_idx, top_vals)
        seq_top_logits[pos] = np.array(top_vals).astype(np.float16)
        seq_top_indices[pos] = np.array(top_idx).astype(np.int32)
    
    all_sequences.append(np.array(seq, dtype=np.int32))
    all_top_logits.append(seq_top_logits)
    all_top_indices.append(seq_top_indices)
    total_tokens += seq_len - 1
    
    if (seq_idx + 1) % 10 == 0:
        elapsed = time.time() - t0
        print(f"  Seq {seq_idx+1}/{NUM_SEQUENCES}: {total_tokens:,} tokens "
              f"({elapsed:.0f}s, ETA: {elapsed * (NUM_SEQUENCES-seq_idx-1) / (seq_idx+1) / 60:.1f}m)")

# Flatten and save
print(f"\nFlattening {total_tokens:,} token positions...")

flat_input = np.concatenate([s[:-1] for s in all_sequences])
flat_target = np.concatenate([s[1:] for s in all_sequences])
flat_logits = np.concatenate(all_top_logits, axis=0)
flat_indices = np.concatenate(all_top_indices, axis=0)

print(f"Saving to {OUTPUT_DIR}/...")
np.savez_compressed(
    f"{OUTPUT_DIR}/teacher_data.npz",
    input_ids=flat_input,
    target_ids=flat_target,
    top_logits=flat_logits,
    top_indices=flat_indices,
)

meta = {
    "total_tokens": int(total_tokens),
    "num_sequences": len(all_sequences),
    "context_len": CONTEXT_LEN,
    "top_k": TOP_K,
    "vocab_size": tokenizer.vocab_size,
}
with open(f"{OUTPUT_DIR}/metadata.json", "w") as f:
    json.dump(meta, f, indent=2)

elapsed = time.time() - t0
sz = os.path.getsize(f"{OUTPUT_DIR}/teacher_data.npz") / 1024**2
print(f"Done! {total_tokens:,} tokens in {elapsed:.0f}s")
print(f"  File: {sz:.1f} MB")
