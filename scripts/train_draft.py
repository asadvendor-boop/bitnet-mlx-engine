"""
Train a tiny draft model via knowledge distillation from BitNet-2B teacher.

Architecture: 2 layers, 512 hidden, 8 heads, shared embedding
Loss: KL divergence on top-K teacher logits
"""
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import time
import os
import json
import math

# ========== Draft Model Architecture ==========
class DraftRMSNorm(nn.Module):
    def __init__(self, dims, eps=1e-5):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps
    
    def __call__(self, x):
        return mx.fast.rms_norm(x, self.weight, self.eps)

class DraftAttention(nn.Module):
    def __init__(self, hidden, num_heads, head_dim):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.q_proj = nn.Linear(hidden, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, num_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, num_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden, bias=False)
        self.rope = nn.RoPE(head_dim, traditional=False)
    
    def __call__(self, x, mask=None, cache=None):
        B, L, _ = x.shape
        q = self.q_proj(x).reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        offset = 0
        if cache is not None:
            offset = cache.offset
            
        q = self.rope(q, offset=offset)
        k = self.rope(k, offset=offset)
        
        if cache is not None:
            k, v = cache.update_and_fetch(k, v)
        
        attn = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.scale,
            mask="causal" if L > 1 else None
        )
        attn = attn.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(attn)

class DraftFFN(nn.Module):
    def __init__(self, hidden, intermediate):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)
    
    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))

class DraftLayer(nn.Module):
    def __init__(self, hidden, num_heads, head_dim, intermediate):
        super().__init__()
        self.input_norm = DraftRMSNorm(hidden)
        self.self_attn = DraftAttention(hidden, num_heads, head_dim)
        self.post_attn_norm = DraftRMSNorm(hidden)
        self.mlp = DraftFFN(hidden, intermediate)
    
    def __call__(self, x, mask=None, cache=None):
        x = x + self.self_attn(self.input_norm(x), mask, cache)
        x = x + self.mlp(self.post_attn_norm(x))
        return x

class DraftModel(nn.Module):
    """Tiny draft model for speculative decoding."""
    
    def __init__(self, vocab_size=152064, hidden=512, num_layers=2,
                 num_heads=8, head_dim=64, intermediate=1408):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden)
        self.layers = [DraftLayer(hidden, num_heads, head_dim, intermediate)
                       for _ in range(num_layers)]
        self.norm = DraftRMSNorm(hidden)
        self.lm_head = nn.Linear(hidden, vocab_size, bias=False)
        
        # Config for reference
        self.hidden = hidden
        self.num_layers = num_layers
        self.vocab_size = vocab_size
    
    def __call__(self, input_ids, cache=None):
        x = self.embed_tokens(input_ids)
        
        caches = [None] * len(self.layers) if cache is None else cache
        for i, layer in enumerate(self.layers):
            x = layer(x, cache=caches[i])
        
        x = self.norm(x)
        return self.lm_head(x)
    
    def make_cache(self):
        return [nn.KVCache() for _ in self.layers]

# ========== Training ==========
def train():
    # Config
    HIDDEN = 512
    NUM_LAYERS = 2
    NUM_HEADS = 8
    HEAD_DIM = 64
    INTERMEDIATE = 1408  # ~2.75x hidden
    VOCAB_SIZE = 152064
    TOP_K = 32
    
    BATCH_SIZE = 64
    SEQ_LEN = 64          # Train on sequences of 64 tokens
    LR = 3e-4
    EPOCHS = 3
    WARMUP_STEPS = 100
    
    DATA_PATH = "data/teacher/teacher_data.npz"
    SAVE_PATH = "models/draft-model"
    
    # Load data
    print("Loading training data...")
    data = np.load(DATA_PATH)
    input_ids = data['input_ids']     # (N,) int32
    target_ids = data['target_ids']   # (N,) int32
    top_logits = data['top_logits']   # (N, TOP_K) float16
    top_indices = data['top_indices'] # (N, TOP_K) int32
    
    N = len(input_ids)
    print(f"  {N:,} token positions loaded")
    print(f"  Top-{TOP_K} logits per position")
    
    # Create model
    print(f"\nCreating draft model: {NUM_LAYERS}L, {HIDDEN}H, {NUM_HEADS}h...")
    model = DraftModel(VOCAB_SIZE, HIDDEN, NUM_LAYERS, NUM_HEADS, HEAD_DIM, INTERMEDIATE)
    mx.eval(model.parameters())
    
    # Count params
    n_params = sum(p.size for p in model.parameters().values() if isinstance(p, mx.array))
    print(f"  Parameters: {n_params / 1e6:.1f}M")
    
    # Optimizer with cosine schedule
    total_steps = (N // (BATCH_SIZE * SEQ_LEN)) * EPOCHS
    print(f"  Total steps: {total_steps}")
    
    scheduler = optim.cosine_decay(LR, total_steps, 1e-5)
    optimizer = optim.AdamW(learning_rate=scheduler, weight_decay=0.01)
    
    # KL distillation loss
    def loss_fn(model, batch_input, batch_target, batch_top_logits, batch_top_indices):
        """KL divergence between student and teacher on top-K logits."""
        # Student forward
        student_logits = model(batch_input)  # (B, L, vocab)
        
        # Cross-entropy on target tokens (standard next-token prediction)
        B, L, V = student_logits.shape
        flat_logits = student_logits.reshape(-1, V)
        flat_target = batch_target.reshape(-1)
        ce_loss = nn.losses.cross_entropy(flat_logits, flat_target, reduction='mean')
        
        # KL distillation on teacher's top-K distribution
        # Gather student logits at teacher's top-K indices
        flat_student = student_logits.reshape(-1, V)  # (B*L, V)
        flat_top_idx = batch_top_indices.reshape(-1, TOP_K)  # (B*L, K)
        flat_top_logits = batch_top_logits.reshape(-1, TOP_K)  # (B*L, K)
        
        # Get student's logits at teacher's top positions
        student_at_top = mx.take_along_axis(flat_student, flat_top_idx, axis=1)  # (B*L, K)
        
        # Temperature-scaled softmax
        T = 2.0
        teacher_probs = mx.softmax(flat_top_logits / T, axis=-1)
        student_log_probs = mx.log(mx.softmax(student_at_top / T, axis=-1) + 1e-8)
        kl_loss = mx.mean(mx.sum(teacher_probs * (mx.log(teacher_probs + 1e-8) - student_log_probs), axis=-1))
        
        # Combined loss: CE + KL
        alpha = 0.5
        total_loss = alpha * ce_loss + (1 - alpha) * T * T * kl_loss
        return total_loss
    
    loss_and_grad = nn.value_and_grad(model, loss_fn)
    
    # Training loop
    print("\nTraining...")
    step = 0
    
    for epoch in range(EPOCHS):
        # Shuffle
        perm = np.random.permutation(N - SEQ_LEN)
        epoch_loss = 0
        epoch_steps = 0
        t0 = time.time()
        
        for batch_start in range(0, len(perm) - BATCH_SIZE * SEQ_LEN, BATCH_SIZE * SEQ_LEN):
            # Build batch: random windows from the data
            batch_inputs = []
            batch_targets = []
            batch_tl = []
            batch_ti = []
            
            for b in range(BATCH_SIZE):
                idx = perm[batch_start + b * SEQ_LEN]
                batch_inputs.append(input_ids[idx:idx + SEQ_LEN])
                batch_targets.append(target_ids[idx:idx + SEQ_LEN])
                batch_tl.append(top_logits[idx:idx + SEQ_LEN])
                batch_ti.append(top_indices[idx:idx + SEQ_LEN])
            
            b_input = mx.array(np.array(batch_inputs))
            b_target = mx.array(np.array(batch_targets))
            b_tl = mx.array(np.array(batch_tl, dtype=np.float32))
            b_ti = mx.array(np.array(batch_ti))
            
            loss, grads = loss_and_grad(model, b_input, b_target, b_tl, b_ti)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss)
            
            epoch_loss += loss.item()
            epoch_steps += 1
            step += 1
            
            if step % 50 == 0:
                avg_loss = epoch_loss / epoch_steps
                elapsed = time.time() - t0
                tps = epoch_steps * BATCH_SIZE * SEQ_LEN / elapsed
                print(f"  Epoch {epoch+1} Step {step}: loss={avg_loss:.4f} "
                      f"({tps:.0f} tok/s, lr={scheduler(step):.2e})")
        
        avg_loss = epoch_loss / max(epoch_steps, 1)
        elapsed = time.time() - t0
        print(f"  Epoch {epoch+1} done: avg_loss={avg_loss:.4f} ({elapsed:.0f}s)")
    
    # Save model
    os.makedirs(SAVE_PATH, exist_ok=True)
    weights = dict(model.parameters())
    flat_weights = {}
    
    def flatten_dict(d, prefix=""):
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                flatten_dict(v, key)
            elif isinstance(v, list):
                for i, item in enumerate(v):
                    if isinstance(item, dict):
                        flatten_dict(item, f"{key}.{i}")
                    elif isinstance(item, mx.array):
                        flat_weights[f"{key}.{i}"] = item
            elif isinstance(v, mx.array):
                flat_weights[key] = v
    
    flatten_dict(weights)
    mx.save_safetensors(f"{SAVE_PATH}/model.safetensors", flat_weights)
    
    # Save config
    config = {
        "hidden": HIDDEN,
        "num_layers": NUM_LAYERS,
        "num_heads": NUM_HEADS,
        "head_dim": HEAD_DIM,
        "intermediate": INTERMEDIATE,
        "vocab_size": VOCAB_SIZE,
    }
    with open(f"{SAVE_PATH}/config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"\nModel saved to {SAVE_PATH}/")
    print(f"  Weights: {os.path.getsize(f'{SAVE_PATH}/model.safetensors')/1024**2:.1f} MB")

if __name__ == "__main__":
    train()
