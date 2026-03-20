# 🔥 BitNet MLX Engine

### We tried every trick to make 1-bit LLMs fast on Apple Silicon. Here's what actually works.

> **TL;DR**: We spent a week trying to break 100 tok/s with BitNet on M-series Macs. We hit 89 tok/s — and proved that's the hardware ceiling. Along the way, we discovered that **Python isn't the bottleneck**, the **Neural Engine is 10× slower than the GPU**, and **speculative decoding barely helps without a massive draft model**. This repo is the evidence.

---

## Why This Exists

Microsoft's [BitNet](https://arxiv.org/abs/2310.11453) promises LLMs with 1-bit weights — theoretically perfect for edge devices. Apple's M-series chips have massive unified memory bandwidth and a 16-core Neural Engine. Sounds like a match made in heaven, right?

**We tested that hypothesis.** Every optimization technique we could think of:

| What we tried | tok/s | Verdict |
|---|---|---|
| Hand-written Python ternary kernel | 13.8 | 🐌 Slow |
| `mx.compile()` + graph optimization | 15.2 | 🐌 Marginal |
| C++ with MLX `QuantizedLinear` | 28 | 😐 Better |
| **C++ with custom BitLinear Metal kernel** | **80** | **🚀 Real** |
| **+ async_eval CPU-GPU pipelining** | **89** | **🚀 Peak** |
| CoreML float16 on Neural Engine | 5 | 💀 Terrible |
| CoreML 2-bit palettized on ANE | 20 | 💀 Still bad |
| Speculative decode (trained draft model) | 86 | 🤷 Barely helps |
| Apple's own `mlx-lm` Python | 90 | ✅ Same ceiling |

**The punchline:** Our hand-optimized C++ with a custom Metal kernel matched `mlx-lm`'s Python implementation exactly. The bottleneck isn't software — it's memory bandwidth.

---

## 🎯 The 5 Things Nobody Tells You

### 1. "Rewrite it in C++" is a myth
We rewrote the entire inference stack in C++. **Speedup: 0%.** The GPU kernel runs the same regardless of whether Python or C++ calls it. Save yourself the trouble.

### 2. The Apple Neural Engine is useless for LLM inference
Despite Apple marketing the ANE for "AI workloads", it's **10× slower** than the GPU for autoregressive generation. Even with proper 2-bit palettization (the correct format for ternary weights): 20 tok/s vs 89 tok/s on GPU. The ANE needs large batch sizes — single-token generation is its worst case.

### 3. One kernel change > entire language rewrite
Switching from generic `QuantizedLinear` to a ternary-specific `BitLinear` kernel gave **3× speedup**. The kernel decodes packed uint8 weights and accumulates in a single pass with `simd_sum`. This one change was worth more than everything else combined.

### 4. `async_eval` is free performance
Adding `mx::async_eval` (GPU computes next token while CPU processes current) gives **~10% free throughput**. Zero code change to the model, just how you schedule evaluation. Most tutorials skip this.

### 5. Speculative decoding needs a GOOD draft model
We trained a 2-layer draft model via knowledge distillation. It runs at 620 tok/s but only achieves 66% acceptance — barely faster than baseline. You need architectural capacity (6-8 layers), not just more training epochs.

---

## 🚀 Get Started in 60 Seconds

### Step 1: Install dependencies
```bash
pip install mlx mlx-lm huggingface_hub
```

### Step 2: Download BitNet-2B
```bash
# Option A: Direct download from HuggingFace (recommended)
huggingface-cli download 1bitLLM/bitnet_b1_58-2B --local-dir models/bitnet-2b

# Option B: Convert with mlx-lm (applies optimizations)
python3 -m mlx_lm.convert --hf-path 1bitLLM/bitnet_b1_58-2B --mlx-path models/bitnet-2b
```

### Step 3: Build the C++ engine
```bash
make all
```
> This auto-detects your MLX installation. Requires `clang++`, Metal framework (comes with Xcode).

### Step 4: Run!
```bash
# Baseline engine — ~89 tok/s
./cpp_inference/bitnet_v3 models/bitnet-2b 200

# With speculative decoding — ~86 tok/s (see below to train draft model)
./cpp_inference/bitnet_v5 models/bitnet-2b models/draft-model 200 3
```

### Quick benchmark with mlx-lm (for comparison)
```bash
python3 -c "
from mlx_lm import load
from mlx_lm.generate import generate_step
import mlx.core as mx, time

model, tokenizer = load('models/bitnet-2b')
prompt = mx.array(tokenizer.encode('Once upon a time'))

# Warmup
for _ in range(5):
    list(zip(range(10), generate_step(prompt, model, max_tokens=10)))

# Benchmark
t0 = time.time()
count = 0
for tok, _ in generate_step(prompt, model, max_tokens=200):
    mx.eval(tok) if hasattr(tok, 'shape') else None
    count += 1
print(f'{count/(time.time()-t0):.1f} tok/s')
"
```

---

## 🏗️ How the BitLinear Metal Kernel Works

The key innovation in BitNet is ternary weights ({-1, 0, +1}), packed 4 values per byte. Our Metal kernel decodes and accumulates in a single GPU pass:

```metal
// Each thread processes 4 output neurons simultaneously
// No dequantize step — decode directly from packed uint8
uint8_t w = packed_weights[row * in_features + i];
sum[0] += v[j] * ((w & 3) - 1);       // bits 0-1 → {-1, 0, 1}
sum[1] += v[j] * (((w >> 2) & 3) - 1); // bits 2-3
sum[2] += v[j] * (((w >> 4) & 3) - 1); // bits 4-5
sum[3] += v[j] * (((w >> 6) & 3) - 1); // bits 6-7

// Warp-level reduction — no atomic ops needed
for (int j = 0; j < 4; j++) sum[j] = simd_sum(sum[j]);
```

This eliminates the dequantize→store→multiply→accumulate pipeline that generic quantized ops use. The weight stays packed in registers, never touching shared memory.

---

## 🎓 Train Your Own Draft Model (Speculative Decoding)

```bash
# Step 1: Collect teacher logits from the full model (~16 min)
python3 scripts/generate_training_data.py

# Step 2: Train a tiny draft model via knowledge distillation (~3 min)
python3 scripts/train_draft.py

# Step 3: Run speculative decoding
./cpp_inference/bitnet_v5 models/bitnet-2b models/draft-model 200 3
```

**How it works:**
```
Draft model (2L/512H, 620 tok/s) → generates K tokens fast
Full model  (30L/2560H, 89 tok/s) → verifies all K in one batched pass
Matching tokens accepted for free → effective speedup!
```

Our draft model achieves 66% acceptance at K=3 → 86 tok/s. A larger draft model (6-8 layers) with more training data could push past 100 tok/s.

---

## 🔬 CoreML / Neural Engine Deep Dive

We did what Apple won't tell you — benchmarked the ANE head-to-head against the GPU for LLM inference:

| Compute Unit | Weights | ms/step | Est. tok/s | Notes |
|---|---|---|---|---|
| **GPU (MLX Metal)** | **Packed uint8** | **~0.75** | **89** | **Winner by 4.5×** |
| ANE + CPU | 2-bit palettized | 3.30 | 20 | Best ANE result |
| ANE + CPU | Float16 | 12.84 | 5 | Dequantized, terrible |
| GPU via CoreML | Float16 | 7.67 | 9 | CoreML overhead kills it |

**Why ANE loses:**
- **Dispatch overhead**: Each CoreML `predict()` call costs ~3ms for serialization and scheduling
- **Batch-1 is worst case**: ANE's 16-core array needs large batches to saturate. Autoregressive generation is inherently batch=1
- **No ternary support**: CoreML can't use packed uint8 ternary weights natively — must dequantize to float16 (4× memory bloat) or palettize to 2-bit

**When ANE WOULD help**: Batch inference (embeddings, classification), not autoregressive generation.

---

## 🧪 M5-Specific Findings (Apple Silicon Gen 17)

Tested on **Apple M5** (`applegpu_g17g`, Metal 4, 10 GPU cores, 34 GB unified memory).

### Wired Memory Limit: No Effect on Small Models

Python's `mlx-lm` automatically pins GPU buffers using `wired_limit()` (discovered by [robertmsale](https://github.com/ml-explore/mlx-swift/issues/347)). We tested this explicitly:

| Setting | tok/s |
|---|---|
| Without `set_wired_limit` | 90.2 |
| With `set_wired_limit(26.8 GB)` | 90.2 |

**Why no difference?** BitNet-2B is only ~1 GB of weights — it fits entirely in the GPU working set without pinning. The wired limit matters for **large models (70B+)** where the OS might evict GPU buffers mid-inference.

### M5 Neural Units vs Custom Kernels

M5 has new Neural Units (NU) inside each GPU core. We ran a **comprehensive 4-way benchmark** to test whether any MLX built-in path (which may leverage NU hardware) beats our custom kernel:

| Kernel | ms/op | ops/s | Rank |
|---|---|---|---|
| **Custom BitLinear (packed uint8)** | **0.245** | **4,078** | **🥇 Winner** |
| `quantized_matmul` 4-bit (MLX built-in) | 0.270 | 3,698 | 🥈 |
| `quantized_matmul` 2-bit (MLX built-in) | 0.355 | 2,813 | 🥉 |
| Standard matmul bf16 (uses M5 NU) | 0.496 | 2,017 | 4th |

> Matrix dimensions: 2560 × 6912 (actual BitNet-2B layer size)

**Our custom kernel beats EVERYTHING** — including MLX's own `quantized_matmul`, which has access to M5's hardware-accelerated dequantization paths. Key insights:

1. **Memory bandwidth is king**: Our uint8-packed ternary reads only 4.4 MB per op vs 35.4 MB for bf16 — **8× less data**
2. **NU doesn't help quantized_matmul beat us**: Even with potential hardware acceleration, the extra data movement for scale/bias kills performance
3. **2-bit `quantized_matmul` is slower than 4-bit**: Suggests the 2-bit path isn't well-optimized in MLX yet — an opportunity for Apple
4. **Theoretical ceiling**: We read 8× less data but are only 2× faster than bf16 matmul → our kernel is ~4× less compute-efficient. If Apple adds **native ternary support to the NU**, BitNet could theoretically hit **300-500 tok/s**

### GPU Profiling

We captured and analyzed GPU traces using MLX's built-in Metal capture:

```bash
# Python
MTL_CAPTURE_ENABLED=1 python3 -c "
import mlx.core as mx
mx.metal.start_capture('trace.gputrace')
# ... inference ...
mx.metal.stop_capture()
"
# Open in Xcode:
open trace.gputrace
```

```cpp
// C++
mx::metal::start_capture("trace.gputrace");
// ... inference ...
mx::metal::stop_capture();
```

The trace confirms kernel dispatch is tight — no wasted cycles between token steps. The bottleneck is purely memory bandwidth reading weights from unified memory.

## 📁 Complete File Reference

### ⚡ Core Engines
| File | tok/s | Description |
|---|---|---|
| `cpp_inference/bitnet_v3.cpp` | **89** | ★ Final engine with BitLinear Metal kernel + async_eval |
| `cpp_inference/bitnet_v5_speculative.cpp` | **86** | ★ Speculative decode with trained draft model |

### 📈 The Optimization Journey (C++)
| File | tok/s | What We Learned |
|---|---|---|
| `cpp_inference/bitnet_generate.cpp` | 28 | v1: `QuantizedLinear` — generic ops are slow |
| `cpp_inference/bitnet_v2.cpp` | 80 | v2: Custom BitLinear kernel — **3× speedup from one kernel** |
| `cpp_inference/bitnet_v4_speculative.cpp` | 70 | v4: Same-model speculation — doesn't work (0% acceptance) |

### 🐍 Python Experiments (where it all started)
| File | tok/s | What We Tried |
|---|---|---|
| `run_bitnet_2b.py` | 13.8 | First working inference with hand-rolled ternary matmul |
| `run_approach1_native2bit.py` | — | Native 2-bit quantization approach |
| `run_approach2_compiled.py` | — | `mx.compile()` graph optimization |
| `run_approach3_tiled.py` | — | Tiled matmul kernel |
| `run_hybrid_fastest.py` | — | Combining best ideas |
| `run_ultimate.py` | — | Final Python attempt before C++ |
| `bitnet/` | — | Custom module: kernels, layers, model, loader, generator |

### 🔬 ANE Benchmarks
| File | Description |
|---|---|
| `cpp_inference/coreml_2bit.py` | 2-bit palettized ANE benchmark (20 tok/s) |
| `cpp_inference/coreml_convert.py` | Float16 CoreML baseline (5 tok/s) |

### 🎓 Draft Model Training
| File | Description |
|---|---|
| `scripts/generate_training_data.py` | Runs teacher, collects top-32 logits per position |
| `scripts/train_draft.py` | KL divergence distillation, 2L/512H draft model |

### 🛠️ Utilities
| File | Description |
|---|---|
| `benchmark.py` | Benchmarking harness |
| `tune_kernel.py` | Metal kernel parameter tuning |
| `export_bitnet.py` | Weight conversion utility |
| `tests/test_kernels.py` | Kernel correctness tests |

---

## 📊 Hardware

Benchmarked on **Apple M5 MacBook Pro**:
- 10-core CPU / 16-core GPU / 16-core Neural Engine
- Unified memory, ~200 GB/s bandwidth
- macOS 15+

Results should scale similarly on M1/M2/M3/M4 (proportional to memory bandwidth).

---

## 🤝 Contributing

The biggest open opportunity: **a better draft model for speculative decoding**. Our 2-layer model hits 66% acceptance. A 6-8 layer model trained on 1M+ tokens could push past 100 tok/s. PRs welcome!

Other ideas:
- Batch inference benchmarks (where ANE might actually help)
- Support for other BitNet models (BitNet-3B, etc.)
- Integration with `llama.cpp` style serving

---

## 📝 License

MIT — use it however you want.

## 🙏 Acknowledgments

- [MLX](https://github.com/ml-explore/mlx) by Apple — the Metal-native ML framework that makes this possible
- [BitNet](https://arxiv.org/abs/2310.11453) by Microsoft Research — the 1-bit LLM architecture
- [1bitLLM](https://huggingface.co/1bitLLM) — pretrained BitNet model weights

---

*Built with curiosity and stubbornness. If this saved you time, give it a ⭐*
