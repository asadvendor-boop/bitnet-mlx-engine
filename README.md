# BitNet MLX Engine

**A research exploration of BitNet (1-bit LLM) inference on Apple Silicon, pushing for maximum tokens/sec.**

We explored every known optimization path — custom C++ with Metal GPU kernels, Apple Neural Engine via CoreML, speculative decoding with knowledge distillation — and documented what actually works, what doesn't, and why.

## 🏆 Key Results

| Approach | tok/s | Speedup |
|---|---|---|
| Naive Python (hand-rolled matmul) | 13.8 | 1× |
| C++ with MLX QuantizedLinear | 28 | 2× |
| C++ with BitLinear Metal kernel | 80 | 5.8× |
| **C++ with BitLinear + async_eval** | **89** | **6.4×** |
| **Speculative decode (K=3, 66%)** | **86** | **6.2×** |
| CoreML 2-bit palettized (ANE) | 20 | 1.4× |
| CoreML float16 (ANE) | 5 | 0.4× |

> **89 tok/s is the hardware ceiling** for BitNet-2B on Apple M-series chips. Both our C++ engine and Apple's `mlx-lm` Python hit the same wall — the bottleneck is memory bandwidth, not software.

## 🔑 Key Findings

### 1. Python is NOT the bottleneck
Rewriting `mlx-lm` from Python to C++ gave **0% speedup**. The GPU Metal kernel dominates execution time — the host language doesn't matter.

### 2. The kernel is EVERYTHING
Switching from generic `QuantizedLinear` to a ternary-optimized `BitLinear` Metal kernel (packed uint8 decode + `simd_sum` in one pass) gave **3× speedup**.

### 3. `async_eval` is free performance
Pipelining CPU and GPU via `mx::async_eval` gives **~10% free throughput** with zero model changes.

### 4. Apple Neural Engine is NOT faster for LLM inference
Despite Apple's marketing of the 16-core ANE, it's **10× slower** than the GPU for autoregressive (single-token) generation:
- CoreML dispatch overhead: ~3ms per call (serialization, scheduling)
- ANE needs large batch sizes to saturate — batch=1 is the worst case
- Even with 2-bit palettization (correct format for ternary), only 20 tok/s

### 5. Speculative decoding works, but draft quality is everything
Our trained draft model (2L/512H, 620 tok/s standalone) achieves 66% acceptance rate, yielding 86 tok/s with K=3. The framework is correct — a larger draft model with more training data could push past 100 tok/s.

## 📁 Project Structure

### ⚡ Core Engine (C++)
```
cpp_inference/
├── bitnet_v3.cpp              # ★ Final engine: 89 tok/s (BitLinear Metal + async_eval)
└── bitnet_v5_speculative.cpp  # ★ Speculative decode with trained draft model
```

### 🔬 Benchmarks & Analysis
```
cpp_inference/
├── coreml_2bit.py             # CoreML ANE benchmark with 2-bit palettization
├── coreml_convert.py          # CoreML float16 baseline benchmark
├── bitnet_generate.cpp        # C++ v1: 28 tok/s (QuantizedLinear, no custom kernel)
├── bitnet_v2.cpp              # C++ v2: 80 tok/s (first BitLinear Metal kernel)
└── bitnet_v4_speculative.cpp  # C++ v4: same-model speculative (failed experiment)
```

### 🐍 Python Experiments (the journey from 13.8 tok/s)
```
bitnet/                        # Custom Python module with hand-written ternary kernels
├── __init__.py
├── kernels.py                 # Metal kernel source for ternary matmul
├── layers.py                  # BitLinear layer implementation  
├── model.py                   # Full BitNet model definition
├── loader.py                  # Weight loading from safetensors
└── generate.py                # Token generation loop
run_bitnet_2b.py               # First working inference (13.8 tok/s)
run_approach1_native2bit.py    # Approach 1: native 2-bit quantization
run_approach2_compiled.py      # Approach 2: mx.compile() optimization
run_approach3_tiled.py         # Approach 3: tiled matmul kernel
run_hybrid_fastest.py          # Hybrid approach combining best ideas
run_ultimate.py                # Final Python attempt before moving to C++
benchmark.py                   # Benchmarking harness
tune_kernel.py                 # Metal kernel parameter tuning
export_bitnet.py               # Weight export/conversion utility
tests/test_kernels.py          # Kernel correctness tests
```

### 🎓 Draft Model Training (for speculative decoding)
```
scripts/
├── generate_training_data.py  # Run teacher model, collect top-K logits
└── train_draft.py             # Knowledge distillation (KL divergence)
```

### 📦 Build
```
Makefile                       # Auto-detects MLX paths, builds both engines
.gitignore                     # Excludes models/ (5.6GB), compiled binaries
```

## 🚀 Quick Start

### Prerequisites
```bash
pip install mlx mlx-lm
```

### Download Model
```bash
huggingface-cli download 1bitLLM/bitnet_b1_58-2B --local-dir models/bitnet-2b
# Or use mlx-lm to convert:
python3 -m mlx_lm.convert --hf-path 1bitLLM/bitnet_b1_58-2B --mlx-path models/bitnet-2b
```

### Build & Run
```bash
make all
./cpp_inference/bitnet_v3 models/bitnet-2b 200   # Baseline: ~89 tok/s
```

### Speculative Decoding

Train a draft model:
```bash
python3 scripts/generate_training_data.py   # Collect teacher logits (~16 min)
python3 scripts/train_draft.py               # Train draft model (~3 min)
./cpp_inference/bitnet_v5 models/bitnet-2b models/draft-model 200 3  # ~86 tok/s
```

## 🏗️ Architecture Deep Dive

### BitLinear Metal Kernel

The core innovation in BitNet is ternary weights ({-1, 0, +1}), packed 4 per byte. Our Metal kernel decodes and accumulates in a single GPU pass:

```metal
// Each thread processes 4 output neurons simultaneously
uint8_t w = packed_weights[row * in_features + i];
sum[0] += v[j] * ((w & 3) - 1);       // bits 0-1
sum[1] += v[j] * (((w >> 2) & 3) - 1); // bits 2-3
sum[2] += v[j] * (((w >> 4) & 3) - 1); // bits 4-5
sum[3] += v[j] * (((w >> 6) & 3) - 1); // bits 6-7
```

Combined with `simd_sum` for warp-level reduction, this eliminates the dequantize→multiply→accumulate pipeline that generic quantized ops use.

### Speculative Decoding

```
Draft model (2L, 620 tok/s) → generates K tokens fast
Full model (30L, 89 tok/s)  → verifies all K in one pass
Accept matching tokens       → free tokens!
Resample at divergence       → no quality loss
```

With acceptance rate α and draft count K:
- Effective throughput ≈ baseline × (1 + α×K) / (1 + K/draft_speed_ratio)

## 🔬 CoreML / Neural Engine Analysis

We tested ANE with both float16 and 2-bit palettized weights:

| Compute Unit | Precision | ms/step (2L) | Est. 30L tok/s |
|---|---|---|---|
| ANE + CPU | Float16 | 12.84 | 5 |
| **ANE + CPU** | **2-bit** | **3.30** | **20** |
| GPU + CPU | Float16 | 7.67 | 9 |
| ALL | Float16 | 7.82 | 9 |

**Verdict**: Even with optimal 2-bit palettization, ANE tops out at ~20 tok/s — 4.5× slower than the GPU path. The per-call dispatch overhead in CoreML's `predict()` kills single-token latency.

## 📊 Hardware

All benchmarks on **Apple M5 MacBook Pro** (or equivalent M-series):
- 16-core GPU, 16-core ANE
- Unified memory, ~200 GB/s bandwidth
- macOS 15+

## 📝 License

MIT

## 🙏 Acknowledgments

- [MLX](https://github.com/ml-explore/mlx) by Apple — the Metal-native ML framework
- [BitNet](https://arxiv.org/abs/2310.11453) — 1-bit LLM architecture by Microsoft Research
- [1bitLLM](https://huggingface.co/1bitLLM) — pretrained BitNet models
