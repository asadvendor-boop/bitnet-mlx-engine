"""
Custom Metal GPU kernels for BitNet 1.58-bit ternary matrix multiplication.

Weight Encoding (2-bit per weight, packed into uint32):
  0b00 = 0  (skip)
  0b01 = +1 (add activation)
  0b10 = -1 (subtract activation)
  
  16 weights packed per uint32.

The key insight: BitNet replaces expensive float multiply-accumulate with
simple conditional add/subtract. This is trivial to parallelize on the GPU.
"""

import mlx.core as mx
import numpy as np


# ---------------------------------------------------------------------------
# Weight packing / unpacking utilities
# ---------------------------------------------------------------------------

def pack_ternary_weights(weights: mx.array) -> mx.array:
    """
    Pack a float ternary weight matrix {-1, 0, +1} into 2-bit packed uint32.
    
    Args:
        weights: float array of shape (out_features, in_features) 
                 containing only values in {-1, 0, +1}
    
    Returns:
        packed: uint32 array of shape (out_features, ceil(in_features / 16))
    """
    # Move to numpy for bit manipulation
    w = np.array(weights)
    out_features, in_features = w.shape
    
    # Pad in_features to multiple of 16
    pad_size = (16 - in_features % 16) % 16
    if pad_size > 0:
        w = np.pad(w, ((0, 0), (0, pad_size)), constant_values=0)
    
    padded_in = w.shape[1]
    packed_cols = padded_in // 16
    
    # Encode: 0 -> 0b00, +1 -> 0b01, -1 -> 0b10
    encoded = np.zeros_like(w, dtype=np.uint32)
    encoded[w == 1] = 1    # 0b01
    encoded[w == -1] = 2   # 0b10
    # w == 0 stays 0b00
    
    # Reshape to groups of 16 and pack into uint32
    encoded = encoded.reshape(out_features, packed_cols, 16)
    
    packed = np.zeros((out_features, packed_cols), dtype=np.uint32)
    for i in range(16):
        packed |= (encoded[:, :, i] << (i * 2))
    
    return mx.array(packed)


def unpack_ternary_weights(packed: mx.array, in_features: int) -> mx.array:
    """
    Unpack 2-bit packed uint32 back to float ternary weights.
    Mainly for verification/testing.
    """
    p = np.array(packed)
    out_features, packed_cols = p.shape
    
    weights = np.zeros((out_features, packed_cols * 16), dtype=np.float32)
    
    for i in range(16):
        bits = (p >> (i * 2)) & 0x3
        col_idx = i
        weights[:, col_idx::16] = 0  # Reset
        # Actually we need contiguous unpacking
    
    # Simpler approach: unpack sequentially
    weights = np.zeros((out_features, packed_cols * 16), dtype=np.float32)
    for i in range(16):
        bits = (p >> (i * 2)) & 0x3
        w_col = np.zeros_like(bits, dtype=np.float32)
        w_col[bits == 1] = 1.0
        w_col[bits == 2] = -1.0
        weights[:, np.arange(packed_cols) * 16 + i] = w_col
    
    return mx.array(weights[:, :in_features])


# ---------------------------------------------------------------------------
# Metal kernel source for ternary matrix-vector multiplication
# ---------------------------------------------------------------------------

TERNARY_MATVEC_KERNEL = """
    // V4 TUNED ternary matvec kernel.
    // 4 rows/thread (sweet spot — more registers = less occupancy).
    // Branchless ternary decode: sign = (bits & 1) - (bits >> 1)
    // Optimal threadgroup=128 found via empirical tuning.
    
    uint tid = thread_position_in_grid.x;
    uint out_features = packed_weights_shape[0];
    uint packed_cols = packed_weights_shape[1];
    uint in_feat = x_shape[0];
    
    uint base_row = tid * 4;
    if (base_row >= out_features) return;
    uint rows = min(4u, out_features - base_row);
    
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    
    for (uint pc = 0; pc < packed_cols; pc++) {
        uint p0 = packed_weights[(base_row + 0) * packed_cols + pc];
        uint p1 = (rows > 1) ? packed_weights[(base_row + 1) * packed_cols + pc] : 0;
        uint p2 = (rows > 2) ? packed_weights[(base_row + 2) * packed_cols + pc] : 0;
        uint p3 = (rows > 3) ? packed_weights[(base_row + 3) * packed_cols + pc] : 0;
        
        uint base_col = pc * 16;
        
        for (uint i = 0; i < 16; i++) {
            uint col = base_col + i;
            if (col >= in_feat) break;
            
            float val = static_cast<float>(x[col]);
            uint shift = i * 2;
            
            uint b0 = (p0 >> shift) & 0x3; acc0 += float(int(b0 & 1u) - int(b0 >> 1u)) * val;
            uint b1 = (p1 >> shift) & 0x3; acc1 += float(int(b1 & 1u) - int(b1 >> 1u)) * val;
            uint b2 = (p2 >> shift) & 0x3; acc2 += float(int(b2 & 1u) - int(b2 >> 1u)) * val;
            uint b3 = (p3 >> shift) & 0x3; acc3 += float(int(b3 & 1u) - int(b3 >> 1u)) * val;
        }
    }
    
    float s = static_cast<float>(scale[0]);
    out[base_row + 0] = static_cast<T>(acc0 * s);
    if (rows > 1) out[base_row + 1] = static_cast<T>(acc1 * s);
    if (rows > 2) out[base_row + 2] = static_cast<T>(acc2 * s);
    if (rows > 3) out[base_row + 3] = static_cast<T>(acc3 * s);
"""

TERNARY_MATMUL_BATCH_KERNEL = """
    // OPTIMIZED batched ternary matmul kernel.
    // x: (batch, in_features) -> out: (batch, out_features)
    // Each thread computes 4 consecutive output rows for one batch element.
    // Branchless ternary decode.
    
    uint tid = thread_position_in_grid.x;
    uint out_features = packed_weights_shape[0];
    uint batch_size = x_shape[0];
    
    uint threads_per_batch = (out_features + 3) / 4;
    uint total_threads = batch_size * threads_per_batch;
    if (tid >= total_threads) return;
    
    uint b = tid / threads_per_batch;
    uint local_tid = tid % threads_per_batch;
    uint base_row = local_tid * 4;
    
    if (base_row >= out_features) return;
    uint rows_this_thread = min(4u, out_features - base_row);
    
    uint packed_cols = packed_weights_shape[1];
    uint in_features = x_shape[1];
    
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    
    for (uint pc = 0; pc < packed_cols; pc++) {
        uint p0 = packed_weights[(base_row + 0) * packed_cols + pc];
        uint p1 = (rows_this_thread > 1) ? packed_weights[(base_row + 1) * packed_cols + pc] : 0;
        uint p2 = (rows_this_thread > 2) ? packed_weights[(base_row + 2) * packed_cols + pc] : 0;
        uint p3 = (rows_this_thread > 3) ? packed_weights[(base_row + 3) * packed_cols + pc] : 0;
        
        uint base_col = pc * 16;
        
        for (uint i = 0; i < 16; i++) {
            uint col = base_col + i;
            if (col >= in_features) break;
            
            float val = static_cast<float>(x[b * in_features + col]);
            uint shift = i * 2;
            
            uint bv0 = (p0 >> shift) & 0x3;
            uint bv1 = (p1 >> shift) & 0x3;
            uint bv2 = (p2 >> shift) & 0x3;
            uint bv3 = (p3 >> shift) & 0x3;
            
            acc0 += float(int(bv0 & 1u) - int(bv0 >> 1u)) * val;
            acc1 += float(int(bv1 & 1u) - int(bv1 >> 1u)) * val;
            acc2 += float(int(bv2 & 1u) - int(bv2 >> 1u)) * val;
            acc3 += float(int(bv3 & 1u) - int(bv3 >> 1u)) * val;
        }
    }
    
    float s = static_cast<float>(scale[0]);
    uint out_base = b * out_features;
    out[out_base + base_row + 0] = static_cast<T>(acc0 * s);
    if (rows_this_thread > 1) out[out_base + base_row + 1] = static_cast<T>(acc1 * s);
    if (rows_this_thread > 2) out[out_base + base_row + 2] = static_cast<T>(acc2 * s);
    if (rows_this_thread > 3) out[out_base + base_row + 3] = static_cast<T>(acc3 * s);
"""


# ---------------------------------------------------------------------------
# Python wrappers for Metal kernels
# ---------------------------------------------------------------------------

_kernel_cache = {}


def _get_ternary_matvec_kernel():
    """Get or compile the ternary matvec Metal kernel."""
    if "matvec" not in _kernel_cache:
        _kernel_cache["matvec"] = mx.fast.metal_kernel(
            name="ternary_matvec",
            input_names=["packed_weights", "x", "scale"],
            output_names=["out"],
            source=TERNARY_MATVEC_KERNEL,
        )
    return _kernel_cache["matvec"]


def _get_ternary_matmul_batch_kernel():
    """Get or compile the batched ternary matmul Metal kernel."""
    if "matmul_batch" not in _kernel_cache:
        _kernel_cache["matmul_batch"] = mx.fast.metal_kernel(
            name="ternary_matmul_batch",
            input_names=["packed_weights", "x", "scale"],
            output_names=["out"],
            source=TERNARY_MATMUL_BATCH_KERNEL,
        )
    return _kernel_cache["matmul_batch"]


def ternary_matmul(
    packed_weights: mx.array,
    x: mx.array,
    scale: mx.array,
) -> mx.array:
    """
    Perform ternary matrix multiplication on the GPU via custom Metal kernel.
    
    Computes: out = (W_ternary @ x) * scale
    where W_ternary is the unpacked ternary weight matrix {-1, 0, +1}.
    
    Args:
        packed_weights: uint32 (out_features, packed_cols) - 2-bit packed weights
        x: float16/float32 - input activations, shape (in_features,) or (batch, in_features)
        scale: float scalar or (1,) - output scale factor
        
    Returns:
        out: same dtype as x, shape (out_features,) or (batch, out_features)
    """
    out_features = packed_weights.shape[0]
    
    # Ensure scale is 1D
    if scale.ndim == 0:
        scale = scale.reshape(1)
    
    if x.ndim == 1:
        # Single vector: (in_features,) -> (out_features,)
        # 4 rows/thread, threadgroup=128 (empirically optimal)
        num_threads = (out_features + 3) // 4
        kernel = _get_ternary_matvec_kernel()
        out = kernel(
            inputs=[packed_weights, x, scale],
            template=[("T", x.dtype)],
            grid=(num_threads, 1, 1),
            threadgroup=(min(128, num_threads), 1, 1),
            output_shapes=[(out_features,)],
            output_dtypes=[x.dtype],
        )[0]
    else:
        # Batched: (batch, in_features) -> (batch, out_features)
        batch_size = x.shape[0]
        threads_per_batch = (out_features + 3) // 4
        total_threads = batch_size * threads_per_batch
        kernel = _get_ternary_matmul_batch_kernel()
        out = kernel(
            inputs=[packed_weights, x, scale],
            template=[("T", x.dtype)],
            grid=(total_threads, 1, 1),
            threadgroup=(min(256, total_threads), 1, 1),
            output_shapes=[(batch_size, out_features)],
            output_dtypes=[x.dtype],
        )[0]
    
    return out


def ternary_matmul_naive(
    packed_weights: mx.array,
    x: mx.array, 
    scale: mx.array,
    in_features: int,
) -> mx.array:
    """
    Naive (non-Metal) ternary matmul for correctness verification.
    Unpacks weights and uses standard MLX matmul.
    """
    weights = unpack_ternary_weights(packed_weights, in_features)
    weights = weights.astype(x.dtype)
    
    if x.ndim == 1:
        out = weights @ x
    else:
        out = x @ weights.T
    
    return out * scale
