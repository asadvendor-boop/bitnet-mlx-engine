"""
BitNet neural network layers built on MLX.
"""

import math
import mlx.core as mx
import mlx.nn as nn

from .kernels import ternary_matmul, pack_ternary_weights


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    
    def __init__(self, dims: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps
    
    def __call__(self, x: mx.array) -> mx.array:
        dtype = x.dtype
        x = x.astype(mx.float32)
        rms = mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + self.eps)
        return (x * rms).astype(dtype) * self.weight


class BitLinear(nn.Module):
    """
    BitNet 1.58-bit linear layer.
    
    Replaces standard nn.Linear with ternary weights {-1, 0, +1}
    and uses the custom Metal GPU kernel for forward pass.
    
    Architecture:
        input -> RMSNorm -> ternary_matmul(packed_weights) -> scale -> output
    
    The weights are stored as 2-bit packed uint32 arrays.
    A per-layer scale factor compensates for the restricted weight range.
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Input normalization (critical for BitNet stability)
        self.input_norm = RMSNorm(in_features)
        
        # Packed ternary weights: (out_features, ceil(in_features/16)) uint32
        packed_cols = math.ceil(in_features / 16)
        self.packed_weights = mx.zeros((out_features, packed_cols), dtype=mx.uint32)
        
        # Per-layer scale factor (learned during training, frozen during inference)
        self.scale = mx.array([1.0], dtype=mx.float32)
        
        # Original in_features (needed for unpacking since we pad to multiple of 16)
        self._in_features = in_features
        
        # Optional bias (BitNet typically doesn't use bias)
        if bias:
            self.bias = mx.zeros((out_features,))
        else:
            self.bias = None
    
    def __call__(self, x: mx.array) -> mx.array:
        # Apply input RMSNorm
        x = self.input_norm(x)
        
        # Ternary matmul via Metal GPU kernel
        out = ternary_matmul(self.packed_weights, x, self.scale)
        
        if self.bias is not None:
            out = out + self.bias
        
        return out
    
    @staticmethod
    def from_float_weights(
        weights: mx.array,
        in_features: int,
        out_features: int,
        bias: mx.array = None,
    ) -> "BitLinear":
        """
        Create a BitLinear layer from float weights by quantizing to ternary.
        
        Quantization: w_ternary = round_ternary(w / mean(|w|))
        Scale: scale = mean(|w|)
        """
        layer = BitLinear(in_features, out_features, bias=bias is not None)
        
        # Compute scale (mean absolute value)
        w = weights.astype(mx.float32)
        scale = mx.mean(mx.abs(w))
        
        # Quantize to ternary
        w_normalized = w / (scale + 1e-8)
        w_ternary = mx.clip(mx.round(w_normalized), -1, 1)
        
        # Pack the ternary weights
        layer.packed_weights = pack_ternary_weights(w_ternary)
        layer.scale = scale.reshape(1)
        
        if bias is not None:
            layer.bias = bias
        
        return layer
