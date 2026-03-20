"""
Unit tests for ternary matmul Metal kernels.
"""

import numpy as np
import mlx.core as mx
import sys
import os

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bitnet.kernels import pack_ternary_weights, unpack_ternary_weights, ternary_matmul


def test_pack_unpack_roundtrip():
    """Test that packing and unpacking preserves ternary weights."""
    print("Test: pack/unpack roundtrip... ", end="")
    
    rng = np.random.default_rng(42)
    
    for out_feat, in_feat in [(16, 16), (32, 48), (128, 256), (1, 32), (64, 17)]:
        w = rng.choice([-1, 0, 1], size=(out_feat, in_feat)).astype(np.float32)
        w_mx = mx.array(w)
        
        packed = pack_ternary_weights(w_mx)
        unpacked = unpack_ternary_weights(packed, in_feat)
        mx.eval(unpacked)
        
        unpacked_np = np.array(unpacked)
        assert np.array_equal(w, unpacked_np), \
            f"Roundtrip failed for shape ({out_feat}, {in_feat}):\n" \
            f"  Original: {w[:2, :8]}\n  Unpacked: {unpacked_np[:2, :8]}"
    
    print("✅ PASSED")


def test_ternary_matmul_correctness_1d():
    """Test Metal kernel produces correct results for 1D input."""
    print("Test: ternary matmul 1D correctness... ", end="")
    
    rng = np.random.default_rng(123)
    
    for out_feat, in_feat in [(32, 32), (64, 128), (256, 512), (16, 48)]:
        # Create ternary weights
        w = rng.choice([-1, 0, 1], size=(out_feat, in_feat)).astype(np.float32)
        w_mx = mx.array(w)
        packed = pack_ternary_weights(w_mx)
        
        # Create random activations
        x = mx.random.normal((in_feat,), dtype=mx.float16)
        scale = mx.array([0.75], dtype=mx.float32)
        
        # Metal kernel result
        out_gpu = ternary_matmul(packed, x, scale)
        mx.eval(out_gpu)
        
        # Reference: numpy matmul
        x_np = np.array(x, dtype=np.float32)
        expected = (w @ x_np) * 0.75
        
        out_np = np.array(out_gpu, dtype=np.float32)
        max_diff = np.max(np.abs(out_np - expected))
        
        assert max_diff < 0.5, \
            f"1D matmul mismatch for ({out_feat}, {in_feat}): max_diff={max_diff}"
    
    print("✅ PASSED")


def test_ternary_matmul_correctness_2d():
    """Test Metal kernel produces correct results for batched 2D input."""
    print("Test: ternary matmul 2D (batched) correctness... ", end="")
    
    rng = np.random.default_rng(456)
    
    for batch, out_feat, in_feat in [(2, 32, 64), (4, 128, 256), (1, 64, 64)]:
        w = rng.choice([-1, 0, 1], size=(out_feat, in_feat)).astype(np.float32)
        w_mx = mx.array(w)
        packed = pack_ternary_weights(w_mx)
        
        x = mx.random.normal((batch, in_feat), dtype=mx.float16)
        scale = mx.array([0.5], dtype=mx.float32)
        
        out_gpu = ternary_matmul(packed, x, scale)
        mx.eval(out_gpu)
        
        # Reference
        x_np = np.array(x, dtype=np.float32)
        expected = (x_np @ w.T) * 0.5
        
        out_np = np.array(out_gpu, dtype=np.float32)
        max_diff = np.max(np.abs(out_np - expected))
        
        assert max_diff < 0.5, \
            f"2D matmul mismatch for ({batch}, {out_feat}, {in_feat}): max_diff={max_diff}"
    
    print("✅ PASSED")


def test_all_zeros_weights():
    """Test with all-zero weights (should produce zero output)."""
    print("Test: all-zero weights... ", end="")
    
    w = np.zeros((32, 64), dtype=np.float32)
    packed = pack_ternary_weights(mx.array(w))
    x = mx.random.normal((64,), dtype=mx.float16)
    scale = mx.array([1.0], dtype=mx.float32)
    
    out = ternary_matmul(packed, x, scale)
    mx.eval(out)
    
    assert mx.allclose(out, mx.zeros_like(out), atol=1e-6).item(), "All-zero weights should give zero output"
    
    print("✅ PASSED")


def test_identity_like():
    """Test with +1 diagonal (identity-like behavior)."""
    print("Test: identity-like weights... ", end="")
    
    n = 32
    w = np.zeros((n, n), dtype=np.float32)
    np.fill_diagonal(w, 1.0)
    packed = pack_ternary_weights(mx.array(w))
    
    x = mx.array(np.arange(n, dtype=np.float32).astype(np.float16))
    scale = mx.array([1.0], dtype=mx.float32)
    
    out = ternary_matmul(packed, x, scale)
    mx.eval(out)
    
    max_diff = mx.max(mx.abs(out.astype(mx.float32) - x.astype(mx.float32))).item()
    assert max_diff < 0.01, f"Identity test failed: max_diff={max_diff}"
    
    print("✅ PASSED")


def test_metal_gpu_used():
    """Verify the kernel actually runs on GPU (not CPU fallback)."""
    print("Test: Metal GPU is being used... ", end="")
    
    assert mx.metal.is_available(), "Metal is not available!"
    assert str(mx.default_device()) == "Device(gpu, 0)", \
        f"Default device is {mx.default_device()}, expected GPU"
    
    print("✅ PASSED")


def test_large_matrix():
    """Test with realistic model-sized matrices."""
    print("Test: large matrix (4096x4096)... ", end="")
    
    rng = np.random.default_rng(789)
    w = rng.choice([-1, 0, 1], size=(4096, 4096)).astype(np.float32)
    packed = pack_ternary_weights(mx.array(w))
    
    x = mx.random.normal((4096,), dtype=mx.float16)
    scale = mx.array([0.01], dtype=mx.float32)
    
    out = ternary_matmul(packed, x, scale)
    mx.eval(out)
    
    assert out.shape == (4096,), f"Wrong output shape: {out.shape}"
    assert not mx.any(mx.isnan(out)).item(), "NaN in output!"
    assert not mx.any(mx.isinf(out)).item(), "Inf in output!"
    
    print("✅ PASSED")


if __name__ == "__main__":
    print(f"MLX {mx.__version__} | Metal: {mx.metal.is_available()} | Device: {mx.default_device()}")
    print("=" * 60)
    
    tests = [
        test_metal_gpu_used,
        test_pack_unpack_roundtrip,
        test_all_zeros_weights,
        test_identity_like,
        test_ternary_matmul_correctness_1d,
        test_ternary_matmul_correctness_2d,
        test_large_matrix,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"❌ FAILED: {e}")
            failed += 1
    
    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed, {len(tests)} total")
    
    if failed > 0:
        sys.exit(1)
