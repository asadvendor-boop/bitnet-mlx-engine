# Detect MLX paths from pip
MLX_INCLUDE := $(shell python3 -c "import mlx; import os; print(os.path.join(os.path.dirname(mlx.__file__), 'include'))" 2>/dev/null)
MLX_LIB := $(shell python3 -c "import mlx; import os; print(os.path.join(os.path.dirname(mlx.__file__), 'lib'))" 2>/dev/null)

CXX := clang++
CXXFLAGS := -std=c++17 -O3 -I$(MLX_INCLUDE)
LDFLAGS := -L$(MLX_LIB) -lmlx -Wl,-rpath,$(MLX_LIB) -framework Metal -framework Foundation -framework Accelerate

.PHONY: all clean help

all: cpp_inference/bitnet_v3 cpp_inference/bitnet_v5

cpp_inference/bitnet_v3: cpp_inference/bitnet_v3.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)
	@echo "✅ Built bitnet_v3 (baseline engine)"

cpp_inference/bitnet_v5: cpp_inference/bitnet_v5_speculative.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)
	@echo "✅ Built bitnet_v5 (speculative decode engine)"

clean:
	rm -f cpp_inference/bitnet_v3 cpp_inference/bitnet_v5 cpp_inference/bitnet_v2 cpp_inference/bitnet_v4 cpp_inference/bitnet_generate

help:
	@echo "BitNet MLX Engine — Build Targets"
	@echo ""
	@echo "  make all        Build both engines"
	@echo "  make clean      Remove compiled binaries"
	@echo ""
	@echo "Prerequisites: pip install mlx mlx-lm"
	@echo "Model: huggingface-cli download 1bitLLM/bitnet_b1_58-2B --local-dir models/bitnet-2b"
