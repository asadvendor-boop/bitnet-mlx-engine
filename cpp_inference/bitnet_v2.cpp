/*
 * BitNet C++ Inference v2 — Replicating mlx-lm's BitLinear kernel
 * 
 * Uses the SAME Metal kernel as mlx-lm's bitlinear_layers.py:
 * - Original uint8 packed weights (no conversion)
 * - simd_sum for fast reductions
 * - Direct ternary decode: (w & 3) - 1
 */

#include <iostream>
#include <chrono>
#include <vector>
#include <string>
#include <optional>
#include <cmath>
#include <functional>

#include "mlx/mlx.h"

namespace mx = mlx::core;

// ========== Config ==========
struct Config {
    int hidden_size = 2560;
    int num_heads = 20;
    int num_kv_heads = 5;
    int num_layers = 30;
    int intermediate_size = 6912;
    int vocab_size = 152064;
    float rope_theta = 500000.0f;
    float rms_norm_eps = 1e-5f;
    int head_dim = 128;
};

// ========== BitLinear Kernel (same as mlx-lm) ==========

static const char* BITLINEAR_SOURCE = R"(
    constexpr int M = 4;
    constexpr int BLOCK = 32;

    uint tid = thread_position_in_grid.y;
    uint in_offset = thread_position_in_grid.x;

    uint batch_idx = tid / (out_features / 4);
    uint row_idx = tid % (out_features / 4);

    float sum[4] = {0.0};

    for (uint i = in_offset * M; i < in_features; i += BLOCK * M) {
        float v[M];
        for (int j=0; j<M; j++) {
            v[j] = x[batch_idx * in_features + i + j];
        }

        for (int j=0; j<M; j++) {
            uint8_t w = packed_weights[row_idx * in_features + i + j];
            sum[0] += v[j] * ((w & 3) - 1);
            sum[1] += v[j] * (((w >> 2) & 3) - 1);
            sum[2] += v[j] * (((w >> 4) & 3) - 1);
            sum[3] += v[j] * (((w >> 6) & 3) - 1);
        }
    }

    for (int j=0; j<4; j++) {
        sum[j] = simd_sum(sum[j]);
    }

    if (in_offset == 0) {
        float scale = invert_weight_scales ? 1 / weight_scale[0] : weight_scale[0];
        for (int i=0; i<4; i++) {
            out[batch_idx * out_features + row_idx + i * (out_features/4)] = static_cast<T>(sum[i] * scale);
        }
    }
)";

mx::array bitlinear_matmul(
    const mx::array& x, const mx::array& packed_weights,
    const mx::array& weight_scale, int in_features, int out_features
) {
    // Flatten batch dimensions
    auto orig_shape = x.shape();
    mx::array flat_x = x;
    if (orig_shape.size() > 2) {
        int batch = 1;
        for (size_t i = 0; i < orig_shape.size() - 1; i++) batch *= orig_shape[i];
        flat_x = mx::reshape(x, {batch, in_features});
    }
    int total_batch = flat_x.shape(0);

    auto dtype = weight_scale.dtype();
    flat_x = mx::astype(flat_x, dtype);

    static auto kernel = mx::fast::metal_kernel(
        "bitlinear_matmul",
        {"x", "packed_weights", "weight_scale"},
        {"out"},
        BITLINEAR_SOURCE
    );

    auto result = kernel(
        {flat_x, packed_weights, weight_scale},        // inputs
        {{total_batch, out_features}},                  // output_shapes
        {dtype},                                        // output_dtypes
        std::make_tuple(32, total_batch * out_features / 4, 1),  // grid
        std::make_tuple(32, 1, 1),                      // threadgroup
        {                                               // template args
            {"T", dtype},
            {"invert_weight_scales", false},
            {"in_features", in_features},
            {"out_features", out_features}
        },
        std::nullopt,                                   // init_value
        false,                                          // verbose
        mx::Device::gpu                                 // stream/device
    );

    auto out = result[0];
    if (orig_shape.size() > 2) {
        mx::Shape new_shape(orig_shape.begin(), orig_shape.end() - 1);
        new_shape.push_back(out_features);
        out = mx::reshape(out, new_shape);
    }
    return out;
}

// ========== Layer Weights (original uint8 format) ==========

struct BitLinearWeights {
    mx::array weight;       // (out/4, in) uint8
    mx::array weight_scale; // (1,) float
    int in_features;
    int out_features;

    BitLinearWeights(mx::array w, mx::array s, int inf, int outf)
        : weight(std::move(w)), weight_scale(std::move(s)),
          in_features(inf), out_features(outf) {}
};

struct LayerWeights {
    mx::array input_norm_w, post_attn_norm_w, attn_sub_norm_w, ffn_sub_norm_w;

    BitLinearWeights q, k, v, o;
    BitLinearWeights gate, up, down;

    LayerWeights(
        mx::array inw, mx::array panw, mx::array asnw, mx::array fsnw,
        BitLinearWeights q_, BitLinearWeights k_, BitLinearWeights v_, BitLinearWeights o_,
        BitLinearWeights g_, BitLinearWeights u_, BitLinearWeights d_
    ) : input_norm_w(std::move(inw)), post_attn_norm_w(std::move(panw)),
        attn_sub_norm_w(std::move(asnw)), ffn_sub_norm_w(std::move(fsnw)),
        q(std::move(q_)), k(std::move(k_)), v(std::move(v_)), o(std::move(o_)),
        gate(std::move(g_)), up(std::move(u_)), down(std::move(d_)) {}
};

// ========== KV Cache ==========
struct KVEntry {
    std::optional<mx::array> keys;
    std::optional<mx::array> values;
    int offset = 0;
};

// ========== Forward Pass ==========

mx::array bl_forward(const mx::array& x, const BitLinearWeights& w) {
    return bitlinear_matmul(x, w.weight, w.weight_scale, w.in_features, w.out_features);
}

mx::array model_forward(
    const mx::array& input_ids,
    const mx::array& embed_w,
    const mx::array& norm_w,
    const std::vector<LayerWeights>& layers,
    const Config& cfg,
    std::vector<KVEntry>& cache
) {
    int B = input_ids.shape(0);
    int L = input_ids.shape(1);

    auto x = mx::take(embed_w, input_ids, 0);

    std::optional<mx::array> mask;
    std::string mask_mode = "";
    if (L > 1) {
        int total = L + cache[0].offset;
        auto m = mx::triu(mx::full({total, total}, -1e9f), 1);
        m = mx::astype(m, mx::bfloat16);
        if (cache[0].offset > 0) {
            m = mx::slice(m, {total - L, 0}, {total, total});
        }
        mask = m;
    }

    for (int i = 0; i < cfg.num_layers; i++) {
        auto& lw = layers[i];
        auto residual = x;

        x = mx::fast::rms_norm(x, lw.input_norm_w, cfg.rms_norm_eps);

        auto q = bl_forward(x, lw.q);
        auto k = bl_forward(x, lw.k);
        auto v = bl_forward(x, lw.v);

        q = mx::transpose(mx::reshape(q, {B, L, cfg.num_heads, cfg.head_dim}), {0, 2, 1, 3});
        k = mx::transpose(mx::reshape(k, {B, L, cfg.num_kv_heads, cfg.head_dim}), {0, 2, 1, 3});
        v = mx::transpose(mx::reshape(v, {B, L, cfg.num_kv_heads, cfg.head_dim}), {0, 2, 1, 3});

        int offset = cache[i].offset;
        q = mx::fast::rope(q, cfg.head_dim, false, cfg.rope_theta, 1.0f, offset);
        k = mx::fast::rope(k, cfg.head_dim, false, cfg.rope_theta, 1.0f, offset);

        if (cache[i].keys.has_value()) {
            k = mx::concatenate({*cache[i].keys, k}, 2);
            v = mx::concatenate({*cache[i].values, v}, 2);
        }
        cache[i].keys = k;
        cache[i].values = v;

        float scale = 1.0f / std::sqrt((float)cfg.head_dim);
        auto attn_out = mx::fast::scaled_dot_product_attention(q, k, v, scale, mask_mode, mask);

        attn_out = mx::reshape(mx::transpose(attn_out, {0, 2, 1, 3}), {B, L, cfg.hidden_size});
        attn_out = mx::fast::rms_norm(attn_out, lw.attn_sub_norm_w, cfg.rms_norm_eps);
        auto attn_proj = bl_forward(attn_out, lw.o);

        x = residual + attn_proj;
        residual = x;

        x = mx::fast::rms_norm(x, lw.post_attn_norm_w, cfg.rms_norm_eps);

        // MLP: relu2(gate) * up -> sub_norm -> down
        auto g = bl_forward(x, lw.gate);
        g = mx::maximum(g, mx::array(0.0f));
        g = g * g; // relu2
        auto u = bl_forward(x, lw.up);
        auto hidden = mx::fast::rms_norm(g * u, lw.ffn_sub_norm_w, cfg.rms_norm_eps);
        x = residual + bl_forward(hidden, lw.down);
    }

    x = mx::fast::rms_norm(x, norm_w, cfg.rms_norm_eps);
    return mx::matmul(x, mx::transpose(embed_w));
}

int main(int argc, char* argv[]) {
    std::string model_path = "models/bitnet-2b";
    int max_tokens = 200;
    if (argc > 1) model_path = argv[1];
    if (argc > 2) max_tokens = std::stoi(argv[2]);

    Config cfg;

    std::cout << "========================================\n";
    std::cout << "BitNet C++ v2 — mlx-lm Kernel Replica\n";
    std::cout << "========================================\n";

    std::cout << "Loading " << model_path << "...\n";
    auto loaded = mx::load_safetensors(model_path + "/model.safetensors");
    auto& wmap = loaded.first;

    auto embed_w = wmap.at("model.embed_tokens.weight");
    auto norm_w = wmap.at("model.norm.weight");

    // Convert embed to bfloat16 (mlx-lm default)
    embed_w = mx::astype(embed_w, mx::bfloat16);
    norm_w = mx::astype(norm_w, mx::bfloat16);

    std::vector<LayerWeights> layers;
    for (int i = 0; i < cfg.num_layers; i++) {
        if (i % 10 == 0) std::cout << "  Layer " << i << "...\n";
        std::string p = "model.layers." + std::to_string(i);

        auto get_bl = [&](const std::string& name, int inf, int outf) -> BitLinearWeights {
            auto w = wmap.at(p + "." + name + ".weight");
            auto s = mx::astype(wmap.at(p + "." + name + ".weight_scale"), mx::bfloat16);
            return BitLinearWeights(w, s, inf, outf);
        };

        int hs = cfg.hidden_size;
        int is = cfg.intermediate_size;
        int kv_dim = cfg.num_kv_heads * cfg.head_dim;
        int q_dim = cfg.num_heads * cfg.head_dim;

        layers.emplace_back(
            mx::astype(wmap.at(p + ".input_layernorm.weight"), mx::bfloat16),
            mx::astype(wmap.at(p + ".post_attention_layernorm.weight"), mx::bfloat16),
            mx::astype(wmap.at(p + ".self_attn.attn_sub_norm.weight"), mx::bfloat16),
            mx::astype(wmap.at(p + ".mlp.ffn_sub_norm.weight"), mx::bfloat16),
            get_bl("self_attn.q_proj", hs, q_dim),
            get_bl("self_attn.k_proj", hs, kv_dim),
            get_bl("self_attn.v_proj", hs, kv_dim),
            get_bl("self_attn.o_proj", q_dim, hs),
            get_bl("mlp.gate_proj", hs, is),
            get_bl("mlp.up_proj", hs, is),
            get_bl("mlp.down_proj", is, hs)
        );
    }
    std::cout << "Loaded!\n";

    // Prompt: "Once upon a time"
    std::vector<int> prompt_tokens = {128000, 12805, 5304, 264, 892};
    int eos_token = 128009;

    std::vector<KVEntry> kv_cache(cfg.num_layers);

    // Prefill
    auto input_ids = mx::array(prompt_tokens.data(), {1, (int)prompt_tokens.size()}, mx::int32);

    auto t0 = std::chrono::high_resolution_clock::now();
    auto logits = model_forward(input_ids, embed_w, norm_w, layers, cfg, kv_cache);
    mx::eval(logits);
    for (auto& c : kv_cache) c.offset = (int)prompt_tokens.size();
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "Prefill: " << std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms\n";

    // Decode
    std::cout << "Generating " << max_tokens << " tokens...\n";
    std::vector<int> generated;
    auto gen_start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < max_tokens; step++) {
        int S = logits.shape(1);
        auto last = mx::reshape(mx::slice(logits, {0, S-1, 0}, {1, S, logits.shape(2)}), {1, -1});
        auto next_token = mx::argmax(last, 1);
        mx::eval(next_token);

        int tid = next_token.item<int>();
        if (tid == eos_token) break;
        generated.push_back(tid);

        auto next_in = mx::astype(mx::reshape(next_token, {1, 1}), mx::int32);
        logits = model_forward(next_in, embed_w, norm_w, layers, cfg, kv_cache);
        for (auto& c : kv_cache) c.offset++;
    }

    auto gen_end = std::chrono::high_resolution_clock::now();
    double gen_s = std::chrono::duration<double>(gen_end - gen_start).count();
    double tps = generated.size() / gen_s;

    std::cout << "========================================\n";
    std::cout << "Generated " << generated.size() << " tokens in " << gen_s << "s\n";
    std::cout << "Speed: " << tps << " tok/s\n";
    std::cout << "IDs: ";
    for (int i = 0; i < std::min((int)generated.size(), 20); i++) std::cout << generated[i] << " ";
    std::cout << "...\n";

    return 0;
}
