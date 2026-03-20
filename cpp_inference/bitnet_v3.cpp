/*
 * BitNet C++ v3 — Maximum Speed
 * 
 * Optimizations over v2:
 * 1. Pre-allocated KV cache (256-step chunks, slice assignment)
 * 2. "causal" string mask mode (no explicit mask matrix)
 * 3. Minimal eval calls
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

// ========== BitLinear Metal Kernel ==========
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
        for (int j=0; j<M; j++) v[j] = x[batch_idx * in_features + i + j];
        for (int j=0; j<M; j++) {
            uint8_t w = packed_weights[row_idx * in_features + i + j];
            sum[0] += v[j] * ((w & 3) - 1);
            sum[1] += v[j] * (((w >> 2) & 3) - 1);
            sum[2] += v[j] * (((w >> 4) & 3) - 1);
            sum[3] += v[j] * (((w >> 6) & 3) - 1);
        }
    }
    for (int j=0; j<4; j++) sum[j] = simd_sum(sum[j]);
    if (in_offset == 0) {
        float scale = invert_weight_scales ? 1 / weight_scale[0] : weight_scale[0];
        for (int i=0; i<4; i++)
            out[batch_idx * out_features + row_idx + i * (out_features/4)] = static_cast<T>(sum[i] * scale);
    }
)";

mx::array bitlinear_matmul(
    const mx::array& x, const mx::array& packed_weights,
    const mx::array& weight_scale, int in_features, int out_features
) {
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
        "bitlinear_matmul", {"x", "packed_weights", "weight_scale"}, {"out"}, BITLINEAR_SOURCE);

    auto result = kernel(
        {flat_x, packed_weights, weight_scale},
        {{total_batch, out_features}}, {dtype},
        std::make_tuple(32, total_batch * out_features / 4, 1),
        std::make_tuple(32, 1, 1),
        {{"T", dtype}, {"invert_weight_scales", false},
         {"in_features", in_features}, {"out_features", out_features}},
        std::nullopt, false, mx::Device::gpu);

    auto out = result[0];
    if (orig_shape.size() > 2) {
        mx::Shape new_shape(orig_shape.begin(), orig_shape.end() - 1);
        new_shape.push_back(out_features);
        out = mx::reshape(out, new_shape);
    }
    return out;
}

// ========== Weights ==========
struct BitLinearWeights {
    mx::array weight; mx::array weight_scale;
    int in_features, out_features;
    BitLinearWeights(mx::array w, mx::array s, int inf, int outf)
        : weight(std::move(w)), weight_scale(std::move(s)), in_features(inf), out_features(outf) {}
};

struct LayerWeights {
    mx::array input_norm_w, post_attn_norm_w, attn_sub_norm_w, ffn_sub_norm_w;
    BitLinearWeights q, k, v, o, gate, up, down;
    LayerWeights(mx::array inw, mx::array panw, mx::array asnw, mx::array fsnw,
        BitLinearWeights q_, BitLinearWeights k_, BitLinearWeights v_, BitLinearWeights o_,
        BitLinearWeights g_, BitLinearWeights u_, BitLinearWeights d_)
        : input_norm_w(std::move(inw)), post_attn_norm_w(std::move(panw)),
          attn_sub_norm_w(std::move(asnw)), ffn_sub_norm_w(std::move(fsnw)),
          q(std::move(q_)), k(std::move(k_)), v(std::move(v_)), o(std::move(o_)),
          gate(std::move(g_)), up(std::move(u_)), down(std::move(d_)) {}
};

// ========== Pre-allocated KV Cache (like mlx-lm) ==========
struct KVCache {
    std::optional<mx::array> keys;
    std::optional<mx::array> values;
    int offset = 0;
    static const int STEP = 256;

    void update_and_fetch(mx::array& new_keys, mx::array& new_values) {
        int prev = offset;
        int n_new = new_keys.shape(2);

        if (!keys.has_value() || (prev + n_new) > keys->shape(2)) {
            int B = new_keys.shape(0);
            int n_kv_heads = new_keys.shape(1);
            int k_dim = new_keys.shape(3);
            int v_dim = new_values.shape(3);
            int n_steps = (STEP + n_new - 1) / STEP;

            auto new_k = mx::zeros({B, n_kv_heads, n_steps * STEP, k_dim}, new_keys.dtype());
            auto new_v = mx::zeros({B, n_kv_heads, n_steps * STEP, v_dim}, new_values.dtype());

            if (keys.has_value()) {
                if (prev % STEP != 0) {
                    keys = mx::slice(*keys, {0,0,0,0}, {B, n_kv_heads, prev, k_dim});
                    values = mx::slice(*values, {0,0,0,0}, {B, n_kv_heads, prev, v_dim});
                }
                keys = mx::concatenate({*keys, new_k}, 2);
                values = mx::concatenate({*values, new_v}, 2);
            } else {
                keys = new_k;
                values = new_v;
            }
        }

        offset += n_new;
        // Slice assignment: keys[..., prev:offset, :] = new_keys
        keys = mx::slice_update(*keys, new_keys, {0, 0, prev, 0}, {keys->shape(0), keys->shape(1), offset, keys->shape(3)});
        values = mx::slice_update(*values, new_values, {0, 0, prev, 0}, {values->shape(0), values->shape(1), offset, values->shape(3)});

        // Return used portion
        new_keys = mx::slice(*keys, {0,0,0,0}, {keys->shape(0), keys->shape(1), offset, keys->shape(3)});
        new_values = mx::slice(*values, {0,0,0,0}, {values->shape(0), values->shape(1), offset, values->shape(3)});
    }
};

// ========== Forward ==========
mx::array bl_fw(const mx::array& x, const BitLinearWeights& w) {
    return bitlinear_matmul(x, w.weight, w.weight_scale, w.in_features, w.out_features);
}

mx::array model_forward(
    const mx::array& input_ids,
    const mx::array& embed_w, const mx::array& norm_w,
    const std::vector<LayerWeights>& layers, const Config& cfg,
    std::vector<KVCache>& cache
) {
    int B = input_ids.shape(0);
    int L = input_ids.shape(1);
    auto x = mx::take(embed_w, input_ids, 0);

    // Use "causal" string mask for prefill, no mask for decode (same as mlx-lm)
    std::optional<mx::array> mask_arr;
    std::string mask_mode = "";
    if (L > 1) {
        mask_mode = "causal";  // Hardware-accelerated causal masking
    }

    for (int i = 0; i < cfg.num_layers; i++) {
        auto& lw = layers[i];
        auto residual = x;

        x = mx::fast::rms_norm(x, lw.input_norm_w, cfg.rms_norm_eps);

        auto q = bl_fw(x, lw.q);
        auto k = bl_fw(x, lw.k);
        auto v = bl_fw(x, lw.v);

        q = mx::transpose(mx::reshape(q, {B, L, cfg.num_heads, cfg.head_dim}), {0, 2, 1, 3});
        k = mx::transpose(mx::reshape(k, {B, L, cfg.num_kv_heads, cfg.head_dim}), {0, 2, 1, 3});
        v = mx::transpose(mx::reshape(v, {B, L, cfg.num_kv_heads, cfg.head_dim}), {0, 2, 1, 3});

        int offset = cache[i].offset;
        q = mx::fast::rope(q, cfg.head_dim, false, cfg.rope_theta, 1.0f, offset);
        k = mx::fast::rope(k, cfg.head_dim, false, cfg.rope_theta, 1.0f, offset);

        // Pre-allocated KV cache update (like mlx-lm)
        cache[i].update_and_fetch(k, v);

        float scale = 1.0f / std::sqrt((float)cfg.head_dim);
        auto attn_out = mx::fast::scaled_dot_product_attention(
            q, k, v, scale, mask_mode, mask_arr);

        attn_out = mx::reshape(mx::transpose(attn_out, {0, 2, 1, 3}), {B, L, cfg.hidden_size});
        attn_out = mx::fast::rms_norm(attn_out, lw.attn_sub_norm_w, cfg.rms_norm_eps);
        x = residual + bl_fw(attn_out, lw.o);
        residual = x;

        x = mx::fast::rms_norm(x, lw.post_attn_norm_w, cfg.rms_norm_eps);
        auto g = bl_fw(x, lw.gate);
        g = mx::maximum(g, mx::array(0.0f)); g = g * g;
        x = residual + bl_fw(mx::fast::rms_norm(g * bl_fw(x, lw.up), lw.ffn_sub_norm_w, cfg.rms_norm_eps), lw.down);
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
    mx::set_memory_limit(8ULL * 1024 * 1024 * 1024);
    mx::set_cache_limit(4ULL * 1024 * 1024 * 1024);

    std::cout << "========================================\n";
    std::cout << "BitNet C++ v3 — Maximum Speed\n";
    std::cout << "========================================\n";

    auto loaded = mx::load_safetensors(model_path + "/model.safetensors");
    auto& wmap = loaded.first;
    auto embed_w = mx::astype(wmap.at("model.embed_tokens.weight"), mx::bfloat16);
    auto norm_w = mx::astype(wmap.at("model.norm.weight"), mx::bfloat16);

    std::vector<LayerWeights> layers;
    for (int i = 0; i < cfg.num_layers; i++) {
        std::string p = "model.layers." + std::to_string(i);
        auto get_bl = [&](const std::string& name, int inf, int outf) -> BitLinearWeights {
            return BitLinearWeights(
                wmap.at(p + "." + name + ".weight"),
                mx::astype(wmap.at(p + "." + name + ".weight_scale"), mx::bfloat16), inf, outf);
        };
        int hs = cfg.hidden_size, is = cfg.intermediate_size;
        int kv = cfg.num_kv_heads * cfg.head_dim, qd = cfg.num_heads * cfg.head_dim;
        layers.emplace_back(
            mx::astype(wmap.at(p + ".input_layernorm.weight"), mx::bfloat16),
            mx::astype(wmap.at(p + ".post_attention_layernorm.weight"), mx::bfloat16),
            mx::astype(wmap.at(p + ".self_attn.attn_sub_norm.weight"), mx::bfloat16),
            mx::astype(wmap.at(p + ".mlp.ffn_sub_norm.weight"), mx::bfloat16),
            get_bl("self_attn.q_proj", hs, qd), get_bl("self_attn.k_proj", hs, kv),
            get_bl("self_attn.v_proj", hs, kv), get_bl("self_attn.o_proj", qd, hs),
            get_bl("mlp.gate_proj", hs, is), get_bl("mlp.up_proj", hs, is),
            get_bl("mlp.down_proj", is, hs));
    }
    std::cout << "Model loaded.\n";

    std::vector<int> prompt_tokens = {128000, 12805, 5304, 264, 892};
    int eos_token = 128009;
    std::vector<KVCache> kv_cache(cfg.num_layers);

    // Warmup
    {
        std::vector<KVCache> warm_cache(cfg.num_layers);
        auto warm_in = mx::array(prompt_tokens.data(), {1, (int)prompt_tokens.size()}, mx::int32);
        auto warm_logits = model_forward(warm_in, embed_w, norm_w, layers, cfg, warm_cache);
        auto warm_tok = mx::argmax(mx::reshape(mx::slice(warm_logits, {0, (int)prompt_tokens.size()-1, 0},
            {1, (int)prompt_tokens.size(), warm_logits.shape(2)}), {1, -1}), 1);
        mx::eval(warm_tok);
        // Decode a few
        for (int j = 0; j < 3; j++) {
            auto nin = mx::astype(mx::reshape(warm_tok, {1, 1}), mx::int32);
            warm_logits = model_forward(nin, embed_w, norm_w, layers, cfg, warm_cache);
            warm_tok = mx::argmax(mx::reshape(mx::slice(warm_logits, {0, 0, 0}, {1, 1, warm_logits.shape(2)}), {1, -1}), 1);
            mx::eval(warm_tok);
        }
    }
    std::cout << "Warmed up.\n";

    // Prefill
    auto input_ids = mx::array(prompt_tokens.data(), {1, (int)prompt_tokens.size()}, mx::int32);
    auto t0 = std::chrono::high_resolution_clock::now();
    auto logits = model_forward(input_ids, embed_w, norm_w, layers, cfg, kv_cache);
    auto t1 = std::chrono::high_resolution_clock::now();

    // Create dedicated generation stream (like mlx-lm)
    auto gen_stream = mx::new_stream(mx::default_device());

    // Decode — async_eval pipelining (same as mlx-lm generate_step)
    std::vector<int> generated;
    auto gen_start = std::chrono::high_resolution_clock::now();

    // Initial: slice last logits → argmax
    auto last_logits = mx::slice(logits, {0, logits.shape(1)-1, 0}, {1, logits.shape(1), logits.shape(2)});
    last_logits = mx::reshape(last_logits, {1, -1});
    auto y = mx::argmax(last_logits, 1);

    for (int step = 0; step < max_tokens; step++) {
        // Build next step's compute graph BEFORE eval'ing current token
        auto next_in = mx::astype(mx::reshape(y, {1, 1}), mx::int32);
        logits = model_forward(next_in, embed_w, norm_w, layers, cfg, kv_cache);
        auto next_logits = mx::reshape(mx::slice(logits, {0, 0, 0}, {1, 1, logits.shape(2)}), {1, -1});
        auto next_y = mx::argmax(next_logits, 1);

        // Async eval: GPU starts computing next_y while we process current y
        mx::async_eval(next_y);

        // Now sync on current token (GPU already started next computation)
        int tid = y.item<int>();
        if (tid == eos_token) break;
        generated.push_back(tid);

        // Swap for next iteration
        y = next_y;
    }

    auto gen_end = std::chrono::high_resolution_clock::now();
    double prefill_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double gen_s = std::chrono::duration<double>(gen_end - gen_start).count();
    double tps = generated.size() / gen_s;

    std::cout << "========================================\n";
    std::cout << "Prefill: " << prefill_ms << " ms\n";
    std::cout << "Generated " << generated.size() << " tokens in " << gen_s << "s\n";
    std::cout << "Speed: " << tps << " tok/s\n";
    std::cout << "IDs: ";
    for (int i = 0; i < std::min((int)generated.size(), 20); i++) std::cout << generated[i] << " ";
    std::cout << "...\n";

    return 0;
}
