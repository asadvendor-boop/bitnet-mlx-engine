/*
 * BitNet-2B C++ Inference — Zero Python Overhead
 * Links against libmlx.dylib, uses MLX C++ API.
 */

#include <iostream>
#include <chrono>
#include <vector>
#include <string>
#include <optional>
#include <cmath>

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
    int group_size = 64;
    int bits = 2;
};

// ========== Quantized Linear Weights ==========
struct QuantizedWeights {
    mx::array weight;
    mx::array scales;
    mx::array biases;
    
    QuantizedWeights(mx::array w, mx::array s, mx::array b)
        : weight(std::move(w)), scales(std::move(s)), biases(std::move(b)) {}
};

// ========== Layer Weights ==========
struct LayerWeights {
    mx::array input_norm_w;
    mx::array post_attn_norm_w;
    mx::array attn_sub_norm_w;
    mx::array ffn_sub_norm_w;
    
    QuantizedWeights q, k, v, o;
    QuantizedWeights gate, up, down;
    
    LayerWeights(
        mx::array inw, mx::array panw, mx::array asnw, mx::array fsnw,
        QuantizedWeights q_, QuantizedWeights k_, QuantizedWeights v_, QuantizedWeights o_,
        QuantizedWeights g_, QuantizedWeights u_, QuantizedWeights d_
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

// ========== Helper Functions ==========

mx::array qlinear(const mx::array& x, const QuantizedWeights& qw, int gs, int bits) {
    return mx::quantized_matmul(x, qw.weight, qw.scales, qw.biases, true, gs, bits);
}

mx::array relu_squared(const mx::array& x) {
    auto r = mx::maximum(x, mx::array(0.0f));
    return r * r;
}

// ========== Model Forward ==========

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
        int total = L;
        if (cache[0].offset > 0) total += cache[0].offset;
        mask = mx::triu(mx::full({total, total}, -1e9f), 1);
        if (cache[0].offset > 0) {
            mask = mx::slice(*mask, {total - L, 0}, {total, total});
        }
    }

    for (int i = 0; i < cfg.num_layers; i++) {
        auto& lw = layers[i];
        auto residual = x;

        x = mx::fast::rms_norm(x, lw.input_norm_w, cfg.rms_norm_eps);

        auto q = qlinear(x, lw.q, cfg.group_size, cfg.bits);
        auto k = qlinear(x, lw.k, cfg.group_size, cfg.bits);
        auto v = qlinear(x, lw.v, cfg.group_size, cfg.bits);

        q = mx::reshape(q, {B, L, cfg.num_heads, cfg.head_dim});
        q = mx::transpose(q, {0, 2, 1, 3});
        k = mx::reshape(k, {B, L, cfg.num_kv_heads, cfg.head_dim});
        k = mx::transpose(k, {0, 2, 1, 3});
        v = mx::reshape(v, {B, L, cfg.num_kv_heads, cfg.head_dim});
        v = mx::transpose(v, {0, 2, 1, 3});

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
        auto attn_out = mx::fast::scaled_dot_product_attention(
            q, k, v, scale, mask_mode, mask);

        attn_out = mx::transpose(attn_out, {0, 2, 1, 3});
        attn_out = mx::reshape(attn_out, {B, L, cfg.hidden_size});
        attn_out = mx::fast::rms_norm(attn_out, lw.attn_sub_norm_w, cfg.rms_norm_eps);
        auto attn_proj = qlinear(attn_out, lw.o, cfg.group_size, cfg.bits);

        x = residual + attn_proj;
        residual = x;

        x = mx::fast::rms_norm(x, lw.post_attn_norm_w, cfg.rms_norm_eps);

        auto g = relu_squared(qlinear(x, lw.gate, cfg.group_size, cfg.bits));
        auto u = qlinear(x, lw.up, cfg.group_size, cfg.bits);
        auto hidden = mx::fast::rms_norm(g * u, lw.ffn_sub_norm_w, cfg.rms_norm_eps);
        auto d = qlinear(hidden, lw.down, cfg.group_size, cfg.bits);

        x = residual + d;
    }

    x = mx::fast::rms_norm(x, norm_w, cfg.rms_norm_eps);
    return mx::matmul(x, mx::transpose(embed_w));
}

int main(int argc, char* argv[]) {
    std::string model_path = "models/bitnet-2b-mlx-q2";
    int max_tokens = 200;
    if (argc > 1) model_path = argv[1];
    if (argc > 2) max_tokens = std::stoi(argv[2]);

    Config cfg;

    std::cout << "========================================\n";
    std::cout << "BitNet C++ Inference — Zero Python\n";
    std::cout << "========================================\n";

    std::cout << "Loading weights from " << model_path << "...\n";
    auto loaded = mx::load_safetensors(model_path + "/model.safetensors");
    auto& wmap = loaded.first;

    auto embed_w = wmap.at("model.embed_tokens.weight");
    auto norm_w = wmap.at("model.norm.weight");

    std::vector<LayerWeights> layers;
    for (int i = 0; i < cfg.num_layers; i++) {
        if (i % 10 == 0) std::cout << "  Loading layer " << i << "...\n";
        std::string p = "model.layers." + std::to_string(i);

        auto load_q = [&](const std::string& name) -> QuantizedWeights {
            return QuantizedWeights(
                wmap.at(p + "." + name + ".weight"),
                wmap.at(p + "." + name + ".scales"),
                wmap.at(p + "." + name + ".biases")
            );
        };

        layers.emplace_back(
            wmap.at(p + ".input_layernorm.weight"),
            wmap.at(p + ".post_attention_layernorm.weight"),
            wmap.at(p + ".self_attn.attn_sub_norm.weight"),
            wmap.at(p + ".mlp.ffn_sub_norm.weight"),
            load_q("self_attn.q_proj"),
            load_q("self_attn.k_proj"),
            load_q("self_attn.v_proj"),
            load_q("self_attn.o_proj"),
            load_q("mlp.gate_proj"),
            load_q("mlp.up_proj"),
            load_q("mlp.down_proj")
        );
    }
    std::cout << "Model loaded!\n";

    // Prompt: "Once upon a time"
    std::vector<int> prompt_tokens = {128000, 12805, 5304, 264, 892};
    int eos_token = 128009;

    // Init KV cache
    std::vector<KVEntry> kv_cache(cfg.num_layers);

    // Prefill
    std::cout << "Prefilling " << prompt_tokens.size() << " tokens...\n";
    auto input_ids = mx::array(
        prompt_tokens.data(),
        {1, (int)prompt_tokens.size()},
        mx::int32
    );

    auto t0 = std::chrono::high_resolution_clock::now();
    auto logits = model_forward(input_ids, embed_w, norm_w, layers, cfg, kv_cache);
    mx::eval(logits);
    for (auto& c : kv_cache) {
        if (c.keys.has_value()) mx::eval(*c.keys, *c.values);
        c.offset = (int)prompt_tokens.size();
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double prefill_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "Prefill: " << prefill_ms << " ms\n";

    // Decode loop — PURE C++
    std::cout << "Generating " << max_tokens << " tokens...\n";
    std::vector<int> generated;

    auto gen_start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < max_tokens; step++) {
        // Argmax over last position
        int S = logits.shape(1);
        auto last = mx::slice(logits, {0, S-1, 0}, {1, S, logits.shape(2)});
        last = mx::reshape(last, {1, -1});
        auto next_token = mx::argmax(last, 1);
        mx::eval(next_token);  // Only eval the token, lazy-eval everything else

        int token_id = next_token.item<int>();
        if (token_id == eos_token) break;
        generated.push_back(token_id);

        // Next step — no explicit cache eval, let MLX handle it lazily
        auto next_input = mx::reshape(next_token, {1, 1});
        next_input = mx::astype(next_input, mx::int32);
        logits = model_forward(next_input, embed_w, norm_w, layers, cfg, kv_cache);
        // Don't eval logits explicitly — it gets eval'd when we call argmax + eval(next_token)
        for (auto& c : kv_cache) c.offset++;
    }

    auto gen_end = std::chrono::high_resolution_clock::now();
    double gen_s = std::chrono::duration<double>(gen_end - gen_start).count();
    double tps = generated.size() / gen_s;

    std::cout << "========================================\n";
    std::cout << "Generated " << generated.size() << " tokens in "
              << gen_s << "s\n";
    std::cout << "Speed: " << tps << " tok/s\n";
    std::cout << "Token IDs: ";
    for (int i = 0; i < std::min((int)generated.size(), 20); i++) {
        std::cout << generated[i] << " ";
    }
    std::cout << "...\n";

    return 0;
}
