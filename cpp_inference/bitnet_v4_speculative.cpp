/*
 * BitNet C++ v4 — Speculative Decoding
 * 
 * Uses the full 30-layer model + a "draft" model (first 4 layers + lm_head)
 * to speculate N tokens ahead. Full model verifies in one batched pass.
 *
 * Expected speedup: if draft acceptance rate = p, and draft_speed = S_draft:
 *   effective_speed ≈ N * p * full_speed (since verification is batched)
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
    mx::array weight, weight_scale;
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

// ========== KV Cache ==========
struct KVCache {
    std::optional<mx::array> keys, values;
    int offset = 0;

    void update(mx::array& k, mx::array& v) {
        if (keys.has_value()) {
            k = mx::concatenate({*keys, k}, 2);
            v = mx::concatenate({*values, v}, 2);
        }
        keys = k; values = v;
        offset += 1;  // Updated externally
    }
    
    void rollback(int n) {
        // Remove last n entries from cache
        if (keys.has_value() && offset > n) {
            int new_len = keys->shape(2) - n;
            keys = mx::slice(*keys, {0,0,0,0}, {keys->shape(0), keys->shape(1), new_len, keys->shape(3)});
            values = mx::slice(*values, {0,0,0,0}, {values->shape(0), values->shape(1), new_len, values->shape(3)});
            offset -= n;
        }
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
    std::vector<KVCache>& cache,
    int num_layers = -1  // Use subset for draft model
) {
    if (num_layers < 0) num_layers = cfg.num_layers;
    int B = input_ids.shape(0);
    int L = input_ids.shape(1);
    auto x = mx::take(embed_w, input_ids, 0);

    std::optional<mx::array> mask_arr;
    std::string mask_mode = "";
    if (L > 1) mask_mode = "causal";

    for (int i = 0; i < num_layers; i++) {
        auto& lw = layers[i];
        auto residual = x;
        x = mx::fast::rms_norm(x, lw.input_norm_w, cfg.rms_norm_eps);

        auto q = bl_fw(x, lw.q);
        auto k = bl_fw(x, lw.k);
        auto v = bl_fw(x, lw.v);

        q = mx::transpose(mx::reshape(q, {B, L, cfg.num_heads, cfg.head_dim}), {0, 2, 1, 3});
        k = mx::transpose(mx::reshape(k, {B, L, cfg.num_kv_heads, cfg.head_dim}), {0, 2, 1, 3});
        v = mx::transpose(mx::reshape(v, {B, L, cfg.num_kv_heads, cfg.head_dim}), {0, 2, 1, 3});

        int off = cache[i].offset;
        q = mx::fast::rope(q, cfg.head_dim, false, cfg.rope_theta, 1.0f, off);
        k = mx::fast::rope(k, cfg.head_dim, false, cfg.rope_theta, 1.0f, off);
        cache[i].update(k, v);

        float scale = 1.0f / std::sqrt((float)cfg.head_dim);
        auto attn = mx::fast::scaled_dot_product_attention(q, k, v, scale, mask_mode, mask_arr);
        attn = mx::reshape(mx::transpose(attn, {0, 2, 1, 3}), {B, L, cfg.hidden_size});
        attn = mx::fast::rms_norm(attn, lw.attn_sub_norm_w, cfg.rms_norm_eps);
        x = residual + bl_fw(attn, lw.o);
        residual = x;

        x = mx::fast::rms_norm(x, lw.post_attn_norm_w, cfg.rms_norm_eps);
        auto g = bl_fw(x, lw.gate);
        g = mx::maximum(g, mx::array(0.0f)); g = g * g;
        x = residual + bl_fw(
            mx::fast::rms_norm(g * bl_fw(x, lw.up), lw.ffn_sub_norm_w, cfg.rms_norm_eps), lw.down);
    }

    x = mx::fast::rms_norm(x, norm_w, cfg.rms_norm_eps);
    return mx::matmul(x, mx::transpose(embed_w));
}

int main(int argc, char* argv[]) {
    std::string model_path = "models/bitnet-2b";
    int max_tokens = 200;
    int DRAFT_LAYERS = 4;          // Draft model uses first N layers
    int SPEC_LOOKAHEAD = 6;        // Speculate N tokens ahead
    if (argc > 1) model_path = argv[1];
    if (argc > 2) max_tokens = std::stoi(argv[2]);
    if (argc > 3) DRAFT_LAYERS = std::stoi(argv[3]);
    if (argc > 4) SPEC_LOOKAHEAD = std::stoi(argv[4]);

    Config cfg;
    mx::set_memory_limit(8ULL * 1024 * 1024 * 1024);
    mx::set_cache_limit(4ULL * 1024 * 1024 * 1024);

    std::cout << "========================================\n";
    std::cout << "BitNet C++ v4 — Speculative Decoding\n";
    std::cout << "  Draft: " << DRAFT_LAYERS << " layers, Lookahead: " << SPEC_LOOKAHEAD << "\n";
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

    // ========== Normal decode (baseline) ==========
    {
        std::vector<int> prompt_tokens = {128000, 12805, 5304, 264, 892};
        int eos_token = 128009;
        std::vector<KVCache> kv(cfg.num_layers);

        auto ids = mx::array(prompt_tokens.data(), {1, (int)prompt_tokens.size()}, mx::int32);
        auto logits = model_forward(ids, embed_w, norm_w, layers, cfg, kv);
        for (auto& c : kv) c.offset = (int)prompt_tokens.size();

        auto last = mx::reshape(mx::slice(logits, {0, logits.shape(1)-1, 0}, {1, logits.shape(1), logits.shape(2)}), {1, -1});
        auto y = mx::argmax(last, 1);

        std::vector<int> gen;
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int s = 0; s < max_tokens; s++) {
            auto next_in = mx::astype(mx::reshape(y, {1, 1}), mx::int32);
            logits = model_forward(next_in, embed_w, norm_w, layers, cfg, kv);
            for (auto& c : kv) c.offset++;
            auto nl = mx::reshape(mx::slice(logits, {0, 0, 0}, {1, 1, logits.shape(2)}), {1, -1});
            auto ny = mx::argmax(nl, 1);
            mx::async_eval(ny);
            int tid = y.item<int>();
            if (tid == eos_token) break;
            gen.push_back(tid);
            y = ny;
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double s = std::chrono::duration<double>(t1 - t0).count();
        std::cout << "\nBaseline: " << gen.size() << " tokens in " << s << "s = "
                  << gen.size()/s << " tok/s\n";
    }

    // ========== Speculative decode ==========
    {
        std::vector<int> prompt_tokens = {128000, 12805, 5304, 264, 892};
        int eos_token = 128009;
        
        // Two separate caches: draft and full
        std::vector<KVCache> draft_kv(cfg.num_layers);
        std::vector<KVCache> full_kv(cfg.num_layers);

        // Prefill both
        auto ids = mx::array(prompt_tokens.data(), {1, (int)prompt_tokens.size()}, mx::int32);
        
        auto draft_logits = model_forward(ids, embed_w, norm_w, layers, cfg, draft_kv, DRAFT_LAYERS);
        for (int i = 0; i < cfg.num_layers; i++) draft_kv[i].offset = (int)prompt_tokens.size();
        
        auto full_logits = model_forward(ids, embed_w, norm_w, layers, cfg, full_kv);
        for (int i = 0; i < cfg.num_layers; i++) full_kv[i].offset = (int)prompt_tokens.size();
        
        // Get first token from full model
        auto last = mx::reshape(mx::slice(full_logits, {0, full_logits.shape(1)-1, 0},
            {1, full_logits.shape(1), full_logits.shape(2)}), {1, -1});
        auto current_token = mx::argmax(last, 1);
        mx::eval(current_token);

        std::vector<int> gen;
        int total_draft = 0, total_accepted = 0;
        
        auto t0 = std::chrono::high_resolution_clock::now();
        
        while ((int)gen.size() < max_tokens) {
            int tid = current_token.item<int>();
            if (tid == eos_token) break;
            gen.push_back(tid);
            
            // Step 1: Generate SPEC_LOOKAHEAD draft tokens
            std::vector<mx::array> draft_tokens;
            auto dt = current_token;
            for (int d = 0; d < SPEC_LOOKAHEAD; d++) {
                auto din = mx::astype(mx::reshape(dt, {1, 1}), mx::int32);
                auto dl = model_forward(din, embed_w, norm_w, layers, cfg, draft_kv, DRAFT_LAYERS);
                for (int i = 0; i < cfg.num_layers; i++) draft_kv[i].offset++;
                auto dnl = mx::reshape(mx::slice(dl, {0, 0, 0}, {1, 1, dl.shape(2)}), {1, -1});
                dt = mx::argmax(dnl, 1);
                mx::eval(dt);
                draft_tokens.push_back(dt);
                if (dt.item<int>() == eos_token) break;
            }
            total_draft += draft_tokens.size();
            
            // Step 2: Build verification sequence [current_token, draft_0, draft_1, ...]
            std::vector<int> verify_seq;
            verify_seq.push_back(tid);
            for (auto& t : draft_tokens) verify_seq.push_back(t.item<int>());
            
            auto verify_ids = mx::array(verify_seq.data(), {1, (int)verify_seq.size()}, mx::int32);
            auto verify_logits = model_forward(verify_ids, embed_w, norm_w, layers, cfg, full_kv);
            for (int i = 0; i < cfg.num_layers; i++) full_kv[i].offset += (int)verify_seq.size();
            mx::eval(verify_logits);
            
            // Step 3: Compare full model's predictions with draft predictions
            int accepted = 0;
            for (int d = 0; d < (int)draft_tokens.size(); d++) {
                // Full model's prediction at position d (for token d+1)
                auto full_pred = mx::argmax(
                    mx::reshape(mx::slice(verify_logits, {0, d, 0}, {1, d+1, verify_logits.shape(2)}), {-1}), 0);
                mx::eval(full_pred);
                
                int draft_tid = draft_tokens[d].item<int>();
                int full_tid = full_pred.item<int>();
                
                if (draft_tid == full_tid) {
                    gen.push_back(draft_tid);
                    accepted++;
                    total_accepted++;
                } else {
                    // Diverged: use full model's token, rollback draft cache
                    gen.push_back(full_tid);
                    accepted++;
                    total_accepted++;
                    
                    // Rollback remaining draft tokens from draft cache
                    int rollback = (int)draft_tokens.size() - d - 1;
                    for (int i = 0; i < cfg.num_layers; i++) {
                        draft_kv[i].rollback(rollback);
                    }
                    // Also rollback from full cache
                    int full_rollback = (int)draft_tokens.size() - d - 1;
                    for (int i = 0; i < cfg.num_layers; i++) {
                        full_kv[i].rollback(full_rollback);
                    }
                    break;
                }
            }
            
            // Get the next token from the last verified position
            int last_pos = accepted;  // Position after all accepted tokens
            auto next_pred = mx::argmax(
                mx::reshape(mx::slice(verify_logits, {0, last_pos, 0}, {1, last_pos+1, verify_logits.shape(2)}), {-1}), 0);
            mx::eval(next_pred);
            current_token = next_pred;
            
            // Sync draft cache position with full cache
            // Feed accepted tokens into draft cache
            if (accepted > 0) {
                // Draft cache needs to catch up — forward draft tokens through draft model
                // This is already handled since we ran draft tokens through draft model
            }
            
            if ((int)gen.size() >= max_tokens) break;
        }
        
        auto t1 = std::chrono::high_resolution_clock::now();
        double s = std::chrono::duration<double>(t1 - t0).count();
        double accept_rate = total_draft > 0 ? (double)total_accepted / total_draft * 100 : 0;
        
        std::cout << "\nSpeculative: " << gen.size() << " tokens in " << s << "s = "
                  << gen.size()/s << " tok/s\n";
        std::cout << "  Draft tokens: " << total_draft << ", Accepted: " << total_accepted
                  << " (" << accept_rate << "%)\n";
        std::cout << "  Effective tokens per step: " << (total_draft > 0 ? (double)gen.size() / (gen.size() / (double)(total_accepted + 1)) : 0) << "\n";
        std::cout << "  IDs: ";
        for (int i = 0; i < std::min((int)gen.size(), 20); i++) std::cout << gen[i] << " ";
        std::cout << "...\n";
    }

    return 0;
}
