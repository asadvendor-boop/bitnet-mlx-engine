/*
 * BitNet C++ v5 — Speculative Decoding with Trained Draft Model
 *
 * Full BitNet-2B (30L, ternary):  ~89 tok/s  (BitLinear Metal kernel)
 * Draft model (2L, float16):      ~500+ tok/s (standard matmul)
 * Combined via speculative decoding: target 150+ tok/s
 */

#include <iostream>
#include <chrono>
#include <vector>
#include <string>
#include <optional>
#include <cmath>
#include <algorithm>
#include <fstream>

#include "mlx/mlx.h"

namespace mx = mlx::core;

// ========== Full Model Config ==========
struct FullConfig {
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

// ========== Draft Model Config ==========
struct DraftConfig {
    int hidden_size = 512;
    int num_heads = 8;
    int num_layers = 2;
    int intermediate_size = 1408;
    int vocab_size = 152064;
    float rope_theta = 500000.0f;
    float rms_norm_eps = 1e-5f;
    int head_dim = 64;
};

// ========== BitLinear Metal Kernel for Full Model ==========
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

// ========== Full Model Weights ==========
struct BitLinearWeights {
    mx::array weight, weight_scale;
    int in_features, out_features;
};

struct FullLayerWeights {
    mx::array input_norm_w, post_attn_norm_w, attn_sub_norm_w, ffn_sub_norm_w;
    BitLinearWeights q, k, v, o, gate, up, down;
};

// ========== Draft Model Weights ==========
struct DraftLayerWeights {
    mx::array input_norm_w, post_attn_norm_w;
    mx::array q_w, k_w, v_w, o_w;
    mx::array gate_w, up_w, down_w;
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
    }
};

// ========== Full Model Forward ==========
mx::array bl_fw(const mx::array& x, const BitLinearWeights& w) {
    return bitlinear_matmul(x, w.weight, w.weight_scale, w.in_features, w.out_features);
}

mx::array full_forward(
    const mx::array& input_ids, const mx::array& embed_w, const mx::array& norm_w,
    const std::vector<FullLayerWeights>& layers, const FullConfig& cfg,
    std::vector<KVCache>& cache
) {
    int B = input_ids.shape(0), L = input_ids.shape(1);
    auto x = mx::take(embed_w, input_ids, 0);
    std::string mask_mode = L > 1 ? "causal" : "";

    for (int i = 0; i < cfg.num_layers; i++) {
        auto& lw = layers[i];
        auto residual = x;
        x = mx::fast::rms_norm(x, lw.input_norm_w, cfg.rms_norm_eps);
        auto q = bl_fw(x, lw.q), k = bl_fw(x, lw.k), v = bl_fw(x, lw.v);
        q = mx::transpose(mx::reshape(q, {B, L, cfg.num_heads, cfg.head_dim}), {0, 2, 1, 3});
        k = mx::transpose(mx::reshape(k, {B, L, cfg.num_kv_heads, cfg.head_dim}), {0, 2, 1, 3});
        v = mx::transpose(mx::reshape(v, {B, L, cfg.num_kv_heads, cfg.head_dim}), {0, 2, 1, 3});
        int off = cache[i].offset;
        q = mx::fast::rope(q, cfg.head_dim, false, cfg.rope_theta, 1.0f, off);
        k = mx::fast::rope(k, cfg.head_dim, false, cfg.rope_theta, 1.0f, off);
        cache[i].update(k, v);
        float scale = 1.0f / std::sqrt((float)cfg.head_dim);
        std::optional<mx::array> mask_arr;
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

// ========== Draft Model Forward ==========
mx::array draft_forward(
    const mx::array& input_ids, const mx::array& embed_w, const mx::array& norm_w,
    const mx::array& lm_head_w,
    const std::vector<DraftLayerWeights>& layers, const DraftConfig& cfg,
    std::vector<KVCache>& cache
) {
    int B = input_ids.shape(0), L = input_ids.shape(1);
    auto x = mx::take(embed_w, input_ids, 0);
    std::string mask_mode = L > 1 ? "causal" : "";

    for (int i = 0; i < cfg.num_layers; i++) {
        auto& lw = layers[i];
        auto residual = x;
        x = mx::fast::rms_norm(x, lw.input_norm_w, cfg.rms_norm_eps);
        
        // Standard matmul (float16, not BitLinear)
        auto q = mx::matmul(x, mx::transpose(lw.q_w));
        auto k = mx::matmul(x, mx::transpose(lw.k_w));
        auto v = mx::matmul(x, mx::transpose(lw.v_w));
        
        q = mx::transpose(mx::reshape(q, {B, L, cfg.num_heads, cfg.head_dim}), {0, 2, 1, 3});
        k = mx::transpose(mx::reshape(k, {B, L, cfg.num_heads, cfg.head_dim}), {0, 2, 1, 3});
        v = mx::transpose(mx::reshape(v, {B, L, cfg.num_heads, cfg.head_dim}), {0, 2, 1, 3});
        int off = cache[i].offset;
        q = mx::fast::rope(q, cfg.head_dim, false, cfg.rope_theta, 1.0f, off);
        k = mx::fast::rope(k, cfg.head_dim, false, cfg.rope_theta, 1.0f, off);
        cache[i].update(k, v);
        float scale = 1.0f / std::sqrt((float)cfg.head_dim);
        std::optional<mx::array> mask_arr;
        auto attn = mx::fast::scaled_dot_product_attention(q, k, v, scale, mask_mode, mask_arr);
        attn = mx::reshape(mx::transpose(attn, {0, 2, 1, 3}), {B, L, cfg.hidden_size});
        x = residual + mx::matmul(attn, mx::transpose(lw.o_w));
        
        residual = x;
        x = mx::fast::rms_norm(x, lw.post_attn_norm_w, cfg.rms_norm_eps);
        // SiLU gate FFN
        auto gate = mx::matmul(x, mx::transpose(lw.gate_w));
        gate = gate * mx::sigmoid(gate); // SiLU
        auto up_val = mx::matmul(x, mx::transpose(lw.up_w));
        x = residual + mx::matmul(gate * up_val, mx::transpose(lw.down_w));
    }
    x = mx::fast::rms_norm(x, norm_w, cfg.rms_norm_eps);
    return mx::matmul(x, mx::transpose(lm_head_w));
}

int main(int argc, char* argv[]) {
    std::string full_path = "models/bitnet-2b";
    std::string draft_path = "models/draft-model";
    int max_tokens = 200;
    int SPEC_K = 5;  // Draft tokens per speculation step
    
    if (argc > 1) full_path = argv[1];
    if (argc > 2) draft_path = argv[2];
    if (argc > 3) max_tokens = std::stoi(argv[3]);
    if (argc > 4) SPEC_K = std::stoi(argv[4]);

    FullConfig fcfg;
    DraftConfig dcfg;
    mx::set_memory_limit(8ULL * 1024 * 1024 * 1024);
    mx::set_cache_limit(4ULL * 1024 * 1024 * 1024);

    std::cout << "========================================\n";
    std::cout << "BitNet C++ v5 — Speculative Decoding\n";
    std::cout << "  Spec-K: " << SPEC_K << " draft tokens\n";
    std::cout << "========================================\n";

    // Load full model
    std::cout << "Loading full model..." << std::endl;
    auto full_loaded = mx::load_safetensors(full_path + "/model.safetensors");
    auto& fmap = full_loaded.first;
    auto full_embed = mx::astype(fmap.at("model.embed_tokens.weight"), mx::bfloat16);
    auto full_norm = mx::astype(fmap.at("model.norm.weight"), mx::bfloat16);

    std::vector<FullLayerWeights> full_layers;
    for (int i = 0; i < fcfg.num_layers; i++) {
        std::string p = "model.layers." + std::to_string(i);
        auto get_bl = [&](const std::string& name, int inf, int outf) -> BitLinearWeights {
            return {fmap.at(p + "." + name + ".weight"),
                    mx::astype(fmap.at(p + "." + name + ".weight_scale"), mx::bfloat16), inf, outf};
        };
        int hs = fcfg.hidden_size, is = fcfg.intermediate_size;
        int kv = fcfg.num_kv_heads * fcfg.head_dim, qd = fcfg.num_heads * fcfg.head_dim;
        full_layers.push_back({
            mx::astype(fmap.at(p + ".input_layernorm.weight"), mx::bfloat16),
            mx::astype(fmap.at(p + ".post_attention_layernorm.weight"), mx::bfloat16),
            mx::astype(fmap.at(p + ".self_attn.attn_sub_norm.weight"), mx::bfloat16),
            mx::astype(fmap.at(p + ".mlp.ffn_sub_norm.weight"), mx::bfloat16),
            get_bl("self_attn.q_proj", hs, qd), get_bl("self_attn.k_proj", hs, kv),
            get_bl("self_attn.v_proj", hs, kv), get_bl("self_attn.o_proj", qd, hs),
            get_bl("mlp.gate_proj", hs, is), get_bl("mlp.up_proj", hs, is),
            get_bl("mlp.down_proj", is, hs)
        });
    }

    // Load draft model
    std::cout << "Loading draft model..." << std::endl;
    auto draft_loaded = mx::load_safetensors(draft_path + "/model.safetensors");
    auto& dmap = draft_loaded.first;
    auto draft_embed = mx::astype(dmap.at("embed_tokens.weight"), mx::float16);
    auto draft_norm = mx::astype(dmap.at("norm.weight"), mx::float16);
    auto draft_lm_head = mx::astype(dmap.at("lm_head.weight"), mx::float16);

    std::vector<DraftLayerWeights> draft_layers;
    for (int i = 0; i < dcfg.num_layers; i++) {
        std::string p = "layers." + std::to_string(i);
        draft_layers.push_back({
            mx::astype(dmap.at(p + ".input_norm.weight"), mx::float16),
            mx::astype(dmap.at(p + ".post_attn_norm.weight"), mx::float16),
            mx::astype(dmap.at(p + ".self_attn.q_proj.weight"), mx::float16),
            mx::astype(dmap.at(p + ".self_attn.k_proj.weight"), mx::float16),
            mx::astype(dmap.at(p + ".self_attn.v_proj.weight"), mx::float16),
            mx::astype(dmap.at(p + ".self_attn.o_proj.weight"), mx::float16),
            mx::astype(dmap.at(p + ".mlp.gate_proj.weight"), mx::float16),
            mx::astype(dmap.at(p + ".mlp.up_proj.weight"), mx::float16),
            mx::astype(dmap.at(p + ".mlp.down_proj.weight"), mx::float16),
        });
    }
    std::cout << "Both models loaded.\n";

    std::vector<int> prompt_tokens = {128000, 12805, 5304, 264, 892};
    int eos_token = 128009;

    // ========== BASELINE: normal decode ==========
    {
        std::vector<KVCache> full_kv(fcfg.num_layers);
        auto ids = mx::array(prompt_tokens.data(), {1, (int)prompt_tokens.size()}, mx::int32);
        auto logits = full_forward(ids, full_embed, full_norm, full_layers, fcfg, full_kv);
        for (auto& c : full_kv) c.offset = (int)prompt_tokens.size();

        auto last = mx::reshape(mx::slice(logits, {0, logits.shape(1)-1, 0}, {1, logits.shape(1), logits.shape(2)}), {1, -1});
        auto y = mx::argmax(last, 1);
        
        std::vector<int> gen;
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int s = 0; s < max_tokens; s++) {
            auto next_in = mx::astype(mx::reshape(y, {1, 1}), mx::int32);
            logits = full_forward(next_in, full_embed, full_norm, full_layers, fcfg, full_kv);
            for (auto& c : full_kv) c.offset++;
            auto nl = mx::reshape(mx::slice(logits, {0, 0, 0}, {1, 1, logits.shape(2)}), {1, -1});
            auto ny = mx::argmax(nl, 1);
            mx::async_eval(ny);
            int tid = y.item<int>();
            if (tid == eos_token) break;
            gen.push_back(tid);
            y = ny;
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double secs = std::chrono::duration<double>(t1 - t0).count();
        std::cout << "\nBaseline: " << gen.size() << " tok in " << secs << "s = "
                  << gen.size()/secs << " tok/s\n";
    }

    // ========== SPECULATIVE DECODE ==========
    {
        std::vector<KVCache> full_kv(fcfg.num_layers);
        std::vector<KVCache> draft_kv(dcfg.num_layers);
        
        // Prefill full model
        auto ids = mx::array(prompt_tokens.data(), {1, (int)prompt_tokens.size()}, mx::int32);
        auto flogits = full_forward(ids, full_embed, full_norm, full_layers, fcfg, full_kv);
        for (auto& c : full_kv) c.offset = (int)prompt_tokens.size();
        
        // Prefill draft model
        auto dlogits = draft_forward(ids, draft_embed, draft_norm, draft_lm_head, draft_layers, dcfg, draft_kv);
        for (auto& c : draft_kv) c.offset = (int)prompt_tokens.size();
        
        // Get first token from full model
        auto last = mx::reshape(mx::slice(flogits, {0, flogits.shape(1)-1, 0}, {1, flogits.shape(1), flogits.shape(2)}), {1, -1});
        auto current_token = mx::argmax(last, 1);
        mx::eval(current_token);

        std::vector<int> gen;
        int total_draft = 0, total_accepted = 0, total_steps = 0;
        
        auto t0 = std::chrono::high_resolution_clock::now();
        
        while ((int)gen.size() < max_tokens) {
            int tid = current_token.item<int>();
            if (tid == eos_token) break;
            gen.push_back(tid);
            
            // Step 1: Draft model generates K tokens
            std::vector<int> draft_toks;
            auto dt = current_token;
            for (int d = 0; d < SPEC_K; d++) {
                auto din = mx::astype(mx::reshape(dt, {1, 1}), mx::int32);
                auto dl = draft_forward(din, draft_embed, draft_norm, draft_lm_head, draft_layers, dcfg, draft_kv);
                for (auto& c : draft_kv) c.offset++;
                auto dnl = mx::reshape(mx::slice(dl, {0, 0, 0}, {1, 1, dl.shape(2)}), {1, -1});
                dt = mx::argmax(dnl, 1);
                mx::eval(dt);
                draft_toks.push_back(dt.item<int>());
                if (dt.item<int>() == eos_token) break;
            }
            total_draft += draft_toks.size();
            
            // Step 2: Full model verifies [current + draft_tokens] in one pass
            std::vector<int> verify_seq;
            verify_seq.push_back(tid);
            for (auto t : draft_toks) verify_seq.push_back(t);
            
            auto verify_ids = mx::array(verify_seq.data(), {1, (int)verify_seq.size()}, mx::int32);
            auto verify_logits = full_forward(verify_ids, full_embed, full_norm, full_layers, fcfg, full_kv);
            for (auto& c : full_kv) c.offset += (int)verify_seq.size();
            mx::eval(verify_logits);
            
            // Step 3: Accept matching tokens
            int accepted = 0;
            for (int d = 0; d < (int)draft_toks.size(); d++) {
                auto full_pred = mx::argmax(
                    mx::reshape(mx::slice(verify_logits, {0, d, 0}, {1, d+1, verify_logits.shape(2)}), {-1}), 0);
                mx::eval(full_pred);
                
                int full_tid = full_pred.item<int>();
                if (draft_toks[d] == full_tid) {
                    gen.push_back(draft_toks[d]);
                    accepted++;
                    total_accepted++;
                } else {
                    // Diverged: accept full model's prediction instead
                    gen.push_back(full_tid);
                    accepted++;
                    total_accepted++;
                    
                    // Rollback extra from full cache
                    int extra = (int)draft_toks.size() - d - 1;
                    if (extra > 0) {
                        for (auto& c : full_kv) {
                            if (c.keys.has_value()) {
                                int new_len = c.keys->shape(2) - extra;
                                c.keys = mx::slice(*c.keys, {0,0,0,0}, {c.keys->shape(0), c.keys->shape(1), new_len, c.keys->shape(3)});
                                c.values = mx::slice(*c.values, {0,0,0,0}, {c.values->shape(0), c.values->shape(1), new_len, c.values->shape(3)});
                                c.offset -= extra;
                            }
                        }
                    }
                    break;
                }
            }
            
            // Get next token from verification logits
            int next_pos = accepted;
            auto next_pred = mx::argmax(
                mx::reshape(mx::slice(verify_logits, {0, next_pos, 0}, {1, next_pos+1, verify_logits.shape(2)}), {-1}), 0);
            mx::eval(next_pred);
            current_token = next_pred;
            
            // Sync draft cache: rollback speculative tokens, forward accepted ones
            // Draft generated SPEC_K tokens, but only some were accepted
            // Rollback all SPEC_K draft entries
            int to_rollback = (int)draft_toks.size();
            for (auto& c : draft_kv) {
                if (c.keys.has_value() && to_rollback > 0) {
                    int new_len = c.keys->shape(2) - to_rollback;
                    if (new_len > 0) {
                        c.keys = mx::slice(*c.keys, {0,0,0,0}, {c.keys->shape(0), c.keys->shape(1), new_len, c.keys->shape(3)});
                        c.values = mx::slice(*c.values, {0,0,0,0}, {c.values->shape(0), c.values->shape(1), new_len, c.values->shape(3)});
                    } else {
                        c.keys.reset(); c.values.reset();
                    }
                    c.offset -= to_rollback;
                }
            }
            // Now forward the accepted tokens through draft (to rebuild correct cache)
            // Accepted = gen tokens from this step (accepted + 1 for the divergence token)
            std::vector<int> accepted_toks;
            accepted_toks.push_back(tid);  // current token before draft
            for (int i = 0; i < accepted; i++) {
                accepted_toks.push_back(gen[gen.size() - accepted + i]);
            }
            auto acc_ids = mx::array(accepted_toks.data(), {1, (int)accepted_toks.size()}, mx::int32);
            draft_forward(acc_ids, draft_embed, draft_norm, draft_lm_head, draft_layers, dcfg, draft_kv);
            for (auto& c : draft_kv) c.offset += (int)accepted_toks.size();
            
            total_steps++;
            if ((int)gen.size() >= max_tokens) break;
        }
        
        auto t1 = std::chrono::high_resolution_clock::now();
        double secs = std::chrono::duration<double>(t1 - t0).count();
        double accept_rate = total_draft > 0 ? (double)total_accepted / total_draft * 100 : 0;
        
        std::cout << "\nSpeculative: " << gen.size() << " tok in " << secs << "s = "
                  << gen.size()/secs << " tok/s\n";
        std::cout << "  Steps: " << total_steps
                  << ", Draft: " << total_draft << ", Accepted: " << total_accepted
                  << " (" << accept_rate << "%)\n";
        std::cout << "  Avg tokens/step: " << (total_steps > 0 ? (double)gen.size() / total_steps : 0) << "\n";
        std::cout << "  IDs: ";
        for (int i = 0; i < std::min((int)gen.size(), 20); i++) std::cout << gen[i] << " ";
        std::cout << "...\n";
    }

    return 0;
}
