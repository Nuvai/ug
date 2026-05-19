#include <metal_stdlib>
using namespace metal;

#define BLOCK_SIZE 32

kernel void flash_attention_f32(
    device const float* Q [[buffer(0)]],
    device const float* K [[buffer(1)]],
    device const float* V [[buffer(2)]],
    device float* O [[buffer(3)]],
    constant uint& seq_len [[buffer(4)]],
    constant uint& head_dim [[buffer(5)]],
    constant uint& kv_len [[buffer(6)]],
    constant float& scale [[buffer(7)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint tpitg [[thread_position_in_threadgroup]],
    uint ntg [[threads_per_threadgroup]]
) {
    uint batch_head = tgpig.x;
    uint q_row = tgpig.y;

    if (q_row >= seq_len) return;

    uint q_base = batch_head * seq_len * head_dim + q_row * head_dim;
    uint kv_base = batch_head * kv_len * head_dim;

    float row_max = -INFINITY;
    float row_sum = 0.0f;
    float acc[128];
    for (uint d = 0; d < head_dim && d < 128; d++) {
        acc[d] = 0.0f;
    }

    for (uint kv_start = 0; kv_start < kv_len; kv_start += BLOCK_SIZE) {
        uint kv_end = min(kv_start + BLOCK_SIZE, kv_len);

        for (uint j = kv_start; j < kv_end; j++) {
            float score = 0.0f;
            uint k_off = kv_base + j * head_dim;
            for (uint d = tpitg; d < head_dim; d += ntg) {
                score += Q[q_base + d] * K[k_off + d];
            }

            // Warp reduce the score
            for (uint offset = ntg / 2; offset > 0; offset >>= 1) {
                score += simd_shuffle_xor(score, offset);
            }

            score *= scale;

            // Causal mask
            if (j > q_row) {
                score = -INFINITY;
            }

            float old_max = row_max;
            row_max = max(row_max, score);
            float exp_diff = exp(old_max - row_max);
            float exp_score = exp(score - row_max);

            row_sum = row_sum * exp_diff + exp_score;

            uint v_off = kv_base + j * head_dim;
            for (uint d = tpitg; d < head_dim; d += ntg) {
                acc[d] = acc[d] * exp_diff + exp_score * V[v_off + d];
            }
        }
    }

    float inv_sum = 1.0f / row_sum;
    uint o_base = batch_head * seq_len * head_dim + q_row * head_dim;
    for (uint d = tpitg; d < head_dim; d += ntg) {
        O[o_base + d] = acc[d] * inv_sum;
    }
}
