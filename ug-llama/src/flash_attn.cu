#define BLOCK_SIZE 32

extern "C" __global__ void flash_attention_f32(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    const unsigned int seq_len,
    const unsigned int head_dim,
    const unsigned int kv_len,
    const float scale
) {
    unsigned int batch_head = blockIdx.x;
    unsigned int q_row = blockIdx.y;
    unsigned int tid = threadIdx.x;
    unsigned int ntg = blockDim.x;

    if (q_row >= seq_len) return;

    unsigned int q_base = batch_head * seq_len * head_dim + q_row * head_dim;
    unsigned int kv_base = batch_head * kv_len * head_dim;

    float row_max = -INFINITY;
    float row_sum = 0.0f;
    float acc[128];
    for (unsigned int d = 0; d < head_dim && d < 128; d++) {
        acc[d] = 0.0f;
    }

    for (unsigned int kv_start = 0; kv_start < kv_len; kv_start += BLOCK_SIZE) {
        unsigned int kv_end = min(kv_start + BLOCK_SIZE, kv_len);

        for (unsigned int j = kv_start; j < kv_end; j++) {
            float score = 0.0f;
            unsigned int k_off = kv_base + j * head_dim;
            for (unsigned int d = tid; d < head_dim; d += ntg) {
                score += Q[q_base + d] * K[k_off + d];
            }

            for (unsigned int offset = ntg / 2; offset > 0; offset >>= 1) {
                score += __shfl_xor_sync(0xFFFFFFFF, score, offset);
            }

            score *= scale;

            if (j > q_row) {
                score = -INFINITY;
            }

            float old_max = row_max;
            row_max = fmaxf(row_max, score);
            float exp_diff = expf(old_max - row_max);
            float exp_score = expf(score - row_max);

            row_sum = row_sum * exp_diff + exp_score;

            unsigned int v_off = kv_base + j * head_dim;
            for (unsigned int d = tid; d < head_dim; d += ntg) {
                acc[d] = acc[d] * exp_diff + exp_score * V[v_off + d];
            }
        }
    }

    float inv_sum = 1.0f / row_sum;
    unsigned int o_base = batch_head * seq_len * head_dim + q_row * head_dim;
    for (unsigned int d = tid; d < head_dim; d += ntg) {
        O[o_base + d] = acc[d] * inv_sum;
    }
}
