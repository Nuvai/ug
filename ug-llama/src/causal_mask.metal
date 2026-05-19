using uint32_t = unsigned int;

kernel void causal_mask_f32(
    device const float *src,
    device float *dst,
    constant uint32_t &num_elements,
    constant uint32_t &s1,
    constant uint32_t &s2,
    uint tid [[ thread_index_in_threadgroup ]],
    uint dst_id [[ threadgroup_position_in_grid ]],
    uint block_dim [[ threads_per_threadgroup ]]
) {
    uint32_t block_size = s1 * s2;
    uint32_t start = dst_id * block_size;
    uint32_t stop = metal::min(start + block_size, num_elements);
    for (uint32_t idx = start + tid; idx < stop; idx += block_dim) {
        uint32_t local = idx - start;
        uint32_t i1 = local / s2;
        uint32_t i2 = local % s2;
        dst[idx] = (i2 <= i1) ? src[idx] : -INFINITY;
    }
}
