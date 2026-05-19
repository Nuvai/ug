using uint32_t = unsigned int;

extern "C" __global__ void causal_mask_f32(
    const float *src,
    float *dst,
    const uint32_t num_elements,
    const uint32_t s1,
    const uint32_t s2
) {
    uint32_t block_size = s1 * s2;
    uint32_t start = blockIdx.x * block_size;
    uint32_t stop = min(start + block_size, num_elements);
    for (uint32_t idx = start + threadIdx.x; idx < stop; idx += blockDim.x) {
        uint32_t local = idx - start;
        uint32_t i1 = local / s2;
        uint32_t i2 = local % s2;
        dst[idx] = (i2 <= i1) ? src[idx] : (-1.0f / 0.0f);
    }
}
