template <typename T>
static inline T simd_reduce_sum(T x) {
    x += simd_shuffle_xor(x, 16);
    x += simd_shuffle_xor(x, 8);
    x += simd_shuffle_xor(x, 4);
    x += simd_shuffle_xor(x, 2);
    x += simd_shuffle_xor(x, 1);
    return x;
}

template <typename T>
static inline T simd_reduce_max(T x) {
    x = max(x, simd_shuffle_xor(x, 16));
    x = max(x, simd_shuffle_xor(x, 8));
    x = max(x, simd_shuffle_xor(x, 4));
    x = max(x, simd_shuffle_xor(x, 2));
    x = max(x, simd_shuffle_xor(x, 1));
    return x;
}

template <typename T>
static inline T simd_reduce_min(T x) {
    x = min(x, simd_shuffle_xor(x, 16));
    x = min(x, simd_shuffle_xor(x, 8));
    x = min(x, simd_shuffle_xor(x, 4));
    x = min(x, simd_shuffle_xor(x, 2));
    x = min(x, simd_shuffle_xor(x, 1));
    return x;
}

template <typename T>
static inline T block_reduce_sum(T x, threadgroup T* smem, uint tid, uint block_size) {
    x = simd_reduce_sum(x);
    if (block_size > 32) {
        uint simd_id = tid / 32;
        uint lane_id = tid % 32;
        if (lane_id == 0) {
            smem[simd_id] = x;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        x = (tid < (block_size / 32)) ? smem[tid] : T(0);
        x = simd_reduce_sum(x);
    }
    return x;
}

template <typename T>
static inline T block_reduce_max(T x, threadgroup T* smem, uint tid, uint block_size) {
    x = simd_reduce_max(x);
    if (block_size > 32) {
        uint simd_id = tid / 32;
        uint lane_id = tid % 32;
        if (lane_id == 0) {
            smem[simd_id] = x;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        x = (tid < (block_size / 32)) ? smem[tid] : T(-INFINITY);
        x = simd_reduce_max(x);
    }
    return x;
}

template <typename T>
static inline T block_reduce_min(T x, threadgroup T* smem, uint tid, uint block_size) {
    x = simd_reduce_min(x);
    if (block_size > 32) {
        uint simd_id = tid / 32;
        uint lane_id = tid % 32;
        if (lane_id == 0) {
            smem[simd_id] = x;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        x = (tid < (block_size / 32)) ? smem[tid] : T(INFINITY);
        x = simd_reduce_min(x);
    }
    return x;
}
