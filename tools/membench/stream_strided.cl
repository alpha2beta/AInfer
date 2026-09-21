// T1.7: strided read-streaming proxy (pointer-chase-hostile access pattern).
// Same reduction discipline as stream_read.cl: per-group partial sums,
// plain indexed stores, no atomics.
__kernel void stream_strided(
    __global const float4 * restrict src, // N float4 words
    __global float * restrict out,        // [num_groups] partial sums
    uint nwords,                          // total float4 words
    uint stride_words                     // stride between consecutive reads
) {
    uint gid = get_global_id(0);
    uint nthreads = get_global_size(0);
    float acc = 0.0f;
    // Each thread walks its own lane with a fixed stride: defeats
    // prefetching/coalescing proportionally to stride_words.
    for (uint i = gid; i < nwords; i += nthreads) {
        uint idx = (i * stride_words) % nwords;
        float4 v = src[idx];
        acc += v.x + v.y + v.z + v.w;
    }
    __local float scratch[256];
    uint lid = get_local_id(0);
    uint lsize = get_local_size(0);
    scratch[lid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint s = lsize >> 1; s > 0; s >>= 1) {
        if (lid < s) scratch[lid] += scratch[lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) {
        out[get_group_id(0)] = scratch[0];
    }
}
