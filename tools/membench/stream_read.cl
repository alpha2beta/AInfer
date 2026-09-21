// T1.6: sequential weight-streaming proxy for Arc 140V (Xe2).
// Each work-item streams float4 words over a large strided slice of the
// buffer; per-workgroup partial sums land in out[group_id] (host folds them)
// so the reads cannot be eliminated. Read-only traffic models INT4 weight
// streaming during decode (weights read once, ~nothing written back).
// No float atomics (extension-dependent) — plain indexed stores only.
__kernel void stream_read(
    __global const float4 * restrict src, // N float4 words
    __global float * restrict out,        // [num_groups] partial sums
    uint nwords                           // total float4 words
) {
    uint gid = get_global_id(0);
    uint nthreads = get_global_size(0);
    float acc = 0.0f;
    // Strided sweep: every thread touches every (nthreads*4)-th float.
    // Coalesced within a subgroup (consecutive lanes read consecutive words).
    for (uint i = gid; i < nwords; i += nthreads) {
        float4 v = src[i];
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
