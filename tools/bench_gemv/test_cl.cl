#pragma OPENCL EXTENSION cl_khr_subgroups : enable
__kernel void test_subgroup(__global float *out, __global const float *in) {
    int lid = get_sub_group_local_id();
    float v = in[get_global_id(0)];
    float s = sub_group_reduce_add(v);
    if (lid == 0) out[get_sub_group_id()] = s;
}
