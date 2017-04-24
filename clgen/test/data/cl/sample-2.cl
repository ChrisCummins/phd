__kernel void A(__global float* a, __global float* b, const int c) {
    int d = get_global_id(0);

    if (d < c) {
        a[d] += 1;
    }
}


__kernel void B(__global float* a, __global float* b, const int c) {
    int d = get_global_id(0);

    for (int i = 0; i < d * 1000; ++i)
        a[d] += 1;
}


__kernel void C(__global int* a, __global int* b, const int c) {
    int d = get_global_id(0);
    if (d < c)
        a[d] = b[d];
}
