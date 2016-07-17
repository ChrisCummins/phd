__kernel void A(__global float *a, __global float *b, __global float *c,
                __global float *d, __global float *e, float f, float g,
                unsigned int h) {
  for (unsigned int i = get_global_id(0); i < h; i += get_global_size(0))
    B(&a[i], &b[i], c[i], d[i], e[i], f, g);
}