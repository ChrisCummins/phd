__kernel void A(__global float *a, __global float *b, __global float *c, int d, int e, int f) {
  int g = get_global_id(0);

  if (g < e) {
    c[g * f + d] = a[g * f + d] / b[d * f + d];
  }
}