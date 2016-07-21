__kernel void A(__global const float* a, __global const float* b, __global float* c, int d) {
  int e = get_global_id(0);

  if (e >= d) {
    return;
  }

  c[e] = a[e] + b[e];
}