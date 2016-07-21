__kernel void A(__global float *a, __global float *b, __global float *c, const int d, const int e) {
  int f = get_global_id(0);

  if (f < d - 1 - e) {
    *(a + d * (f + e + 1) + e) = *(b + d * (f + e + 1) + e) / *(b + d * e + e);
  }
}