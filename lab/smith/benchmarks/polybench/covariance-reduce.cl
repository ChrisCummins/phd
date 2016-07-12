__kernel void A(__global float *a, __global float *b, int c, int d) {
  int e = get_global_id(0) + 1;
  int f = get_global_id(1) + 1;

  if ((f >= 1) && (f < (d + 1)) && (e >= 1) && (e < (c + 1))) {
    b[f * (c + 1) + e] -= a[e];
  }
}