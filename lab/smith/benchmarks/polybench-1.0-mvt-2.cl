__kernel void A(__global float *a, __global float *b, __global float *c,
                int d) {
  int e = get_global_id(0);

  if (e < d) {
    int f;
    for (f = 0; f < d; f++) {
      b[e] += a[f * d + e] * c[f];
    }
  }
}