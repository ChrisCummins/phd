__kernel void A(__global float *a, __global float *b, int c, int d) {
  unsigned int e = get_global_id(0);

  if (e < c) {
    for (int f = 0; f < d; f++) b[f * c + e] = a[e * d + f];
  }
}