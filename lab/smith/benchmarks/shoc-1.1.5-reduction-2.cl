__kernel void A(__global float *a, __global float *b, unsigned int c) {
  float d = 0.0f;
  for (int e = 0; e < c; e++) {
    d += a[e];
  }
  b[0] = d;
}