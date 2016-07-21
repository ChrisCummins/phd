__kernel void A(__global float *a, __global float *b, __global float *c, int d, int e) {
  int f = get_global_id(0);

  if (f < d) {
    c[f] = 0.0;

    int g;
    for (g = 0; g < e; g++) {
      c[f] += a[f * e + g] * b[g];
    }
  }
}