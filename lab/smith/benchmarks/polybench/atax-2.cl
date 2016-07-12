__kernel void A(__global float *a, __global float *b, __global float *c, int d,
                int e) {
  int f = get_global_id(0);

  if (f < e) {
    int g;
    for (g = 0; g < d; g++) {
      b[f] += a[g * e + f] * c[g];
    }
  }
}