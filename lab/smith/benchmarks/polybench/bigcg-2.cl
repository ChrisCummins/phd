__kernel void A(__global float *a, __global float *b, __global float *c, int d,
                int e) {
  int f = get_global_id(0);

  if (f < e) {
    c[f] = 0.0;

    int g;
    for (g = 0; g < d; g++) {
      c[f] += a[g * e + f] * b[g];
    }
  }
}