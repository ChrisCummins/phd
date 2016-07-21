__kernel void A(__global float *a, __global float *b, __global float *c, int d, int e, int f) {
  int g = get_global_id(0);

  if ((g > d) && (g < f)) {
    b[d * f + g] = 0.0;

    int h;
    for (h = 0; h < e; h++) {
      b[d * f + g] += c[h * f + d] * a[h * f + g];
    }

    for (h = 0; h < e; h++) {
      a[h * f + g] -= c[h * f + d] * b[d * f + g];
    }
  }
}