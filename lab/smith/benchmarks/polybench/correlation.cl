__kernel void A(__global float *a, __global float *b, int c, int d) {
  int e = get_global_id(0) + 1;

  int f, g;
  if ((e >= 1) && (e < c)) {
    a[e * (c + 1) + e] = 1.0;

    for (g = (e + 1); g < (c + 1); g++) {
      for (f = 1; f < (d + 1); f++) {
        a[e * (c + 1) + g] += b[f * (c + 1) + e] * b[f * (c + 1) + g];
      }
      a[g * (c + 1) + e] = a[e * (c + 1) + g];
    }
  }
}