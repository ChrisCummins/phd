__kernel void A(__global float *a, __local float *b, int c, int d) {
  int e, f;
  int g = get_local_id(0);

  int h = d * c + d;
  for (e = 0; e < 64; e++) {
    b[e * 64 + g] = a[h + g];
    h += c;
  }

  barrier(1);

  for (e = 0; e < 64 - 1; e++) {
    if (g > e) {
      for (f = 0; f < e; f++) b[g * 64 + e] -= b[g * 64 + f] * b[f * 64 + e];
      b[g * 64 + e] /= b[e * 64 + e];
    }

    barrier(1);
    if (g > e) {
      for (f = 0; f < e + 1; f++)
        b[(e + 1) * 64 + g] -= b[(e + 1) * 64 + f] * b[f * 64 + g];
    }

    barrier(1);
  }

  h = (d + 1) * c + d;
  for (e = 1; e < 64; e++) {
    a[h + g] = b[e * 64 + g];
    h += c;
  }
}