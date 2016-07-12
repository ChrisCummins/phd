__kernel void A(__global float *a, __global float *b, int c, int d) {
  int e = get_global_id(0);
  int f = get_global_id(1);

  float g, h, i, j, k, l, m, n, o;
  g = +0.2;
  j = +0.5;
  m = -0.8;
  h = -0.3;
  k = +0.6;
  n = -0.9;
  i = +0.4;
  l = +0.7;
  o = +0.10;
  if ((f < (c - 1)) && (e < (d - 1)) && (f > 0) && (e > 0)) {
    b[f * d + e] = g * a[(f - 1) * d + (e - 1)] + j * a[(f - 1) * d + (e + 0)] +
                   m * a[(f - 1) * d + (e + 1)] + h * a[(f + 0) * d + (e - 1)] +
                   k * a[(f + 0) * d + (e + 0)] + n * a[(f + 0) * d + (e + 1)] +
                   i * a[(f + 1) * d + (e - 1)] + l * a[(f + 1) * d + (e + 0)] +
                   o * a[(f + 1) * d + (e + 1)];
  }
}