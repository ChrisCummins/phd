__kernel void A(__global float *a, __global float *b, int c, int d, int e,
                int f) {
  int g = get_global_id(0);
  int h = get_global_id(1);

  float i, j, k, l, m, n, o, p, q;
  i = +2;
  l = +5;
  o = -8;
  j = -3;
  m = +6;
  p = -9;
  k = +4;
  n = +7;
  q = +10;

  if ((f < (c - 1)) && (h < (d - 1)) && (g < (e - 1)) && (f > 0) && (h > 0) &&
      (g > 0)) {
    b[f * (e * d) + h * e + g] =
        i * a[(f - 1) * (e * d) + (h - 1) * e + (g - 1)] +
        k * a[(f + 1) * (e * d) + (h - 1) * e + (g - 1)] +
        l * a[(f - 1) * (e * d) + (h - 1) * e + (g - 1)] +
        n * a[(f + 1) * (e * d) + (h - 1) * e + (g - 1)] +
        o * a[(f - 1) * (e * d) + (h - 1) * e + (g - 1)] +
        q * a[(f + 1) * (e * d) + (h - 1) * e + (g - 1)] +
        j * a[(f + 0) * (e * d) + (h - 1) * e + (g + 0)] +
        m * a[(f + 0) * (e * d) + (h + 0) * e + (g + 0)] +
        p * a[(f + 0) * (e * d) + (h + 1) * e + (g + 0)] +
        i * a[(f - 1) * (e * d) + (h - 1) * e + (g + 1)] +
        k * a[(f + 1) * (e * d) + (h - 1) * e + (g + 1)] +
        l * a[(f - 1) * (e * d) + (h + 0) * e + (g + 1)] +
        n * a[(f + 1) * (e * d) + (h + 0) * e + (g + 1)] +
        o * a[(f - 1) * (e * d) + (h + 1) * e + (g + 1)] +
        q * a[(f + 1) * (e * d) + (h + 1) * e + (g + 1)];
  } else {
    b[f * (e * d) + h * e + g] = 0;
  }
}