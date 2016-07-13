__kernel void A(int a, __global float *b, __global float *c,
                __constant float *d, __constant float *e, __constant int *f,
                __constant int *g, __global float *h, int i, int j) {
  int k, l, m, n, o, p;
  int q = get_global_id(0);
  if (q >= i * j) return;

  k = q / i + 20 + 2;
  l = q % i + 20 + 2;

  float r = 0.f;

  for (m = 0; m < 7; m++) {
    float s = 0.f, t = 0.f, u = 0.f;

    for (n = 0; n < 150; n++) {
      p = l + g[(m * 150) + n];
      o = k + f[(m * 150) + n];

      int v = o * a + p;
      float w = b[v] * e[n] + c[v] * d[n];

      s += w;

      float x = w - u;
      u = u + (x / (float)(n + 1));
      t = t + (x * (w - u));
    }

    u = s / ((float)150);

    float y = t / ((float)(150 - 1));

    if (((u * u) / y) > r) r = (u * u) / y;
  }

  h[(k * a) + l] = r;
}