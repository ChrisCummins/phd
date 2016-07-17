__kernel void A(__global int* a, __global int* b, __global int* c,
                __local int* d, __local int* e, int f, int g, int h, int i,
                int j, int k, int l) {
  int m = get_group_id(0);

  int n = get_local_id(0);

  int o = k * f + l;

  int p = m + i - h;
  int q = i - m - 1;

  int r = o + f * 64 * q + 64 * p + n + (f + 1);
  int s = o + f * 64 * q + 64 * p + n + (1);
  int t = o + f * 64 * q + 64 * p + (f);
  int u = o + f * 64 * q + 64 * p;

  if (n == 0) d[0 + n * (64 + 1)] = b[u];

  for (int v = 0; v < 64; v++) e[n + v * 64] = a[r + f * v];

  barrier(1);

  d[0 + (n + 1) * (64 + 1)] = b[t + f * n];

  barrier(1);

  d[(n + 1) + 0 * (64 + 1)] = b[s];

  barrier(1);

  for (int w = 0; w < 64; w++) {
    if (n <= w) {
      int x = n + 1;
      int y = w - n + 1;

      d[x + y * (64 + 1)] = A(
          d[(x - 1) + (y - 1) * (64 + 1)] + e[(x - 1) + (y - 1) * 64],
          d[(x - 1) + (y) * (64 + 1)] - (g), d[(x) + (y - 1) * (64 + 1)] - (g));
    }
    barrier(1);
  }

  for (int w = 64 - 2; w >= 0; w--) {
    if (n <= w) {
      int x = n + 64 - w;
      int y = 64 - n;

      d[x + y * (64 + 1)] = A(
          d[(x - 1) + (y - 1) * (64 + 1)] + e[(x - 1) + (y - 1) * 64],
          d[(x - 1) + (y) * (64 + 1)] - (g), d[(x) + (y - 1) * (64 + 1)] - (g));
    }

    barrier(1);
  }

  for (int v = 0; v < 64; v++) b[r + v * f] = d[(n + 1) + (v + 1) * (64 + 1)];

  return;
}