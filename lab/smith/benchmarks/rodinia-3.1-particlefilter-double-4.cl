__kernel void A(__global double* a, __global double* b, __global double* c,
                __global double* d, __global double* e, __global int* f,
                __global int* g, __global double* h, __global unsigned char* i,
                __global double* j, __global double* k, const int l,
                const int m, const int n, int o, const int p, const int q,
                __global int* r, __global double* s, __local double* t) {
  int u = get_group_id(0);
  int v = get_local_id(0);
  int w = get_global_id(0);
  size_t x = get_local_size(0);
  int y;
  int z, aa;

  if (w < l) {
    a[w] = c[w];
    b[w] = d[w];

    k[w] = 1 / ((double)(l));

    a[w] = a[w] + 1.0 + 5.0 * d_randn(r, w);
    b[w] = b[w] - 2.0 + 2.0 * d_randn(r, w);
  }

  barrier(2);

  if (w < l) {
    for (y = 0; y < m; y++) {
      z = dev_round_double(a[w]) + g[y * 2 + 1];
      aa = dev_round_double(b[w]) + g[y * 2];

      f[w * m + y] = abs(z * p * q + aa * q + o);
      if (f[w * m + y] >= n) f[w * m + y] = 0;
    }
    h[w] = calcLikelihoodSum(i, f, m, w);

    h[w] = h[w] / m;

    k[w] = k[w] * h[w];
  }

  t[v] = 0.0;

  barrier(1 | 2);

  if (w < l) {
    t[v] = k[w];
  }

  barrier(1);

  for (unsigned int ab = x / 2; ab > 0; ab >>= 1) {
    if (v < ab) {
      t[v] += t[v + ab];
    }
    barrier(1);
  }
  if (v == 0) {
    s[u] = t[0];
  }
}