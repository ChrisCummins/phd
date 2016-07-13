__kernel void A(__global float* a, __global float* b, __global float* c,
                __global float* d, __global float* e, __global int* f,
                __global int* g, __global float* h, __global unsigned char* i,
                __global float* j, __global float* k, const int l, const int m,
                const int n, int o, const int p, const int q, __global int* r,
                __global float* s, __local float* t) {
  int u = get_group_id(0);
  int v = get_local_id(0);
  int w = get_global_id(0);
  size_t x = get_local_size(0);
  int y;
  int z, aa;

  if (w < l) {
    a[w] = c[w];
    b[w] = d[w];

    k[w] = 1 / ((float)(l));

    a[w] = a[w] + 1.0 + 5.0 * E(r, w);
    b[w] = b[w] - 2.0 + 2.0 * E(r, w);
  }

  barrier(2);

  if (w < l) {
    for (y = 0; y < m; y++) {
      z = A(a[w]) + g[y * 2 + 1];
      aa = A(b[w]) + g[y * 2];

      f[w * m + y] = abs(z * p * q + aa * q + o);
      if (f[w * m + y] >= n) f[w * m + y] = 0;
    }
    h[w] = B(i, f, m, w);

    h[w] = h[w] / m - 300;

    k[w] = k[w] * exp(h[w]);
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