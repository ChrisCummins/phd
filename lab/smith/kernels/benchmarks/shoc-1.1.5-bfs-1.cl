__kernel void A(__global unsigned int *a, __global unsigned int *b, __global unsigned int *c, int d, int e, unsigned int f, int g, __global int *h) {
  int i = get_global_id(0);
  int j = i % d;
  int k = i / d;
  int l = k * e;
  int m = e + 1;

  if ((l + e) >= f) {
    m = f - l + 1;
    if (m < 0)
      m = 0;
  }

  for (int n = l; n < m - 1 + l; n++) {
    if (a[n] == g) {
      unsigned int o = b[n + 1] - b[n];
      unsigned int p = b[n];
      for (int q = j; q < o; q += d) {
        int n = c[q + p];
        if (a[n] == 0xffffffff) {
          a[n] = g + 1;
          *h = 1;
        }
      }
    }
  }
}