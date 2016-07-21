__kernel void A(

    volatile __global unsigned int *a, unsigned int b, volatile __global unsigned int *c, volatile __global int *d, volatile __global unsigned int *e, __global unsigned int *f, __global unsigned int *g, unsigned int h, unsigned int i, volatile __global unsigned int *j, const unsigned int k,

    volatile __local unsigned int *l) {
  volatile __local unsigned int m[1];
  volatile __local unsigned int n[1];

  unsigned int o = get_global_id(0);
  unsigned int p = get_local_id(0);

  if (p == 0) {
    m[0] = 0;
    n[0] = 0;
  }

  barrier(1 | 2);
  if (o < b) {
    unsigned int q = a[o];
    d[q] = 0;

    unsigned int r = f[q];
    unsigned int s = f[q + 1];

    while (r < s) {
      unsigned int t = g[r];

      unsigned int u = atomic_min(&e[t], e[q] + 1);

      if (u > e[q] + 1) {
        int v = atomic_xchg(&d[t], 1);

        if (v == 0) {
          unsigned int w = atomic_add(&m[0], 1);
          if (w < k) {
            l[w] = t;
          }

          else {
            int x = atomic_add(j, 1);
            c[x] = t;
          }
        }
      }
      r++;
    }
  }
  barrier(1 | 2);

  if (p == 0) {
    if (m[0] > k) {
      m[0] = k;
    }
    n[0] = atomic_add(j, m[0]);
  }

  barrier(1 | 2);

  if (p < m[0])
    c[p + n[0]] = l[p];
}