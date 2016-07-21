__kernel void A(

    volatile __global unsigned int *a, unsigned int b, volatile __global int *c, volatile __global unsigned int *d, __global unsigned int *e, __global unsigned int *f, unsigned int g, unsigned int h, volatile __global unsigned int *i, const unsigned int j,

    volatile __local unsigned int *k, volatile __local unsigned int *l) {
  volatile __local unsigned int m[1];
  volatile __local unsigned int n[1];

  unsigned int o = get_local_id(0);

  if (o < b) {
    k[o] = a[o];
  }

  unsigned int p = b;
  barrier(1 | 2);
  while (1) {
    if (o == 0) {
      n[0] = 0;
      m[0] = 0;
    }
    barrier(1 | 2);
    if (o < p) {
      unsigned int q = k[o];

      c[q] = 0;

      unsigned int r = e[q];
      unsigned int s = e[q + 1];

      while (r < s) {
        unsigned int t = f[r];

        unsigned int u = atomic_min(&d[t], d[q] + 1);

        if (u > d[q] + 1) {
          int v = atomic_xchg(&c[t], 1);

          if (v == 0) {
            unsigned int w = atomic_add(&n[0], 1);
            if (w < j) {
              l[w] = t;
            }

            else {
              int x = atomic_add(&m[0], 1);
              a[x] = t;
            }
          }
        }
        r++;
      }
    }
    barrier(1 | 2);

    if (o < j)
      k[o] = l[o];
    barrier(1 | 2);

    if (n[0] == 0) {
      if (o == 0)
        i[0] = 0;

      return;
    }

    else if (n[0] > get_local_size(0) || n[0] > j) {
      if (o < (n[0] - m[0]))
        a[m[0] + o] = k[o];
      if (o == 0) {
        i[0] = n[0];
      }
      return;
    }
    p = n[0];
    barrier(1 | 2);
  }
}