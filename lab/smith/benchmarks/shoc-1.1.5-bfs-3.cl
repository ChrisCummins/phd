__kernel void A(

    volatile __global unsigned int *a, unsigned int b,
    volatile __global unsigned int *c, volatile __global int *d,
    volatile __global unsigned int *e, __global unsigned int *f,
    __global unsigned int *g, unsigned int h, unsigned int i,
    volatile __global unsigned int *j, volatile __global unsigned int *k,
    volatile __global unsigned int *l, volatile __global unsigned int *m,
    volatile __global unsigned int *n, const unsigned int o,

    volatile __local unsigned int *p) {
  volatile __local unsigned int q[1];
  volatile __local unsigned int r[1];

  unsigned int s = get_global_id(0);
  unsigned int t = get_local_id(0);

  int u = 0;
  unsigned int v = l[0];
  unsigned int w = b;
  barrier(1 | 2);
  while (1) {
    if (t == 0) {
      q[0] = 0;
      r[0] = 0;
    }
    barrier(1 | 2);
    if (s < w) {
      unsigned int x;

      if (u == 0)
        x = a[s];
      else
        x = c[s];

      d[x] = 0;

      unsigned int y = f[x];
      unsigned int z = f[x + 1];

      while (y < z) {
        unsigned int aa = g[y];

        unsigned int ab = atomic_min(&e[aa], e[x] + 1);

        if (ab > e[x] + 1) {
          int ac = atomic_xchg(&d[aa], 1);

          if (ac == 0) {
            unsigned int ad = atomic_add(&q[0], 1);
            if (ad < o) {
              p[ad] = aa;
            }

            else {
              int ae = atomic_add(m, 1);
              if (u == 0)
                c[ae] = aa;
              else
                a[ae] = aa;
            }
          }
        }
        y++;
      }
    }

    barrier(1 | 2);

    if (t == 0) {
      if (q[0] > o) {
        q[0] = o;
      }
      r[0] = atomic_add(m, q[0]);
    }

    barrier(1 | 2);
    v += get_num_groups(0);

    if (s == 0) {
      n[0] = m[0];
      m[0] = 0;
    }

    if (t < q[0]) {
      if (u == 0)
        c[t + r[0]] = p[t];
      else
        a[t + r[0]] = p[t];
    }

    v += get_num_groups(0);

    if (n[0] < get_local_size(0) ||
        n[0] > get_local_size(0) * get_num_groups(0))
      break;

    u = (u + 1) % 2;

    w = n[0];
  }

  if (u == 0) {
    for (int af = s; af < n[0]; af += get_global_size(0)) a[af] = c[af];
  }
  if (s == 0) {
    j[0] = n[0];
  }
}