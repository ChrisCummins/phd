struct Node {
  int x;
  int y;
};
struct Edge {
  int x;
  int y;
};

__kernel void A(__global int *a, __global int *b, __global struct Node *c,
                __global struct Edge *d, __global int *e, __global int *f,
                __global int *g, int h, int i, int j, __local int *k,
                __local int *l, __local int *m) {
  if (get_local_id(0) == 0) {
    *k = 0;
  }
  barrier(1);

  int n = get_global_id(0);

  if (n < h) {
    int o = a[n];
    e[o] = 16677221;
    int p = f[o];

    struct Node q = c[o];

    for (int r = q.x; r < q.y + q.x; r++)

    {
      struct Edge s = d[r];
      int t = s.x;
      int u = s.y;
      u += p;
      int v = atom_min(&f[t], u);
      if (v > u) {
        if (e[t] > 16677216) {
          int w = atom_xchg(&e[t], i);

          if (w != i) {
            int x = atom_add(k, 1);
            l[x] = t;
          }
        }
      }
    }
  }
  barrier(1);

  if (get_local_id(0) == 0) {
    int y = *k;

    *m = atom_add(g, y);
  }
  barrier(1);

  int z = get_local_id(0);

  while (z < *k) {
    b[*m + z] = l[z];

    z += get_local_size(0);
  }
}