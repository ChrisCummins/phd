__kernel void A(__global const uint *a, __global const uint *b,
                __global uint *c, const int d, __local uint *e, const int f) {
  __local uint g[16];

  __local uint h[16];

  __private int i[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  __global uint4 *j = (__global uint4 *)a;
  __global uint4 *k = (__global uint4 *)c;
  int l = d / 4;

  int m = l / get_num_groups(0);
  int n = get_group_id(0) * m;

  int o = (get_group_id(0) == get_num_groups(0) - 1) ? l : n + m;

  int p = n + get_local_id(0);
  int q = n;

  if (get_local_id(0) < 16) {
    h[get_local_id(0)] = 0;
    g[get_local_id(0)] =
        b[(get_local_id(0) * get_num_groups(0)) + get_group_id(0)];
  }
  barrier(1);

  while (q < o) {
    for (int r = 0; r < 16; r++) i[r] = 0;
    uint4 s;
    uint4 t;

    if (p < o) {
      s = j[p];

      t.x = (s.x >> f) & 0xFU;
      t.y = (s.y >> f) & 0xFU;
      t.z = (s.z >> f) & 0xFU;
      t.w = (s.w >> f) & 0xFU;

      i[t.x]++;
      i[t.y]++;
      i[t.z]++;
      i[t.w]++;
    }

    for (int u = 0; u < 16; u++) {
      i[u] = B(i[u], e, 1);
      barrier(1);
    }

    if (p < o) {
      int v;
      v = i[t.x] + g[t.x] + h[t.x];
      c[v] = s.x;
      i[t.x]++;

      v = i[t.y] + g[t.y] + h[t.y];
      c[v] = s.y;
      i[t.y]++;

      v = i[t.z] + g[t.z] + h[t.z];
      c[v] = s.z;
      i[t.z]++;

      v = i[t.w] + g[t.w] + h[t.w];
      c[v] = s.w;
      i[t.w]++;
    }

    barrier(1);

    if (get_local_id(0) == get_local_size(0) - 1) {
      for (int r = 0; r < 16; r++) {
        h[r] += i[r];
      }
    }
    barrier(1);

    q += get_local_size(0);
    p += get_local_size(0);
  }
}