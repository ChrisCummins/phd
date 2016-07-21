__kernel void A(int a, __global int* b, __global int* c, __global int* d, int e, int f, int g, int h, int i, __local int* j, __local int* k, __global int* l) {
  int m = get_local_size(0);
  int n = get_group_id(0);
  int o = get_local_id(0);

  int p = m - (a * i * 2);

  int q = (p * n) - h;
  int r = q + m - 1;

  int s = q + o;

  int t = (q < 0) ? -q : 0;
  int u = (r > e - 1) ? m - 1 - (r - e + 1) : m - 1;

  int v = o - 1;
  int w = o + 1;

  v = (v < t) ? t : v;
  w = (w > u) ? u : w;

  bool x = ((o) >= (t) && (o) <= (u));

  if (((s) >= (0) && (s) <= (e - 1))) {
    j[o] = c[s];
  }

  barrier(1);

  bool y;
  for (int z = 0; z < a; z++) {
    y = false;

    if (((o) >= (z + 1) && (o) <= (m - z - 2)) && x) {
      y = true;
      int aa = j[v];
      int ab = j[o];
      int ac = j[w];
      int ad = ((aa) <= (ab) ? (aa) : (ab));
      ad = ((ad) <= (ac) ? (ad) : (ac));

      int ae = e * (g + z) + s;
      k[o] = ad + b[ae];

      if (o == 11 && z == 0) {
        int af = c[s];

        l[af] = 1;
      }
    }

    barrier(1);

    if (z == a - 1) {
      break;
    }

    if (y) {
      j[o] = k[o];
    }
    barrier(1);
  }

  if (y) {
    d[s] = k[o];
  }
}