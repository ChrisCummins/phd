__kernel void A(__global float* a, __global float* b, const int c, float d, float e, float f, __local float* g) {
  int h = get_group_id(0);
  int i = get_group_id(1);
  int j = get_num_groups(0);
  int k = get_num_groups(1);
  int l = get_local_id(0);
  int m = get_local_id(1);
  int n = 16;
  int o = get_local_size(1);

  int p = FOOBAR(h, n, l);
  int q = B(i, o, m);

  int r = k * o + 2;
  int s = r + (((r % c) == 0) ? 0 : (c - (r % c)));
  int t = s - 2;

  int u = o;
  for (int v = 0; v < (n + 2); v++) {
    int w = C(l - 1 + v, m, u);
    int x = C(p - 1 + v, q, t);
    g[w] = a[x];
  }

  if (m == 0) {
    for (int v = 0; v < (n + 2); v++) {
      int w = C(l - 1 + v, m - 1, u);
      int x = C(p - 1 + v, q - 1, t);
      g[w] = a[x];
    }
  } else if (m == (o - 1)) {
    for (int v = 0; v < (n + 2); v++) {
      int w = C(l - 1 + v, m + 1, u);
      int x = C(p - 1 + v, q + 1, t);
      g[w] = a[x];
    }
  }

  barrier(1);

  for (int v = 0; v < n; v++) {
    int y = C(l + v, m, u);
    int z = C(l - 1 + v, m, u);
    int aa = C(l + 1 + v, m, u);
    int ab = C(l + v, m + 1, u);
    int ac = C(l + v, m - 1, u);
    int ad = C(l - 1 + v, m + 1, u);
    int ae = C(l + 1 + v, m + 1, u);
    int af = C(l - 1 + v, m - 1, u);
    int ag = C(l + 1 + v, m - 1, u);

    float ah = g[y];
    float ai = g[z] + g[aa] + g[ab] + g[ac];
    float aj = g[ad] + g[ae] + g[af] + g[ag];

    b[C(p + v, q, t)] = d * ah + e * ai + f * aj;
  }
}