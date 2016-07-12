__kernel void E(__global float* k, int l, int m, __global float* n, int o,
                int p, int q, int r) {
  int s = get_group_id(0);
  int t = get_local_id(0);
  int u = get_global_size(0);
  int v = get_local_size(0);
  int w = s * v + t;

  if (w < r) {
    for (int x = 0; x < q; x++) {
      (k + l)[D(w, x, m)] = (n + o)[D(w, x, p)];
    }
  }
}
