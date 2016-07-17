__kernel void A(global float *a, global int *b, __global float *c, const int d,
                global uint *e, global uint *f) {
  volatile __local unsigned int g[((1 << 10) * (1))];

  int h = get_group_id(0) * ((1 << 10) * (1));
  const int i = (get_local_id(0) >> (5)) * (1 << 10);
  const int j = get_global_size(0);

  for (int k = get_local_id(0); k < ((1 << 10) * (1)); k += get_local_size(0)) {
    g[k] = f[k & ((1 << 10) - 1)] + e[h + k];
  }

  barrier(1 | 2);

  for (int l = get_global_id(0); l < d; l += j) {
    float m = a[l];
    int n = b[l];
    c[g[i + (n & ((1 << 10) - 1))] + (n >> (10))] = m;
    int o = g[i + (n & ((1 << 10) - 1))] + (n >> (10));
  }
}