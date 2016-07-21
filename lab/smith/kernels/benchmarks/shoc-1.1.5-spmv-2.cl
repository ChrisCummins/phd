__kernel void A(__global const float* restrict a, __global const float* restrict b, __global const int* restrict c, __global const int* restrict d, const int e, const int f, __global float* restrict g) {
  int h = get_local_id(0);

  int i = h & (f - 1);

  int j = get_local_size(0) / f;
  int k = (get_group_id(0) * j) + (h / f);

  __local volatile float l[128];
  l[h] = 0;

  if (k < e) {
    int m = d[k];
    int n = d[k + 1];
    float o = 0;
    for (int p = m + i; p < n; p += f) {
      int q = c[p];

      o += a[p] * b[q];
    }

    l[h] = o;
    barrier(1);

    int r = f / 2;
    while (r > 0) {
      if (i < r)
        l[h] += l[h + r];
      barrier(1);
      r = r / 2;
    }

    if (i == 0) {
      g[k] = l[h];
    }
  }
}
