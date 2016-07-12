__kernel void B(__global const float* restrict a,

                __global const float* restrict b,

                __global const int* restrict c, __global const int* restrict d,
                const int e, const int m, __global float* restrict f) {
  int h = get_local_id(0);

  int n = h & (m - 1);

  int o = get_local_size(0) / m;
  int g = (get_group_id(0) * o) + (h / m);

  __local volatile float p[128];
  p[h] = 0;

  if (g < e) {
    int q = d[g];
    int r = d[g + 1];
    float s = 0;
    for (int k = q + n; k < r; k += m) {
      int l = c[k];

      s += a[k] * b[l];
    }

    p[h] = s;
    barrier(1);

    int t = m / 2;
    while (t > 0) {
      if (n < t) p[h] += p[h + t];
      barrier(1);
      t = t / 2;
    }

    if (n == 0) {
      f[g] = p[h];
    }
  }
}
