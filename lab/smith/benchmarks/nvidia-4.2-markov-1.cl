__kernel void A(__global float *a, __global int *b, __global float *c,
                __global float *d, __global float *e, __local float f[],
                __local int g[], int h, int i, int j) {
  uint k = get_group_id(0) + get_group_id(1) * get_num_groups(0);
  uint l = get_local_id(0);
  uint m = get_local_size(0);

  uint n = k;

  float o = -1.0f;
  int p = -1;
  float q;
  for (int r = l; r < h; r += m) {
    q = c[r] + d[n * h + r];
    if (q > o) {
      o = q;
      p = r;
    }
  }
  f[l] = o;
  g[l] = p;
  barrier(1);

  if (l == 0) {
    a[n] = f[0] + e[i * h + n];
    b[(j - 1) * h + n] = g[0];
  }
}