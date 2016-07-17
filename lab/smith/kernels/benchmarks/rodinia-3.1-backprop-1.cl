__kernel void A(__global float *a, __global float *b, __global float *c,
                __global float *d, __local float *e, __local float *f, int g,
                int h) {
  int i = get_group_id(1);
  int j = get_local_id(0);
  int k = get_local_id(1);

  int l = (h + 1) * 16 * i + (h + 1) * k + j + 1 + (h + 1);

  int m = 16 * i + k + 1;

  if (j == 0) e[k] = a[m];
  barrier(1);

  f[k * 16 + j] = c[l];
  barrier(1);

  f[k * 16 + j] = f[k * 16 + j] * e[k];
  barrier(1);

  for (int n = 1; n <= 16; n = n * 2) {
    int o = n;

    if (k % o == 0) f[k * 16 + j] = f[k * 16 + j] + f[(k + o / 2) * 16 + j];

    barrier(1);
  }

  c[l] = f[k * 16 + j];

  barrier(1);

  if (j == 0) {
    d[i * h + k] = f[j * 16 + k];
  }
}