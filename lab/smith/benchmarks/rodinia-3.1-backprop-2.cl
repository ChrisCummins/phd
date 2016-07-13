__kernel void A(__global float *a, int b, __global float *c, int d,
                __global float *e, __global float *f) {
  int g = get_group_id(1);
  int h = get_local_id(0);
  int i = get_local_id(1);

  int j = (b + 1) * 16 * g + (b + 1) * i + h + 1 + (b + 1);
  int k = 16 * g + i + 1;
  int l = h + 1;

  e[j] += ((0.3f * a[l] * c[k]) + (0.3f * f[j]));
  f[j] = ((0.3f * a[l] * c[k]) + (0.3f * f[j]));

  barrier(1);

  if (i == 0 && g == 0) {
    e[l] += ((0.3f * a[l]) + (0.3f * f[l]));
    f[l] = ((0.3f * a[l]) + (0.3f * f[l]));
  }
}