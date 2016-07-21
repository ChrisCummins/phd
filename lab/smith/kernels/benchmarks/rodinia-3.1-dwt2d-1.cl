__kernel void A(__global int *a, __global int *b, __global int *c, __global unsigned char *d, int e) {
  int f = get_local_id(0);
  int g = get_local_size(0) * get_group_id(0);

  __local unsigned char h[256 * 3];

  h[3 * f + 0] = d[g * 3 + 3 * f + 0];
  h[3 * f + 1] = d[g * 3 + 3 * f + 1];
  h[3 * f + 2] = d[g * 3 + 3 * f + 2];

  barrier(1);

  int i, j, k;
  int l = f * 3;
  i = (int)(h[l]);
  j = (int)(h[l + 1]);
  k = (int)(h[l + 2]);

  int m = g + f;
  if (m < e) {
    B(a, b, c, i, j, k, m);
  }
}