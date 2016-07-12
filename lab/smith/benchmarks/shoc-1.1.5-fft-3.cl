__kernel void A(__global float2 *a, int b, __global int *c) {
  int d, e = get_local_id(0);
  int f = get_group_id(0) * 512 + e;
  float2 g[8], h[8];

  a = a + f;

  for (d = 0; d < 8; d++) {
    g[d] = a[d * 64];
  }

  for (d = 0; d < 8; d++) {
    h[d] = a[b + d * 64];
  }

  for (d = 0; d < 8; d++) {
    if (g[d].x != h[d].x || g[d].y != h[d].y) {
      *c = 1;
    }
  }
}