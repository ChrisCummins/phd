__kernel void A(__global float* a, int b, int c, __global float* d, int e, int f, int g, int h) {
  int i = get_group_id(0);
  int j = get_local_id(0);
  int k = get_global_size(0);
  int l = get_local_size(0);
  int m = i * l + j;

  if (m < h) {
    for (int n = 0; n < g; n++) {
      (a + b)[D(m, n, c)] = (d + e)[D(m, n, f)];
    }
  }
}