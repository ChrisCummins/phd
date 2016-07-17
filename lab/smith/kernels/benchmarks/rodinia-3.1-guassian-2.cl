__kernel void A(__global float *a, __global float *b, __global float *c,
                const int d, const int e) {
  int f = get_global_id(0);

  int g = get_global_id(0);
  int h = get_global_id(1);
  if (g < d - 1 - e && h < d - e) {
    b[d * (g + 1 + e) + (h + e)] -= a[d * (g + 1 + e) + e] * b[d * e + (h + e)];

    if (h == 0) {
      c[g + 1 + e] -= a[d * (g + 1 + e) + (h + e)] * c[e];
    }
  }
}