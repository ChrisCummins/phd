__kernel void A(__global float *a, __global float *b, __global float *c, int d,
                int e) {
  int f = get_global_id(0);
  int g = get_global_id(1);

  if ((g < d) && (f < e) && (f > 0)) {
    a[g * (e + 1) + f] =
        a[g * (e + 1) + f] - 0.5 * (c[g * e + f] - c[g * e + (f - 1)]);
  }
}