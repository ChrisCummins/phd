__kernel void A(__global float *a, __global float *b, __global float *c, int d,
                int e) {
  int f = get_global_id(0);
  int g = get_global_id(1);

  if ((g < d) && (f < e)) {
    c[g * e + f] = c[g * e + f] -
                   0.7 * (a[g * (e + 1) + (f + 1)] - a[g * (e + 1) + f] +
                          b[(g + 1) * e + f] - b[g * e + f]);
  }
}