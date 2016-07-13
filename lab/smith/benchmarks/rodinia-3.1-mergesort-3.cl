__kernel void A(__global float *a, __global float *b, __constant int *c,
                __constant int *d, __constant int *e) {
  int f = get_global_id(0);
  int g = get_group_id(1);

  if ((e[g] + f) >= e[g + 1]) return;
  b[e[g] + f] = a[c[g] * 4 + d[g] + f];
}