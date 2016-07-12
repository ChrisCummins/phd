__kernel void A(__global float *a, __global float *b, __global float *c,
                __global float *d, int e, int f, int g) {
  int h = get_global_id(0);
  int i = get_global_id(1);

  if ((i < f) && (h < g)) {
    int j = i * g + h;

    if (i == 0) {
      c[i * g + h] = a[e];
    } else {
      c[i * g + h] = c[i * g + h] - 0.5 * (d[i * g + h] - d[(i - 1) * g + h]);
    }
  }
}