__kernel void A(__global float *a, __global float *b, __global float *c, __global float *d, __global float *e, float f, float g, int h) {
  int i = get_global_id(0);

  if (i < h) {
    int j;
    for (j = 0; j < h; j++) {
      e[i] += a[i * h + j] * c[j];
      d[i] += b[i * h + j] * c[j];
    }
    d[i] = f * e[i] + g * d[i];
  }
}