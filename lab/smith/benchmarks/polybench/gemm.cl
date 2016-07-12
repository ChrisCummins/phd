__kernel void A(__global float *a, __global float *b, __global float *c,
                float d, float e, int f, int g, int h) {
  int i = get_global_id(0);
  int j = get_global_id(1);

  if ((j < f) && (i < g)) {
    c[j * g + i] *= e;
    int k;
    for (k = 0; k < h; k++) {
      c[j * g + i] += d * a[j * h + k] * b[k * g + i];
    }
  }
}