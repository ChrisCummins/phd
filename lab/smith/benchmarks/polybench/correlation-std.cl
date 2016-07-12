__kernel void A(__global float *a, __global float *b, __global float *c,
                float d, float e, int f, int g) {
  int h = get_global_id(0) + 1;

  if ((h >= 1) && (h < (f + 1))) {
    b[h] = 0.0;

    int i;
    for (i = 1; i < (g + 1); i++) {
      b[h] += (c[i * (f + 1) + h] - a[h]) * (c[i * (f + 1) + h] - a[h]);
    }
    b[h] /= d;
    b[h] = sqrt(b[h]);
    if (b[h] <= e) {
      b[h] = 1.0;
    }
  }
}