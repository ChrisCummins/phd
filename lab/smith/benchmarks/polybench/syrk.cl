__kernel void A(__global float *a, __global float *b, float c, float d, int e,
                int f) {
  int g = get_global_id(0);
  int h = get_global_id(1);

  if ((h < f) && (g < f)) {
    b[h * f + g] *= d;
    int i;
    for (i = 0; i < e; i++) {
      b[h * f + g] += c * a[h * e + i] * a[g * e + i];
    }
  }
}