__kernel void A(__global float *a, __global int *b, __global float *c, __global int *d, int e, int f) {
  if (get_global_id(0) == 0) {
    float g = 0.0;
    int h = -1;
    for (int i = 0; i < e; i++) {
      if (c[i] > g) {
        g = c[i];
        h = i;
      }
    }
    *a = g;

    b[f - 1] = h;
    mem_fence(2);
    for (int j = f - 2; j >= 0; j--) {
      b[j] = d[j * e + b[j + 1]];
      mem_fence(2);
    }
  }
}