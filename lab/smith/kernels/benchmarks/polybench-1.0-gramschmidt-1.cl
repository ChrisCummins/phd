__kernel void A(__global float *a, __global float *b, __global float *c, int d,
                int e, int f) {
  int g = get_global_id(0);

  if (g == 0) {
    float h = 0.0;
    int i;
    for (i = 0; i < e; i++) {
      h += a[i * f + d] * a[i * f + d];
    }
    b[d * f + d] = sqrt(h);
  }
}