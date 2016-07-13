__kernel void A(__global float *a, __global float *b, __global int *c, int d,
                int e, int f, int g, int h) {
  unsigned int i = get_global_id(0);
  int j = 0;

  if (i < d) {
    float k = 0x1.fffffep127f;
    for (int l = 0; l < e; l++) {
      float m = 0;
      float n = 0;
      for (int o = 0; o < f; o++) {
        n += (a[o * d + i] - b[l * f + o]) * (a[o * d + i] - b[l * f + o]);
      }

      m = n;
      if (m < k) {
        k = m;
        j = l;
      }
    }

    c[i] = j;
  }

  return;
}