__kernel void A(__global float *a, __global float *b, __global int *c,
                __global int *d, __global float *e, const int f,
                __constant int *g, __constant int *h) {
  int i = get_global_id(0);

  if (i < f) {
    float j = 0.0f;

    int k = h[i / 32];

    for (int l = 0; l < k; l++) {
      int m = g[l] + i;
      int n = c[m];

      float o = b[m];
      float p = e[n];

      j += o * p;
    }

    a[d[i]] = j;
  }
}