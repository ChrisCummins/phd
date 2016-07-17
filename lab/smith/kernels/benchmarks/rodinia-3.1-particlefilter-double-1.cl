__kernel void A(__global double* a, __global double* b, __global double* c,
                __global double* d, __global double* e, __global double* f,
                __global double* g, int h) {
  int i = get_global_id(0);

  if (i < h) {
    int j = -1;
    int k;

    for (k = 0; k < h; k++) {
      if (c[k] >= d[i]) {
        j = k;
        break;
      }
    }
    if (j == -1) {
      j = h - 1;
    }

    e[i] = a[j];
    f[i] = b[j];
  }
  barrier(2);
}