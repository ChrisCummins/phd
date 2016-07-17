__kernel void A(__global double* a, __global double* b, __global double* c,
                __global double* d, __global double* e, __global double* f,
                int g) {
  int h = get_global_id(0);

  if (h < g) {
    int i = -1;
    int j;

    for (j = 0; j < g; j++) {
      if (c[j] >= d[h]) {
        i = j;
        break;
      }
    }
    if (i == -1) {
      i = g - 1;
    }

    e[h] = a[i];
    f[h] = b[i];
  }
}