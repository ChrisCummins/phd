__kernel void A(__global float* a, __global float* b, __global float* c,
                __global float* d, __global float* e, __global float* f,
                __global float* g, int h) {
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