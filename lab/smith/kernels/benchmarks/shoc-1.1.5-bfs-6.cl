__kernel void A(__global unsigned int *a, __global unsigned int *b,
                __global unsigned int *c, __global volatile int *d,
                __global volatile int *e, __global volatile int *f,
                __global volatile int *g) {
  unsigned int h = get_global_id(0);

  if (h < *c) {
    a[h] = b[h];
  }
}