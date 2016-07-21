__kernel void A(

    __global unsigned int *a, __global volatile int *b, __global volatile int *c, __global volatile int *d, __global volatile int *e) {
  b[0] = 0;
  c[0] = 0;
  *a = 0;
  *d = 0;
  e[0] = 0;
}