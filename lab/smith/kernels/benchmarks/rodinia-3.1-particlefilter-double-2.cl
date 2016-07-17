__kernel void A(__global double* a, int b, __global double* c,
                __global double* d, __global double* e, __global int* f) {
  int g = get_global_id(0);
  int h = get_local_id(0);
  __local double i;
  __local double j;

  if (0 == h) j = c[0];

  barrier(1);

  if (g < b) {
    a[g] = a[g] / j;
  }

  barrier(2);

  if (g == 0) {
    cdfCalc(d, a, b);
    e[0] = (1 / ((double)(b))) * d_randu(f, g);
  }

  barrier(2);

  if (0 == h) i = e[0];

  barrier(1);

  if (g < b) {
    e[g] = i + g / ((double)(b));
  }
}