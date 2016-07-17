__kernel void A(__global float* a, int b, __global float* c, __global float* d,
                __global float* e, __global int* f) {
  int g = get_global_id(0);
  int h = get_local_id(0);
  __local float i;
  __local float j;

  if (0 == h) j = c[0];

  barrier(1);

  if (g < b) {
    a[g] = a[g] / j;
  }

  barrier(2);

  if (g == 0) {
    C(d, a, b);
    e[0] = (1 / ((float)(b))) * D(f, g);
  }

  barrier(2);

  if (0 == h) i = e[0];

  barrier(1);

  if (g < b) {
    e[g] = i + g / ((float)(b));
  }
}