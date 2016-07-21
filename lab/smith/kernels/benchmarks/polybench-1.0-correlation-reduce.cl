__kernel void A(__global float *a, __global float *b, __global float *c, float d, int e, int f) {
  int g = get_global_id(0) + 1;
  int h = get_global_id(1) + 1;

  if ((h >= 1) && (h < (f + 1)) && (g >= 1) && (g < (e + 1))) {
    c[h * (e + 1) + g] -= a[g];
    c[h * (e + 1) + g] /= (sqrt(d) * b[g]);
  }
}