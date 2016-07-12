__kernel void A(__global float *a, __global float *b, int c, int d, int e) {
  unsigned int f = get_global_id(0);
  unsigned int g = get_global_id(1);

  if (f + c < d && g < e) {
    unsigned int h = g + e * (f + c);
    a[h] = b[h];
  }
}