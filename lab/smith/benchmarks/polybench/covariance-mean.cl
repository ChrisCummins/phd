__kernel void A(__global float *a, __global float *b, float c, int d, int e) {
  int f = get_global_id(0) + 1;

  if ((f >= 1) && (f < (d + 1))) {
    a[f] = 0.0;

    int g;
    for (g = 1; g < (e + 1); g++) {
      a[f] += b[g * (d + 1) + f];
    }
    a[f] /= (float)c;
  }
}