__kernel void A(__global float* a, int b) {
  int c = get_global_id(0);
  size_t d = get_local_size(0);

  if (c == 0) {
    int e;
    float f = 0;
    int g = __clc_ceil((float)b / (float)d);
    for (e = 0; e < g; e++) {
      f += a[e];
    }
    a[0] = f;
  }
}