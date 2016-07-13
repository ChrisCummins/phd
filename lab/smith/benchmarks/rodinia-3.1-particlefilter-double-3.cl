__kernel void A(__global double* a, int b) {
  int c = get_global_id(0);
  size_t d = get_local_size(0);

  if (c == 0) {
    int e;
    double f = 0;
    int g = (double)b;
    for (e = 0; e < g; e++) {
      f += a[e];
    }
    a[0] = f;
  }
}