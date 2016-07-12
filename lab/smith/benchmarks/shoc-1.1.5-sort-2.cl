__kernel void A(__global uint *a, const int b, __local uint *c) {
  __local int d;
  d = 0;
  barrier(1);

  int e = (get_local_id(0) < b && (get_local_id(0) + 1) == b) ? 1 : 0;

  for (int f = 0; f < 16; f++) {
    uint g = 0;

    if (get_local_id(0) < b) {
      g = a[(b * f) + get_local_id(0)];
    }

    uint h = B(g, c, 1);

    if (get_local_id(0) < b) {
      a[(b * f) + get_local_id(0)] = h + d;
    }
    barrier(1);

    if (e) {
      d += h + g;
    }
    barrier(1);
  }
}