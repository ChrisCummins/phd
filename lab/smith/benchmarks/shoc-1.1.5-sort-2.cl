__kernel void C(__global uint *b, const int c, __local uint *d) {
  __local int r;
  r = 0;
  barrier(1);

  int s = (get_local_id(0) < c && (get_local_id(0) + 1) == c) ? 1 : 0;

  for (int l = 0; l < 16; l++) {
    uint n = 0;

    if (get_local_id(0) < c) {
      n = b[(c * l) + get_local_id(0)];
    }

    uint t = B(n, d, 1);

    if (get_local_id(0) < c) {
      b[(c * l) + get_local_id(0)] = t + r;
    }
    barrier(1);

    if (s) {
      r += t + n;
    }
    barrier(1);
  }
}
