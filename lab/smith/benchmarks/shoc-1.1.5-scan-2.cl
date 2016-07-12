__kernel void C(__global float *b, const int c, __local float *d) {
  float l = get_local_id(0) < c ? b[get_local_id(0)] : 0.0f;
  l = B(l, d, 1);

  if (get_local_id(0) < c) {
    b[get_local_id(0)] = l;
  }
}
