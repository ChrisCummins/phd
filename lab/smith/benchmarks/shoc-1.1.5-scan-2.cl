__kernel void A(__global float *a, const int b, __local float *c) {
  float d = get_local_id(0) < b ? a[get_local_id(0)] : 0.0f;
  d = B(d, c, 1);

  if (get_local_id(0) < b) {
    a[get_local_id(0)] = d;
  }
}