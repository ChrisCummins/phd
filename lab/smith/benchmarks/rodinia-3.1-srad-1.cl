__kernel void A(long a, __global float* b) {
  int c = get_group_id(0);
  int d = get_local_id(0);
  int e = (c * 256) + d;

  if (e < a) {
    b[e] = exp(b[e] / 255);
  }
}