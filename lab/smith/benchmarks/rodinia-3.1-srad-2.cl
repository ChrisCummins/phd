__kernel void A(long a, __global float* b, __global float* c,
                __global float* d) {
  int e = get_group_id(0);
  int f = get_local_id(0);
  int g = (e * 256) + f;

  if (g < a) {
    c[g] = b[g];
    d[g] = b[g] * b[g];
  }
}