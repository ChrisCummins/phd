__kernel void A(__global uint4 *a, __global uint4 *b, __local uint *c, uint d) {
  uint4 e = b[get_global_id(0)];

  uint4 f = D(e, c, d);

  a[get_global_id(0)] = f;
}