__kernel void A(__global uint4 *a, __global uint *b) {
  __local uint c[1];

  uint4 d = a[get_global_id(0)];

  if (get_local_id(0) == 0) c[0] = b[get_group_id(0)];

  barrier(1);
  d += (uint4)c[0];
  a[get_global_id(0)] = d;
}