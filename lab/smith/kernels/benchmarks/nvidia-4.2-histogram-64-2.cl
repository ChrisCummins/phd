__kernel void A(__global uint *a, __global uint *b, uint c) {
  __local uint d[32];

  uint e = 0;
  for (uint f = get_local_id(0); f < c; f += 32) e += b[get_group_id(0) + f * 64];
  d[get_local_id(0)] = e;

  for (uint g = 32 / 2; g > 0; g >>= 1) {
    barrier(1);
    if (get_local_id(0) < g)
      d[get_local_id(0)] += d[get_local_id(0) + g];
  }

  if (get_local_id(0) == 0)
    a[get_group_id(0)] = d[0];
}