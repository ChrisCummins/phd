__kernel void A(__global uint *a, __global uint *b, uint c) {
  __local uint d[8 * 256];
  __local uint *e = d + (get_local_id(0) >> 5) * 256;

  for (uint f = 0; f < (256 / (1U << 5)); f++) d[get_local_id(0) + f * (8 * (1U << 5))] = 0;

  const uint g = get_local_id(0) << (32 - 5);

  barrier(1);
  for (uint h = get_global_id(0); h < c; h += get_global_size(0)) {
    uint i = b[h];
    B(e, i, g);
  }

  barrier(1);
  for (uint h = get_local_id(0); h < 256; h += (8 * (1U << 5))) {
    uint j = 0;

    for (uint f = 0; f < 8; f++) j += d[h + f * 256] & ((1U << (32U - 5)) - 1U);

    a[get_group_id(0) * 256 + h] = j;
  }
}