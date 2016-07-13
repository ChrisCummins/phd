__kernel void A(__global uint *a, __global float4 *b, uint c) {
  const uint d = ((get_local_id(0) & ~(16 * 4 - 1)) << 0) |
                 ((get_local_id(0) & (16 - 1)) << 2) |
                 ((get_local_id(0) & (16 * 3)) >> 4);

  __local uchar e[32 * 64];
  __local uchar *f = e + d;

  for (uint g = 0; g < (64 / 4); g++) ((__local uint *)e)[d + g * 32] = 0;

  barrier(1);
  for (uint h = get_global_id(0); h < c; h += get_global_size(0)) {
    float4 i = b[h];
    B(f, i.x);
    B(f, i.y);
    B(f, i.z);
    B(f, i.w);
  }

  barrier(1);
  if (get_local_id(0) < 64) {
    __local uchar *j = e + ((get_local_id(0)) * (32));

    uint k = 0;
    uint h = 4 * (get_local_id(0) & (16 - 1));
    for (uint g = 0; g < (32 / 4); g++) {
      k += j[h + 0] + j[h + 1] + j[h + 2] + j[h + 3];
      h = (h + 4) & (32 - 1);
    }

    a[get_group_id(0) * 64 + get_local_id(0)] = k;
  }
}