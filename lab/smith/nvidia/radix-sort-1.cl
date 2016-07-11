__kernel void A(__global uint4* a, __global uint4* b, uint c, uint d, uint e,
                uint f, __local uint* g) {
  int h = get_global_id(0);
  __local uint i[1];

  uint4 j;
  j = a[h];

  barrier(1);

  D(&j, c, d, g, i);

  b[h] = j;
}