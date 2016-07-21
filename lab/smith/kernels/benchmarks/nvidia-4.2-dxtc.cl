__kernel void A(__global const uint* a, __global const uint* b, __global uint2* c, __constant float* d, __constant int* e, __constant float* f, __constant int* g, int h) {
  __local float4 i[16];
  __local float4 j[16];
  __local int k[64];
  __local float l[16 * 6];
  __local uint m[160];
  __local int n[16];

  const int o = get_local_id(0);

  E(b, i, j, n, l, h);

  barrier(1);

  uint4 p = I(i, a, l, j[0], m, d, e, f, g);

  const int q = J(l, k);

  barrier(1);

  if (o == q) {
    K(p.x, p.y, p.z, n, c, h);
  }
}