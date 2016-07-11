__kernel void A(__global uint* a, __global uint2* b, __global uint* c,
                __global uint* d, __global uint* e, uint f, uint g, uint h,
                __local uint2* i) {
  __local uint j[16];
  __local uint k[16];

  __local uint* l = (__local uint*)i;

  uint m = get_group_id(0);

  uint n = get_global_id(0);
  uint o = get_local_id(0);
  uint p = get_local_size(0);

  i[o] = b[n];

  if (o < 16) {
    j[o] = d[o * h + m];
    k[o] = c[m * 16 + o];
  }
  barrier(1);

  uint q = (l[o] >> f) & 0xF;
  uint r = j[q] + o - k[q];

  if (r < g) {
    a[r] = l[o];
  }

  q = (l[o + p] >> f) & 0xF;
  r = j[q] + o + p - k[q];

  if (r < g) {
    a[r] = l[o + p];
  }
}