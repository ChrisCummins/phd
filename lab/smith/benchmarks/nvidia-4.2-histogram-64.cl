typedef uint4 data_t;
inline void A(__local uchar *a, uint b) { a[((b) * (32))]++; }

inline void B(__local uchar *a, uint b) {
  A(a, (b >> 2) & 0x3FU);
  A(a, (b >> 10) & 0x3FU);
  A(a, (b >> 18) & 0x3FU);
  A(a, (b >> 26) & 0x3FU);
}

__kernel __attribute__((reqd_work_group_size(32, 1, 1))) void C(
    __global uint *c, __global data_t *d, uint e) {
  const uint f = ((get_local_id(0) & ~(16 * 4 - 1)) << 0) |
                 ((get_local_id(0) & (16 - 1)) << 2) |
                 ((get_local_id(0) & (16 * 3)) >> 4);

  __local uchar g[32 * 64];
  __local uchar *a = g + f;

  for (uint h = 0; h < (64 / 4); h++) ((__local uint *)g)[f + h * 32] = 0;

  barrier(1);
  for (uint i = get_global_id(0); i < e; i += get_global_size(0)) {
    data_t b = d[i];
    B(a, b.x);
    B(a, b.y);
    B(a, b.z);
    B(a, b.w);
  }

  barrier(1);
  if (get_local_id(0) < 64) {
    __local uchar *j = g + ((get_local_id(0)) * (32));

    uint k = 0;
    uint i = 4 * (get_local_id(0) & (16 - 1));
    for (uint h = 0; h < (32 / 4); h++) {
      k += j[i + 0] + j[i + 1] + j[i + 2] + j[i + 3];
      i = (i + 4) & (32 - 1);
    }

    c[get_group_id(0) * 64 + get_local_id(0)] = k;
  }
}
__kernel __attribute__((reqd_work_group_size(32, 1, 1))) void D(
    __global uint *l, __global uint *c, uint m) {
  __local uint n[32];

  uint k = 0;
  for (uint h = get_local_id(0); h < m; h += 32)
    k += c[get_group_id(0) + h * 64];
  n[get_local_id(0)] = k;

  for (uint o = 32 / 2; o > 0; o >>= 1) {
    barrier(1);
    if (get_local_id(0) < o) n[get_local_id(0)] += n[get_local_id(0) + o];
  }

  if (get_local_id(0) == 0) l[get_group_id(0)] = n[0];
}