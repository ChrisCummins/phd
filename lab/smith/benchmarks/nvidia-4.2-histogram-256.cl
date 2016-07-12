inline void A(volatile __local uint *a, uint b, uint c) {
  uint d;
  do {
    d = a[b] & ((1U << (32U - 5)) - 1U);
    d = c | (d + 1);
    a[b] = d;
  } while (a[b] != d);
}

inline void B(volatile __local uint *a, uint b, uint c) {
  A(a, (b >> 0) & 0xFFU, c);
  A(a, (b >> 8) & 0xFFU, c);
  A(a, (b >> 16) & 0xFFU, c);
  A(a, (b >> 24) & 0xFFU, c);
}

__kernel __attribute__((reqd_work_group_size((8 * (1U << 5)), 1, 1))) void C(
    __global uint *e, __global uint *f, uint g) {
  __local uint h[8 * 256];
  __local uint *a = h + (get_local_id(0) >> 5) * 256;

  for (uint i = 0; i < (256 / (1U << 5)); i++)
    h[get_local_id(0) + i * (8 * (1U << 5))] = 0;

  const uint c = get_local_id(0) << (32 - 5);

  barrier(1);
  for (uint j = get_global_id(0); j < g; j += get_global_size(0)) {
    uint b = f[j];
    B(a, b, c);
  }

  barrier(1);
  for (uint j = get_local_id(0); j < 256; j += (8 * (1U << 5))) {
    uint k = 0;

    for (uint i = 0; i < 8; i++) k += h[j + i * 256] & ((1U << (32U - 5)) - 1U);

    e[get_group_id(0) * 256 + j] = k;
  }
}
__kernel __attribute__((reqd_work_group_size(32, 1, 1))) void D(
    __global uint *l, __global uint *e, uint m) {
  __local uint n[32];

  uint k = 0;
  for (uint i = get_local_id(0); i < m; i += 32)
    k += e[get_group_id(0) + i * 256];
  n[get_local_id(0)] = k;

  for (uint o = 32 / 2; o > 0; o >>= 1) {
    barrier(1);
    if (get_local_id(0) < o) n[get_local_id(0)] += n[get_local_id(0) + o];
  }

  if (get_local_id(0) == 0) l[get_group_id(0)] = n[0];
}