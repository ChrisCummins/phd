__kernel void A(__global uint *a, __global float *b, float c, float d, uint e) {
  const int f = get_global_id(0);
  const int g = get_global_size(0);

  int h = (get_local_id(0) >> 5);
  const int i = mul24(h, (1024));
  __local unsigned int j[(3 * (1024))];
  int k = 0;

  const uint l = get_local_id(0) << (32 - 5);

  for (uint m = get_local_id(0); m < (3 * (1024)); m += get_local_size(0)) {
    j[m] = 0;
  }

  barrier(1 | 2);
  for (int n = get_global_id(0); n < e; n += get_global_size(0)) {
    uint o = ((b[n] - c) / (d - c)) * (1024);
    Foo(j + i, o & 0x3FFU, l);
  }

  barrier(1 | 2);
  for (int n = get_local_id(0); n < (1024); n += get_local_size(0)) {
    uint p = 0;
    for (int m = 0; m < (3 * (1024)); m += (1024)) {
      p += j[n + m] & 0x07FFFFFFU;
    }
    atomic_add(a + n, p);
  }
}