__kernel void A(global float *a, global int *b, global uint *c, const int d, global float *e) {
  volatile __local uint f[((1 << 10) * (1))];

  const uint g = get_local_id(0) << (32 - (5));
  const int h = (get_local_id(0) >> (5)) * (1 << 10);
  const int i = get_global_size(0);
  for (int j = get_local_id(0); j < ((1 << 10) * (1)); j += get_local_size(0)) f[j] = 0;

  barrier(1 | 2);

  for (int k = get_global_id(0); k < d; k += i) {
    float l = a[k];

    int m = (1 << 10) / 2 - 1;
    int n = (1 << 10) / 4;
    float o = e[m];

    while (n >= 1) {
      m = (l < o) ? (m - n) : (m + n);
      o = e[m];
      n /= 2;
    }
    m = (l < o) ? m : (m + 1);

    b[k] = (Foo(f + h, m, g) << (10)) + m;
  }

  barrier(1 | 2);

  int p = get_group_id(0) * ((1 << 10) * (1));

  for (int j = get_local_id(0); j < ((1 << 10) * (1)); j += get_local_size(0)) c[p + j] = f[j] & 0x07FFFFFFU;
}