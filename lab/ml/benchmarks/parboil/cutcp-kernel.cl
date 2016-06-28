typedef int4 xyz;

__kernel void A(int a, int b, __global float4 *c, int d, float e, float f,
                float g, __global float *h, int i, __constant int *j,
                __constant xyz *k) {
  __global float4 *l = c + d;

  __local float m[32 * 8 * 4];
  __global float *n;
  __local xyz o;

  const int p = (get_local_id(2) * 8 + get_local_id(1)) * 8 + get_local_id(0);

  int q;

  n = h +
      ((i * get_num_groups(1) + get_group_id(1)) * (get_num_groups(0) >> 2) +
       (get_group_id(0) >> 2)) *
          512 +
      (get_group_id(0) & 3) * 128;

  float r = (8 * (get_group_id(0) >> 2) + get_local_id(0)) * e;
  float s = (8 * get_group_id(1) + get_local_id(1)) * e;
  float t = (8 * i + 2 * (get_group_id(0) & 3) + get_local_id(2)) * e;

  int u = 0;
  int v;

  o.x = (int)__clc_floor((8 * (get_group_id(0) >> 2) + 4) * e * (1.f / 4.f));
  o.y = (int)__clc_floor((8 * get_group_id(1) + 4) * e * (1.f / 4.f));
  o.z = (int)__clc_floor((8 * i + 4) * e * (1.f / 4.f));

  q = (p >> 4);

  v = 32;

  float w = 0.f;
  for (u = 0; u < *j; u += v) {
    int x;

    int y = 32 * (p >> 4);

    for (x = 0; x < 4 && q < *j; x++, q += 8) {
      int z = o.x + k[q].x;
      int aa = o.y + k[q].y;
      int ab = o.z + k[q].z;

      __global float *ac =
          ((__global float *)l) + (((ab * b) + aa) * a + z) * 32;

      int ad = p & 15;
      int ae = y + x * 8 * 32;

      m[ae + ad] = ac[ad];
      m[ae + ad + 16] = ac[ad + 16];
    }
    barrier(1 | 2);

    if (u + 32 > *j) {
      v = *j - u;
    }

    for (x = 0; x < v; x++) {
      int z;
      float af;

      for (z = 0; z < 8; z++) {
        float ag = m[x * 32 + z * 4];
        float ah = m[x * 32 + z * 4 + 1];
        float ai = m[x * 32 + z * 4 + 2];
        float aj = m[x * 32 + z * 4 + 3];
        if (0.f == aj) break;
        af = (ag - r) * (ag - r) + (ah - s) * (ah - s) + (ai - t) * (ai - t);
        if (af < f) {
          float ak = (1.f - af * g);
          w += aj * (1.f / sqrt(af)) * ak * ak;
        }
      }
    }
    barrier(1 | 2);
  }

  n[p] = w;
}