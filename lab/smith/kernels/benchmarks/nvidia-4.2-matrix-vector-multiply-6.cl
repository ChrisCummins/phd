__kernel void A(const __global float* a, const __global float* b, uint c, uint d, __global float* e, __local float* f) {
  for (uint g = get_group_id(0); g < d; g += get_num_groups(0)) {
    const __global float* h = a + g * c;

    float i = 0;
    for (uint j = get_local_id(0); j < c; j += get_local_size(0)) i += h[j] * b[j];

    f[get_local_id(0)] = i;

    barrier(1);

    uint k = get_local_id(0) & (32 - 1);

    float l = 0.0f;
    if (get_local_id(0) < get_local_size(0) / 2) {
      volatile __local float* m = f + 2 * get_local_id(0) - k;
      m[0] += m[32];
      m[0] += m[16];
      m[0] += m[8];
      m[0] += m[4];
      m[0] += m[2];
      m[0] += m[1];
      l = m[0];
    }

    barrier(1);

    if (k == 0)
      f[get_local_id(0) / 32] = l;

    barrier(1);

    uint n = get_local_size(0) / (2 * 32);

    if (get_local_id(0) < n / 2) {
      volatile __local float* m = f + get_local_id(0);
      if (n >= 8)
        m[0] += m[4];
      if (n >= 4)
        m[0] += m[2];
      if (n >= 2)
        m[0] += m[1];
    }

    if (get_local_id(0) == 0)
      e[g] = f[0];

    barrier(1);
  }
}