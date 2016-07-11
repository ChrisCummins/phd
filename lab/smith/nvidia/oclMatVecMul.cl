__kernel void A(const __global float* a, const __global float* b, uint c,
                uint d, __global float* e) {
  uint f = get_global_id(0);
  if (f < d) {
    const __global float* g = a + f * c;

    float h = 0;
    for (int i = 0; i < c; ++i) h += g[i] * b[i];

    e[f] = h;
  }
}

__kernel void B(const __global float* a, const __global float* b, uint c,
                uint d, __global float* e) {
  for (uint f = get_global_id(0); f < d; f += get_global_size(0)) {
    const __global float* g = a + f * c;

    float h = 0;
    for (uint i = 0; i < c; ++i) h += g[i] * b[i];

    e[f] = h;
  }
}

__kernel void C(const __global float* a, const __global float* b, uint c,
                uint d, __global float* e, __local float* j) {
  for (uint f = get_group_id(0); f < d; f += get_num_groups(0)) {
    const __global float* g = a + f * c;

    float k = 0;
    for (uint i = get_local_id(0); i < c; i += get_local_size(0))
      k += g[i] * b[i];

    j[get_local_id(0)] = k;

    barrier(1);

    if (get_local_id(0) == 0) {
      float h = 0;
      for (uint l = 0; l < get_local_size(0); ++l) h += j[l];
      e[f] = h;
    }

    barrier(1);
  }
}

__kernel void D(const __global float* a, const __global float* b, uint c,
                uint d, __global float* e, __local float* j) {
  for (uint f = get_group_id(0); f < d; f += get_num_groups(0)) {
    const __global float* g = a + f * c;

    float k = 0;
    for (uint i = get_local_id(0); i < c; i += get_local_size(0))
      k += g[i] * b[i];

    j[get_local_id(0)] = k;

    for (uint m = 1; m < get_local_size(0); m *= 2) {
      barrier(1);

      uint n = 2 * m * get_local_id(0);

      if (n < get_local_size(0)) {
        j[n] += j[n + m];
      }
    }

    if (get_local_id(0) == 0) e[f] = j[0];

    barrier(1);
  }
}

__kernel void E(const __global float* a, const __global float* b, uint c,
                uint d, __global float* e, __local float* j) {
  for (uint f = get_group_id(0); f < d; f += get_num_groups(0)) {
    const __global float* g = a + f * c;

    float k = 0;
    for (uint i = get_local_id(0); i < c; i += get_local_size(0))
      k += g[i] * b[i];

    j[get_local_id(0)] = k;

    for (uint m = get_local_size(0) / 2; m > 0; m /= 2) {
      barrier(1);

      if (get_local_id(0) < m) {
        j[get_local_id(0)] += j[get_local_id(0) + m];
      }
    }

    if (get_local_id(0) == 0) e[f] = j[0];

    barrier(1);
  }
}

__kernel void F(const __global float* a, const __global float* b, uint c,
                uint d, __global float* e, __local float* j) {
  for (uint f = get_group_id(0); f < d; f += get_num_groups(0)) {
    const __global float* g = a + f * c;

    float k = 0;
    for (uint i = get_local_id(0); i < c; i += get_local_size(0))
      k += g[i] * b[i];

    j[get_local_id(0)] = k;

    barrier(1);

    uint o = get_local_id(0) & (32 - 1);

    float p = 0.0f;
    if (get_local_id(0) < get_local_size(0) / 2) {
      volatile __local float* q = j + 2 * get_local_id(0) - o;
      q[0] += q[32];
      q[0] += q[16];
      q[0] += q[8];
      q[0] += q[4];
      q[0] += q[2];
      q[0] += q[1];
      p = q[0];
    }

    barrier(1);

    if (o == 0) j[get_local_id(0) / 32] = p;

    barrier(1);

    uint r = get_local_size(0) / (2 * 32);

    if (get_local_id(0) < r / 2) {
      volatile __local float* q = j + get_local_id(0);
      if (r >= 8) q[0] += q[4];
      if (r >= 4) q[0] += q[2];
      if (r >= 2) q[0] += q[1];
    }

    if (get_local_id(0) == 0) e[f] = j[0];

    barrier(1);
  }
}
