__kernel void A(const __global float* a, const __global float* b, uint c,
                uint d, __global float* e, __local float* f) {
  for (uint g = get_group_id(0); g < d; g += get_num_groups(0)) {
    const __global float* h = a + g * c;

    float i = 0;
    for (uint j = get_local_id(0); j < c; j += get_local_size(0))
      i += h[j] * b[j];

    f[get_local_id(0)] = i;

    for (uint k = get_local_size(0) / 2; k > 0; k /= 2) {
      barrier(1);

      if (get_local_id(0) < k) {
        f[get_local_id(0)] += f[get_local_id(0) + k];
      }
    }

    if (get_local_id(0) == 0) e[g] = f[0];

    barrier(1);
  }
}