__kernel void A(const __global float* a, const __global float* b, uint c,
                uint d, __global float* e, __local float* f) {
  for (uint g = get_group_id(0); g < d; g += get_num_groups(0)) {
    const __global float* h = a + g * c;

    float i = 0;
    for (uint j = get_local_id(0); j < c; j += get_local_size(0))
      i += h[j] * b[j];

    f[get_local_id(0)] = i;

    for (uint k = 1; k < get_local_size(0); k *= 2) {
      barrier(1);

      uint l = 2 * k * get_local_id(0);

      if (l < get_local_size(0)) {
        f[l] += f[l + k];
      }
    }

    if (get_local_id(0) == 0) e[g] = f[0];

    barrier(1);
  }
}