__kernel void A(const __global float* a, const __global float* b, uint c,
                uint d, __global float* e, __local float* f) {
  for (uint g = get_group_id(0); g < d; g += get_num_groups(0)) {
    const __global float* h = a + g * c;

    float i = 0;
    for (uint j = get_local_id(0); j < c; j += get_local_size(0))
      i += h[j] * b[j];

    f[get_local_id(0)] = i;

    barrier(1);

    if (get_local_id(0) == 0) {
      float k = 0;
      for (uint l = 0; l < get_local_size(0); ++l) k += f[l];
      e[g] = k;
    }

    barrier(1);
  }
}