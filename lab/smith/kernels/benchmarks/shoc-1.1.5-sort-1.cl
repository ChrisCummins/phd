__kernel void A(__global const uint *a, __global uint *b, const int c,
                __local uint *d, const int e) {
  int f = ((c / 4) / get_num_groups(0)) * 4;
  int g = get_group_id(0) * f;

  int h = (get_group_id(0) == get_num_groups(0) - 1) ? c : g + f;

  int i = get_local_id(0);
  int j = g + i;

  int k[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  while (j < h) {
    k[(a[j] >> e) & 0xFU]++;
    j += get_local_size(0);
  }

  for (int l = 0; l < 16; l++) {
    d[i] = k[l];
    barrier(1);

    for (unsigned int m = get_local_size(0) / 2; m > 0; m >>= 1) {
      if (i < m) {
        d[i] += d[i + m];
      }
      barrier(1);
    }

    if (i == 0) {
      b[(l * get_num_groups(0)) + get_group_id(0)] = d[0];
    }
  }
}