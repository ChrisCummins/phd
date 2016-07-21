__kernel void A(__global const float *a, __global float *b, const int c, __local float *d) {
  int e = ((c / 4) / get_num_groups(0)) * 4;
  int f = get_group_id(0) * e;

  int g = (get_group_id(0) == get_num_groups(0) - 1) ? c : f + e;

  int h = get_local_id(0);
  int i = f + h;

  float j = 0.0f;

  while (i < g) {
    j += a[i];
    i += get_local_size(0);
  }

  d[h] = j;
  barrier(1);

  for (unsigned int k = get_local_size(0) / 2; k > 0; k >>= 1) {
    if (h < k) {
      d[h] += d[h + k];
    }
    barrier(1);
  }

  barrier(1);

  if (h == 0) {
    b[get_group_id(0)] = d[0];
  }
}