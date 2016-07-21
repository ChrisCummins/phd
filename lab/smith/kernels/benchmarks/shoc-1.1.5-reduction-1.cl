__kernel void A(__global const float *a, __global float *b, __local float *c, const unsigned int d) {
  const unsigned int e = get_local_id(0);
  unsigned int f = (get_group_id(0) * (get_local_size(0) * 2)) + e;
  const unsigned int g = get_local_size(0) * 2 * get_num_groups(0);
  const unsigned int h = get_local_size(0);

  c[e] = 0;

  while (f < d) {
    c[e] += a[f] + a[f + h];
    f += g;
  }
  barrier(1);

  for (unsigned int i = h / 2; i > 0; i >>= 1) {
    if (e < i) {
      c[e] += c[e + i];
    }
    barrier(1);
  }

  if (e == 0) {
    b[get_group_id(0)] = c[0];
  }
}