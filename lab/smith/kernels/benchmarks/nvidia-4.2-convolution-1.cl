__kernel void A(__global float *a, __global float *b, __constant float *c, int d, int e, int f) {
  __local float g[4][(4 + 2 * 1) * 16];

  const int h = (get_group_id(0) * 4 - 1) * 16 + get_local_id(0);
  const int i = get_group_id(1) * 4 + get_local_id(1);

  b += i * f + h;
  a += i * f + h;

  for (int j = 1; j < 1 + 4; j++) g[get_local_id(1)][get_local_id(0) + j * 16] = b[j * 16];

  for (int j = 0; j < 1; j++) g[get_local_id(1)][get_local_id(0) + j * 16] = (h + j * 16 >= 0) ? b[j * 16] : 0;

  for (int j = 1 + 4; j < 1 + 4 + 1; j++) g[get_local_id(1)][get_local_id(0) + j * 16] = (h + j * 16 < d) ? b[j * 16] : 0;

  barrier(1);
  for (int j = 1; j < 1 + 4; j++) {
    float k = 0;

    for (int l = -8; l <= 8; l++) k += c[8 - l] * g[get_local_id(1)][get_local_id(0) + j * 16 + l];

    a[j * 16] = k;
  }
}