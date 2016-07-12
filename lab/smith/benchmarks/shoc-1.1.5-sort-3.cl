__kernel void D(__global const uint *a, __global const uint *b,
                __global uint *u, const int c, __local uint *d, const int e) {
  __local uint v[16];

  __local uint w[16];

  __private int x[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  __global uint4 *y = (__global uint4 *)a;
  __global uint4 *z = (__global uint4 *)u;
  int aa = c / 4;

  int f = aa / get_num_groups(0);
  int g = get_group_id(0) * f;

  int h = (get_group_id(0) == get_num_groups(0) - 1) ? aa : g + f;

  int j = g + get_local_id(0);
  int ab = g;

  if (get_local_id(0) < 16) {
    w[get_local_id(0)] = 0;
    v[get_local_id(0)] =
        b[(get_local_id(0) * get_num_groups(0)) + get_group_id(0)];
  }
  barrier(1);

  while (ab < h) {
    for (int ac = 0; ac < 16; ac++) x[ac] = 0;
    uint4 ad;
    uint4 ae;

    if (j < h) {
      ad = y[j];

      ae.x = (ad.x >> e) & 0xFU;
      ae.y = (ad.y >> e) & 0xFU;
      ae.z = (ad.z >> e) & 0xFU;
      ae.w = (ad.w >> e) & 0xFU;

      x[ae.x]++;
      x[ae.y]++;
      x[ae.z]++;
      x[ae.w]++;
    }

    for (int af = 0; af < 16; af++) {
      x[af] = B(x[af], d, 1);
      barrier(1);
    }

    if (j < h) {
      int ag;
      ag = x[ae.x] + v[ae.x] + w[ae.x];
      u[ag] = ad.x;
      x[ae.x]++;

      ag = x[ae.y] + v[ae.y] + w[ae.y];
      u[ag] = ad.y;
      x[ae.y]++;

      ag = x[ae.z] + v[ae.z] + w[ae.z];
      u[ag] = ad.z;
      x[ae.z]++;

      ag = x[ae.w] + v[ae.w] + w[ae.w];
      u[ag] = ad.w;
      x[ae.w]++;
    }

    barrier(1);

    if (get_local_id(0) == get_local_size(0) - 1) {
      for (int ac = 0; ac < 16; ac++) {
        w[ac] += x[ac];
      }
    }
    barrier(1);

    ab += get_local_size(0);
    j += get_local_size(0);
  }
}
