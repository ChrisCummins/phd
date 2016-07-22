__kernel void A(__global unsigned short *a, __global unsigned short *b, int c, int d, __read_only image2d_t e) {
  const sampler_t f = 0x0000 | 0x0002 | 0x0000;

  int g = (get_local_id(0) / CEIL(1089, 18)) % 1;
  int h = (get_local_id(0) / CEIL(1089, 18)) / 1;
  int i = get_group_id(0);
  int j = get_group_id(1);
  int k = c * 16;

  int l = (g + i * 1) >> 2;
  int m = (h + j * 1) >> 2;
  int n = (g + i * 1) & 0x03;
  int o = (h + j * 1) & 0x03;

  if ((l < c) && (m < d)) {
    int p = ((l << 2) + n) << 2;
    int q = ((m << 2) + o) << 2;

    int r = p - 16;
    int s = q - 16;

    int t = q * k + p;

    int u;
    int v = (get_local_id(0) % CEIL(1089, 18)) * 18;
    int w = v + 18;

    a += c * d * 1096 * (9 + 16) + (m * c + l) * 1096 * 16 + (4 * o + n) * 1096;

    if (w > 1089)
      w = 1089;

    for (u = v; u < w; u++) {
      unsigned short x = 0;
      int y = r + (u % (2 * 16 + 1));
      int z = s + (u / (2 * 16 + 1));

      for (int aa = 0; aa < 4; aa++) {
        for (int ab = 0; ab < 4; ab++) {
          x += abs((unsigned short)((read_imageui(e, f, (int2)(y + ab, z + aa))).x) - b[t + aa * k + ab]);
        }
      }

      a[u] = x;
    }
  }
}