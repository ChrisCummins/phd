__kernel void A(int a, global float *b, global float *c, global float *d, int e,
                int f, int g, int h, float i, float j, float k, float l,
                float m) {
  local float n[64][64];
  local float o[64][64];
  local float p[64][64];

  float q = 80.0f;
  float r;
  float s, t, u;

  int v = get_group_id(0);
  int w = get_group_id(1);

  int x = get_local_id(0);
  int y = get_local_id(1);

  r = m / i;

  s = 1 / j;
  t = 1 / k;
  u = 1 / l;

  int z = 64 - a * 2;
  int aa = 64 - a * 2;

  int ab = z * w - h;
  int ac = aa * v - g;
  int ad = ab + 64 - 1;
  int ae = ac + 64 - 1;

  int af = ab + y;
  int ag = ac + x;

  int ah = af, ai = ag;
  int aj = e * ah + ai;

  if (((ah) >= (0) && (ah) <= (f - 1)) && ((ai) >= (0) && (ai) <= (e - 1))) {
    n[y][x] = c[aj];
    o[y][x] = b[aj];
  }
  barrier(1);

  int ak = (ab < 0) ? -ab : 0;
  int al = (ad > f - 1) ? 64 - 1 - (ad - f + 1) : 64 - 1;
  int am = (ac < 0) ? -ac : 0;
  int an = (ae > e - 1) ? 64 - 1 - (ae - e + 1) : 64 - 1;

  int ao = y - 1;
  int ap = y + 1;
  int aq = x - 1;
  int ar = x + 1;

  ao = (ao < ak) ? ak : ao;
  ap = (ap > al) ? al : ap;
  aq = (aq < am) ? am : aq;
  ar = (ar > an) ? an : ar;

  bool as;
  for (int at = 0; at < a; at++) {
    as = false;
    if (((x) >= (at + 1) && (x) <= (64 - at - 2)) &&
        ((y) >= (at + 1) && (y) <= (64 - at - 2)) &&
        ((x) >= (am) && (x) <= (an)) && ((y) >= (ak) && (y) <= (al))) {
      as = true;
      p[y][x] =
          n[y][x] +
          r * (o[y][x] + (n[ap][x] + n[ao][x] - 2.0f * n[y][x]) * t +
               (n[y][ar] + n[y][aq] - 2.0f * n[y][x]) * s + (q - n[y][x]) * u);
    }
    barrier(1);

    if (at == a - 1) break;
    if (as) n[y][x] = p[y][x];

    barrier(1);
  }

  if (as) {
    d[aj] = p[y][x];
  }
}