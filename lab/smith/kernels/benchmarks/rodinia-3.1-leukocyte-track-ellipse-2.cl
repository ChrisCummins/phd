__kernel void A(__global float *a, __global float *b, __constant int *c,
                int __constant *d, __constant int *e, float f, float g, float h,
                int i, float j) {
  __local float k[41 * 81];

  __local float l[256];

  int m = get_group_id(0);

  int n = c[m];
  __global float *o = &(a[n]);
  __global float *p = &(b[n]);

  int q = d[m];
  int r = e[m];

  int s = q * r;
  int t = (s + 256 - 1) / 256;

  int u = get_local_id(0);
  int v, w, x;
  for (v = 0; v < t; v++) {
    int y = v * 256 + u;
    if (y < s) k[y] = o[y];
  }
  barrier(1);

  __local int z;
  if (u == 0) z = 0;
  barrier(1);

  const float aa = 1.0f / (float)r;
  const int ab = u % r;
  const int ac = 256 % r;

  float ad = 1.0f / h;

  int ae = 0;
  while ((!z) && (ae < i)) {
    float af = 0.0f;

    int ag = 0, ah = 0;
    x = ab - ac;

    for (v = 0; v < t; v++) {
      ag = w;
      ah = x;

      int y = v * 256;
      w = (u + y) * aa;
      x += ac;
      if (x >= r) x -= r;

      float ai, aj;

      if (w < q) {
        int ak = (w == 0) ? 0 : w - 1;
        int al = (w == q - 1) ? q - 1 : w + 1;
        int am = (x == 0) ? 0 : x - 1;
        int an = (x == r - 1) ? r - 1 : x + 1;

        aj = k[(w * r) + x];
        float ao = k[(ak * r) + x] - aj;
        float ap = k[(al * r) + x] - aj;
        float aq = k[(w * r) + am] - aj;
        float ar = k[(w * r) + an] - aj;
        float as = k[(ak * r) + an] - aj;
        float at = k[(al * r) + an] - aj;
        float au = k[(ak * r) + am] - aj;
        float av = k[(al * r) + am] - aj;

        float aw = A((ao * -g) * ad);
        float ax = A((ap * g) * ad);
        float ay = A((aq * -f) * ad);
        float az = A((ar * f) * ad);
        float ba = A((as * (f - g)) * ad);
        float bb = A((at * (f + g)) * ad);
        float bc = A((au * (-f - g)) * ad);
        float bd = A((av * (-f + g)) * ad);

        ai = aj +
             (0.5f / (8.0f * 0.5f + 1.0f)) *
                 (aw * ao + ax * ap + ay * aq + az * ar + ba * as + bb * at +
                  bc * au + bd * av);

        float be = p[(w * r) + x];
        ai -= ((1.0 / (8.0f * 0.5f + 1.0f)) * be * (ai - be));

        af += __clc_fabs(ai - aj);
      }
      barrier(1);

      if (v > 0) {
        y = (v - 1) * 256;
        if (ag < q) k[(ag * r) + ah] = l[u];
      }
      if (v < t - 1) {
        l[u] = ai;
      } else {
        if (w < q) k[(w * r) + x] = ai;
      }

      barrier(1);
    }

    l[u] = af;
    barrier(1);

    if (u >= 256) {
      l[u - 256] += l[u];
    }
    barrier(1);

    int bf;
    for (bf = 256 / 2; bf > 0; bf /= 2) {
      if (u < bf) {
        l[u] += l[u + bf];
      }
      barrier(1);
    }

    if (u == 0) {
      float bg = l[u] / (float)(q * r);
      if (bg < j) {
        z = 1;
      }
    }

    barrier(1);

    ae++;
  }

  for (v = 0; v < t; v++) {
    int y = v * 256 + u;
    if (y < s) o[y] = k[y];
  }
}