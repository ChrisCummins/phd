__kernel void A(__global float *a, __global float *b, __constant int *c, int __constant *d, __constant int *e, float f, float g, float h, int i, float j) {
  __local float k[41 * 81];

  __local float l[256];

  int m = get_group_id(0);

  int n = c[m];
  __global float *o = &(a[n]);
  __global float *p = &(b[n]);

  int q = d[m];
  int r = e[m];

  int s = (q * r + 256 - 1) / 256;

  int t = get_local_id(0);
  int u, v, w;
  for (u = 0; u < s; u++) {
    int x = u * 256;
    v = (t + x) / r;
    w = (t + x) % r;
    if (v < q)
      k[(v * r) + w] = o[(v * r) + w];
  }
  barrier(1);

  __local int y;
  if (t == 0)
    y = 0;
  barrier(1);

  const float z = 1.0f / (float)r;
  const int aa = t % r;
  const int ab = 256 % r;

  float ac = 1.0f / h;

  int ad = 0;
  while ((!y) && (ad < i)) {
    float ae = 0.0f;

    int af = 0, ag = 0;
    w = aa - ab;

    for (u = 0; u < s; u++) {
      af = v;
      ag = w;

      int x = u * 256;
      v = (t + x) * z;
      w += ab;
      if (w >= r)
        w -= r;

      float ah = 0.0f, ai = 0.0f;

      if (v < q) {
        int aj = (v == 0) ? 0 : v - 1;
        int ak = (v == q - 1) ? q - 1 : v + 1;
        int al = (w == 0) ? 0 : w - 1;
        int am = (w == r - 1) ? r - 1 : w + 1;

        ai = k[(v * r) + w];
        float an = k[(aj * r) + w] - ai;
        float ao = k[(ak * r) + w] - ai;
        float ap = k[(v * r) + al] - ai;
        float aq = k[(v * r) + am] - ai;
        float ar = k[(aj * r) + am] - ai;
        float as = k[(ak * r) + am] - ai;
        float at = k[(aj * r) + al] - ai;
        float au = k[(ak * r) + al] - ai;

        float av = An((an * -g) * ac);
        float aw = An((ao * g) * ac);
        float ax = An((ap * -f) * ac);
        float ay = An((aq * f) * ac);
        float az = An((ar * (f - g)) * ac);
        float ba = An((as * (f + g)) * ac);
        float bb = An((at * (-f - g)) * ac);
        float bc = An((au * (-f + g)) * ac);

        ah = ai + (0.5f / (8.0f * 0.5f + 1.0f)) * (av * an + aw * ao + ax * ap + ay * aq + az * ar + ba * as + bb * at + bc * au);

        float bd = p[(v * r) + w];
        ah -= ((1.0f / (8.0f * 0.5f + 1.0f)) * bd * (ah - bd));
      }

      if (u > 0) {
        x = (u - 1) * 256;
        if (af < q)
          k[(af * r) + ag] = l[t];
      }
      if (u < s - 1) {
        l[t] = ah;
      } else {
        if (v < q)
          k[(v * r) + w] = ah;
      }

      ae += __clc_fabs(ah - ai);

      barrier(1);
    }

    l[t] = ae;
    barrier(1);

    if (t >= 256) {
      l[t - 256] += l[t];
    }
    barrier(1);

    int be;
    for (be = 256 / 2; be > 0; be /= 2) {
      if (t < be) {
        l[t] += l[t + be];
      }
      barrier(1);
    }

    if (t == 0) {
      float bf = l[t] / (float)(q * r);
      if (bf < j) {
        y = 1;
      }
    }

    barrier(1);

    ad++;
  }

  for (u = 0; u < s; u++) {
    int x = u * 256;
    v = (t + x) / r;
    w = (t + x) % r;
    if (v < q)
      o[(v * r) + w] = k[(v * r) + w];
  }
}