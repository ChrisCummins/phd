__kernel void A(__global int* a, __global float* b, __global float* c, __constant float* d, __global float* e, __constant float3* f, __constant float3* g, __constant float3* h, __constant float3* i, int j) {
  const float k = (float)(0.2f);

  const int l = get_global_id(0);
  if (l >= j)
    return;
  int m, n;
  float3 o;
  float p;
  float q;

  float r = c[l + 0 * j];
  float3 s;
  s.x = c[l + (1 + 0) * j];
  s.y = c[l + (1 + 1) * j];
  s.z = c[l + (1 + 2) * j];

  float t = c[l + (1 + 3) * j];

  float3 u;
  B(r, s, &u);
  float v = C(u);

  float w = sqrt(v);
  float x = D(r, t, v);
  float y = E(r, x);
  float3 z, aa, ab;
  float3 ac;
  F(r, s, t, x, u, &z, &aa, &ab, &ac);

  float ad = (float)(0.0f);
  float3 ae;
  ae.x = (float)(0.0f);
  ae.y = (float)(0.0f);
  ae.z = (float)(0.0f);
  float af = (float)(0.0f);

  float3 ag;
  float ah, ai;
  float3 aj;
  float3 ak, al, am;
  float3 an;
  float ao, ap, aq;

  for (m = 0; m < 4; m++) {
    n = a[l + m * j];
    o.x = b[l + (m + 0 * 4) * j];
    o.y = b[l + (m + 1 * 4) * j];
    o.z = b[l + (m + 2 * 4) * j];

    p = sqrt(o.x * o.x + o.y * o.y + o.z * o.z);

    if (n >= 0) {
      ah = c[n + 0 * j];
      aj.x = c[n + (1 + 0) * j];
      aj.y = c[n + (1 + 1) * j];
      aj.z = c[n + (1 + 2) * j];
      ai = c[n + (1 + 3) * j];
      B(ah, aj, &ag);
      ao = C(ag);
      aq = D(ah, ai, ao);
      ap = E(ah, aq);
      F(ah, aj, ai, aq, ag, &ak, &al, &am, &an);

      q = -p * k * (float)(0.5f) * (w + sqrt(ao) + y + ap);
      ad += q * (r - ah);
      af += q * (t - ai);
      ae.x += q * (s.x - aj.x);
      ae.y += q * (s.y - aj.y);
      ae.z += q * (s.z - aj.z);

      q = (float)(0.5f) * o.x;
      ad += q * (aj.x + s.x);
      af += q * (an.x + ac.x);
      ae.x += q * (ak.x + z.x);
      ae.y += q * (al.x + aa.x);
      ae.z += q * (am.x + ab.x);

      q = (float)(0.5f) * o.y;
      ad += q * (aj.y + s.y);
      af += q * (an.y + ac.y);
      ae.x += q * (ak.y + z.y);
      ae.y += q * (al.y + aa.y);
      ae.z += q * (am.y + ab.y);

      q = (float)(0.5f) * o.z;
      ad += q * (aj.z + s.z);
      af += q * (an.z + ac.z);
      ae.x += q * (ak.z + z.z);
      ae.y += q * (al.z + aa.z);
      ae.z += q * (am.z + ab.z);
    } else if (n == -1) {
      ae.x += o.x * x;
      ae.y += o.y * x;
      ae.z += o.z * x;
    } else if (n == -2) {
      q = (float)(0.5f) * o.x;
      ad += q * (d[1 + 0] + s.x);
      af += q * (f[0].x + ac.x);
      ae.x += q * (g[0].x + z.x);
      ae.y += q * (h[0].x + aa.x);
      ae.z += q * (i[0].x + ab.x);

      q = (float)(0.5f) * o.y;
      ad += q * (d[1 + 1] + s.y);
      af += q * (f[0].y + ac.y);
      ae.x += q * (g[0].y + z.y);
      ae.y += q * (h[0].y + aa.y);
      ae.z += q * (i[0].y + ab.y);

      q = (float)(0.5f) * o.z;
      ad += q * (d[1 + 2] + s.z);
      af += q * (f[0].z + ac.z);
      ae.x += q * (g[0].z + z.z);
      ae.y += q * (h[0].z + aa.z);
      ae.z += q * (i[0].z + ab.z);
    }
  }

  e[l + 0 * j] = ad;
  e[l + (1 + 0) * j] = ae.x;
  e[l + (1 + 1) * j] = ae.y;
  e[l + (1 + 2) * j] = ae.z;
  e[l + (1 + 3) * j] = af;
}