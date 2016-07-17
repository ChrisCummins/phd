__kernel void A(float a, int b, int c, long d, __global int* e, __global int* f,
                __global int* g, __global int* h, __global float* i,
                __global float* j, __global float* k, __global float* l,
                float m, __global float* n, __global float* o) {
  int p = get_group_id(0);
  int q = get_local_id(0);
  int r = p * 256 + q;
  int s;
  int t;

  float u;
  float v, w, x, y;
  float z;
  float aa, ab, ac, ad, ae;

  s = (r + 1) % b - 1;
  t = (r + 1) / b + 1 - 1;
  if ((r + 1) % b == 0) {
    s = b - 1;
    t = t - 1;
  }

  if (r < d) {
    u = o[r];

    v = o[e[s] + b * t] - u;
    w = o[f[s] + b * t] - u;
    x = o[s + b * h[t]] - u;
    y = o[s + b * g[t]] - u;

    aa = (v * v + w * w + x * x + y * y) / (u * u);

    ab = (v + w + x + y) / u;

    ac = (0.5 * aa) - ((1.0 / 16.0) * (ab * ab));
    ad = 1 + (0.25 * ab);
    ae = ac / (ad * ad);

    ad = (ae - m) / (m * (1 + m));
    z = 1.0 / (1.0 + ad);

    if (z < 0) {
      z = 0;
    } else if (z > 1) {
      z = 1;
    }

    i[r] = v;
    j[r] = w;
    l[r] = x;
    k[r] = y;
    n[r] = z;
  }
}