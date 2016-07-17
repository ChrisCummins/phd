__kernel void A(__global const float* a, __global const float* b,
                __global float* c, __global float* d, __global const float* e,
                const float f) {
  const float g = a[get_global_id(0)] * f;
  const float h = log((g));
  float i = 0.0f;
  float j, k, l, m, n, o;
  float p, q, r, s;

  const float t = 0x1.0p-126f;

  for (unsigned int u = 1; u <= 22; u++) {
    i += (b[(((u)-1) * (8)) + (get_global_id(0))]);
  }

  float v = i - (b[(((1) - 1) * (8)) + (get_global_id(0))]) -
            (b[(((6) - 1) * (8)) + (get_global_id(0))]) +
            (b[(((10) - 1) * (8)) + (get_global_id(0))]) -
            (b[(((12) - 1) * (8)) + (get_global_id(0))]) +
            2.e0 * (b[(((16) - 1) * (8)) + (get_global_id(0))]) +
            2.e0 * (b[(((14) - 1) * (8)) + (get_global_id(0))]) +
            2.e0 * (b[(((15) - 1) * (8)) + (get_global_id(0))]);
  float w = i - 2.7e-1f * (b[(((1) - 1) * (8)) + (get_global_id(0))]) +
            2.65e0f * (b[(((6) - 1) * (8)) + (get_global_id(0))]) +
            (b[(((10) - 1) * (8)) + (get_global_id(0))]) +
            2.e0 * (b[(((16) - 1) * (8)) + (get_global_id(0))]) +
            2.e0 * (b[(((14) - 1) * (8)) + (get_global_id(0))]) +
            2.e0 * (b[(((15) - 1) * (8)) + (get_global_id(0))]);
  float x = i + (b[(((1) - 1) * (8)) + (get_global_id(0))]) +
            5.e0 * (b[(((6) - 1) * (8)) + (get_global_id(0))]) +
            (b[(((10) - 1) * (8)) + (get_global_id(0))]) +
            5.e-1 * (b[(((11) - 1) * (8)) + (get_global_id(0))]) +
            (b[(((12) - 1) * (8)) + (get_global_id(0))]) +
            2.e0 * (b[(((16) - 1) * (8)) + (get_global_id(0))]) +
            2.e0 * (b[(((14) - 1) * (8)) + (get_global_id(0))]) +
            2.e0 * (b[(((15) - 1) * (8)) + (get_global_id(0))]);
  float y = i + 1.4e0f * (b[(((1) - 1) * (8)) + (get_global_id(0))]) +
            1.44e1f * (b[(((6) - 1) * (8)) + (get_global_id(0))]) +
            (b[(((10) - 1) * (8)) + (get_global_id(0))]) +
            7.5e-1f * (b[(((11) - 1) * (8)) + (get_global_id(0))]) +
            2.6e0f * (b[(((12) - 1) * (8)) + (get_global_id(0))]) +
            2.e0 * (b[(((16) - 1) * (8)) + (get_global_id(0))]) +
            2.e0 * (b[(((14) - 1) * (8)) + (get_global_id(0))]) +
            2.e0 * (b[(((15) - 1) * (8)) + (get_global_id(0))]);
  float z = i - (b[(((4) - 1) * (8)) + (get_global_id(0))]) -
            (b[(((6) - 1) * (8)) + (get_global_id(0))]) -
            2.5e-1f * (b[(((11) - 1) * (8)) + (get_global_id(0))]) +
            5.e-1 * (b[(((12) - 1) * (8)) + (get_global_id(0))]) +
            5.e-1 * (b[(((16) - 1) * (8)) + (get_global_id(0))]) -
            (b[(((22) - 1) * (8)) + (get_global_id(0))]) +
            2.e0 * (b[(((14) - 1) * (8)) + (get_global_id(0))]) +
            2.e0 * (b[(((15) - 1) * (8)) + (get_global_id(0))]);
  float aa = i + (b[(((1) - 1) * (8)) + (get_global_id(0))]) +
             5.e0 * (b[(((4) - 1) * (8)) + (get_global_id(0))]) +
             5.e0 * (b[(((6) - 1) * (8)) + (get_global_id(0))]) +
             (b[(((10) - 1) * (8)) + (get_global_id(0))]) +
             5.e-1 * (b[(((11) - 1) * (8)) + (get_global_id(0))]) +
             2.5e0f * (b[(((12) - 1) * (8)) + (get_global_id(0))]) +
             2.e0 * (b[(((16) - 1) * (8)) + (get_global_id(0))]) +
             2.e0 * (b[(((14) - 1) * (8)) + (get_global_id(0))]) +
             2.e0 * (b[(((15) - 1) * (8)) + (get_global_id(0))]);
  float ab = i + (b[(((1) - 1) * (8)) + (get_global_id(0))]) +
             5.e0 * (b[(((6) - 1) * (8)) + (get_global_id(0))]) +
             (b[(((10) - 1) * (8)) + (get_global_id(0))]) +
             5.e-1 * (b[(((11) - 1) * (8)) + (get_global_id(0))]) +
             (b[(((12) - 1) * (8)) + (get_global_id(0))]) +
             2.e0 * (b[(((16) - 1) * (8)) + (get_global_id(0))]);

  (c[(((5) - 1) * (8)) + (get_global_id(0))]) =
      (c[(((5) - 1) * (8)) + (get_global_id(0))]) * v *
      (b[(((2) - 1) * (8)) + (get_global_id(0))]) *
      (b[(((2) - 1) * (8)) + (get_global_id(0))]);
  (d[(((5) - 1) * (8)) + (get_global_id(0))]) =
      (d[(((5) - 1) * (8)) + (get_global_id(0))]) * v *
      (b[(((1) - 1) * (8)) + (get_global_id(0))]);
  (c[(((9) - 1) * (8)) + (get_global_id(0))]) =
      (c[(((9) - 1) * (8)) + (get_global_id(0))]) * w *
      (b[(((2) - 1) * (8)) + (get_global_id(0))]) *
      (b[(((5) - 1) * (8)) + (get_global_id(0))]);
  (d[(((9) - 1) * (8)) + (get_global_id(0))]) =
      (d[(((9) - 1) * (8)) + (get_global_id(0))]) * w *
      (b[(((6) - 1) * (8)) + (get_global_id(0))]);
  (c[(((10) - 1) * (8)) + (get_global_id(0))]) =
      (c[(((10) - 1) * (8)) + (get_global_id(0))]) * x *
      (b[(((3) - 1) * (8)) + (get_global_id(0))]) *
      (b[(((2) - 1) * (8)) + (get_global_id(0))]);
  (d[(((10) - 1) * (8)) + (get_global_id(0))]) =
      (d[(((10) - 1) * (8)) + (get_global_id(0))]) * x *
      (b[(((5) - 1) * (8)) + (get_global_id(0))]);
  (c[(((11) - 1) * (8)) + (get_global_id(0))]) =
      (c[(((11) - 1) * (8)) + (get_global_id(0))]) * y *
      (b[(((3) - 1) * (8)) + (get_global_id(0))]) *
      (b[(((3) - 1) * (8)) + (get_global_id(0))]);
  (d[(((11) - 1) * (8)) + (get_global_id(0))]) =
      (d[(((11) - 1) * (8)) + (get_global_id(0))]) * y *
      (b[(((4) - 1) * (8)) + (get_global_id(0))]);
  (c[(((12) - 1) * (8)) + (get_global_id(0))]) =
      (c[(((12) - 1) * (8)) + (get_global_id(0))]) * z *
      (b[(((2) - 1) * (8)) + (get_global_id(0))]) *
      (b[(((4) - 1) * (8)) + (get_global_id(0))]);
  (d[(((12) - 1) * (8)) + (get_global_id(0))]) =
      (d[(((12) - 1) * (8)) + (get_global_id(0))]) * z *
      (b[(((7) - 1) * (8)) + (get_global_id(0))]);
  (c[(((29) - 1) * (8)) + (get_global_id(0))]) =
      (c[(((29) - 1) * (8)) + (get_global_id(0))]) * aa *
      (b[(((11) - 1) * (8)) + (get_global_id(0))]) *
      (b[(((3) - 1) * (8)) + (get_global_id(0))]);
  (d[(((29) - 1) * (8)) + (get_global_id(0))]) =
      (d[(((29) - 1) * (8)) + (get_global_id(0))]) * aa *
      (b[(((12) - 1) * (8)) + (get_global_id(0))]);
  (c[(((46) - 1) * (8)) + (get_global_id(0))]) =
      (c[(((46) - 1) * (8)) + (get_global_id(0))]) * x;
  (d[(((46) - 1) * (8)) + (get_global_id(0))]) =
      (d[(((46) - 1) * (8)) + (get_global_id(0))]) * x *
      (b[(((11) - 1) * (8)) + (get_global_id(0))]) *
      (b[(((2) - 1) * (8)) + (get_global_id(0))]);
  (c[(((121) - 1) * (8)) + (get_global_id(0))]) =
      (c[(((121) - 1) * (8)) + (get_global_id(0))]) * i *
      (b[(((14) - 1) * (8)) + (get_global_id(0))]) *
      (b[(((9) - 1) * (8)) + (get_global_id(0))]);
  (d[(((121) - 1) * (8)) + (get_global_id(0))]) =
      (d[(((121) - 1) * (8)) + (get_global_id(0))]) * i *
      (b[(((20) - 1) * (8)) + (get_global_id(0))]);

  j = (e[(((13) - 1) * (8)) + (get_global_id(0))]) *
      ((x) * (1.0f / ((c[(((126) - 1) * (8)) + (get_global_id(0))]))));
  k = ((j) * (1.0f / ((1.0 + j))));
  l = log10(fmax(j, t));
  m = 6.63e-1f * exp(((-g) * (1.0f / (1.707e3f)))) +
      3.37e-1f * exp(((-g) * (1.0f / (3.2e3f)))) +
      exp(((-4.131e3f) * (1.0f / (g))));
  n = log10(fmax(m, t));
  o = 0.75 - 1.27 * n;
  p = l - (0.4 + 0.67 * n);
  s = ((p) * (1.0f / ((o - 0.14 * p))));
  q = ((n) * (1.0f / ((1.0 + s * s))));
  r = exp10(q);
  k = r * k;
  (c[(((126) - 1) * (8)) + (get_global_id(0))]) =
      (c[(((126) - 1) * (8)) + (get_global_id(0))]) * k;
  (d[(((126) - 1) * (8)) + (get_global_id(0))]) =
      (d[(((126) - 1) * (8)) + (get_global_id(0))]) * k;

  j = (e[(((14) - 1) * (8)) + (get_global_id(0))]) *
      ((x) * (1.0f / ((c[(((132) - 1) * (8)) + (get_global_id(0))]))));
  k = ((j) * (1.0f / ((1.0 + j))));
  l = log10(fmax(j, t));
  m = 2.18e-1f * exp(((-g) * (1.0f / (2.075e2f)))) +
      7.82e-1f * exp(((-g) * (1.0f / (2.663e3f)))) +
      exp(((-6.095e3f) * (1.0f / (g))));
  n = log10(fmax(m, t));
  o = 0.75 - 1.27 * n;
  p = l - (0.4 + 0.67 * n);
  s = ((p) * (1.0f / ((o - 0.14 * p))));
  q = ((n) * (1.0f / ((1.0 + s * s))));
  r = exp10(q);
  k = r * k;
  (c[(((132) - 1) * (8)) + (get_global_id(0))]) =
      (c[(((132) - 1) * (8)) + (get_global_id(0))]) * k;
  (d[(((132) - 1) * (8)) + (get_global_id(0))]) =
      (d[(((132) - 1) * (8)) + (get_global_id(0))]) * k;

  j = (e[(((15) - 1) * (8)) + (get_global_id(0))]) *
      ((x) * (1.0f / ((c[(((145) - 1) * (8)) + (get_global_id(0))]))));
  k = ((j) * (1.0f / ((1.0 + j))));
  l = log10(fmax(j, t));
  m = 8.25e-1f * exp(((-g) * (1.0f / (1.3406e3f)))) +
      1.75e-1f * exp(((-g) * (1.0f / (6.e4f)))) +
      exp(((-1.01398e4f) * (1.0f / (g))));
  n = log10(fmax(m, t));
  o = 0.75 - 1.27 * n;
  p = l - (0.4 + 0.67 * n);
  s = ((p) * (1.0f / ((o - 0.14 * p))));
  q = ((n) * (1.0f / ((1.0 + s * s))));
  r = exp10(q);
  k = r * k;
  (c[(((145) - 1) * (8)) + (get_global_id(0))]) =
      (c[(((145) - 1) * (8)) + (get_global_id(0))]) * k;
  (d[(((145) - 1) * (8)) + (get_global_id(0))]) =
      (d[(((145) - 1) * (8)) + (get_global_id(0))]) * k;

  j = (e[(((16) - 1) * (8)) + (get_global_id(0))]) *
      ((x) * (1.0f / ((c[(((148) - 1) * (8)) + (get_global_id(0))]))));
  k = ((j) * (1.0f / ((1.0 + j))));
  l = log10(fmax(j, t));
  m = 4.5e-1f * exp(((-g) * (1.0f / (8.9e3f)))) +
      5.5e-1f * exp(((-g) * (1.0f / (4.35e3f)))) +
      exp(((-7.244e3f) * (1.0f / (g))));
  n = log10(fmax(m, t));
  o = 0.75 - 1.27 * n;
  p = l - (0.4 + 0.67 * n);
  s = ((p) * (1.0f / ((o - 0.14 * p))));
  q = ((n) * (1.0f / ((1.0 + s * s))));
  r = exp10(q);
  k = r * k;
  (c[(((148) - 1) * (8)) + (get_global_id(0))]) =
      (c[(((148) - 1) * (8)) + (get_global_id(0))]) * k;
  (d[(((148) - 1) * (8)) + (get_global_id(0))]) =
      (d[(((148) - 1) * (8)) + (get_global_id(0))]) * k;

  j = (e[(((17) - 1) * (8)) + (get_global_id(0))]) *
      ((x) * (1.0f / ((c[(((155) - 1) * (8)) + (get_global_id(0))]))));
  k = ((j) * (1.0f / ((1.0 + j))));
  l = log10(fmax(j, t));
  m = 2.655e-1f * exp(((-g) * (1.0f / (1.8e2f)))) +
      7.345e-1f * exp(((-g) * (1.0f / (1.035e3f)))) +
      exp(((-5.417e3f) * (1.0f / (g))));
  n = log10(fmax(m, t));
  o = 0.75 - 1.27 * n;
  p = l - (0.4 + 0.67 * n);
  s = ((p) * (1.0f / ((o - 0.14 * p))));
  q = ((n) * (1.0f / ((1.0 + s * s))));
  r = exp10(q);
  k = r * k;
  (c[(((155) - 1) * (8)) + (get_global_id(0))]) =
      (c[(((155) - 1) * (8)) + (get_global_id(0))]) * k;
  (d[(((155) - 1) * (8)) + (get_global_id(0))]) =
      (d[(((155) - 1) * (8)) + (get_global_id(0))]) * k;

  j = (e[(((18) - 1) * (8)) + (get_global_id(0))]) *
      ((x) * (1.0f / ((c[(((156) - 1) * (8)) + (get_global_id(0))]))));
  k = ((j) * (1.0f / ((1.0 + j))));
  l = log10(fmax(j, t));
  m = 2.47e-2f * exp(((-g) * (1.0f / (2.1e2f)))) +
      9.753e-1f * exp(((-g) * (1.0f / (9.84e2f)))) +
      exp(((-4.374e3f) * (1.0f / (g))));
  n = log10(fmax(m, t));
  o = 0.75 - 1.27 * n;
  p = l - (0.4 + 0.67 * n);
  s = ((p) * (1.0f / ((o - 0.14 * p))));
  q = ((n) * (1.0f / ((1.0 + s * s))));
  r = exp10(q);
  k = r * k;
  (c[(((156) - 1) * (8)) + (get_global_id(0))]) =
      (c[(((156) - 1) * (8)) + (get_global_id(0))]) * k;
  (d[(((156) - 1) * (8)) + (get_global_id(0))]) =
      (d[(((156) - 1) * (8)) + (get_global_id(0))]) * k;

  j = (e[(((19) - 1) * (8)) + (get_global_id(0))]) *
      ((x) * (1.0f / ((c[(((170) - 1) * (8)) + (get_global_id(0))]))));
  k = ((j) * (1.0f / ((1.0 + j))));
  l = log10(fmax(j, t));
  m = 1.578e-1f * exp(((-g) * (1.0f / (1.25e2f)))) +
      8.422e-1f * exp(((-g) * (1.0f / (2.219e3f)))) +
      exp(((-6.882e3f) * (1.0f / (g))));
  n = log10(fmax(m, t));
  o = 0.75 - 1.27 * n;
  p = l - (0.4 + 0.67 * n);
  s = ((p) * (1.0f / ((o - 0.14 * p))));
  q = ((n) * (1.0f / ((1.0 + s * s))));
  r = exp10(q);
  k = r * k;
  (c[(((170) - 1) * (8)) + (get_global_id(0))]) =
      (c[(((170) - 1) * (8)) + (get_global_id(0))]) * k;
  (d[(((170) - 1) * (8)) + (get_global_id(0))]) =
      (d[(((170) - 1) * (8)) + (get_global_id(0))]) * k;

  j = (e[(((20) - 1) * (8)) + (get_global_id(0))]) *
      ((x) * (1.0f / ((c[(((185) - 1) * (8)) + (get_global_id(0))]))));
  k = ((j) * (1.0f / ((1.0 + j))));
  l = log10(fmax(j, t));
  m = 9.8e-1f * exp(((-g) * (1.0f / (1.0966e3f)))) +
      2.e-2 * exp(((-g) * (1.0f / (1.0966e3f)))) +
      exp(((-6.8595e3f) * (1.0f / (g))));
  n = log10(fmax(m, t));
  o = 0.75 - 1.27 * n;
  p = l - (0.4 + 0.67 * n);
  s = ((p) * (1.0f / ((o - 0.14 * p))));
  q = ((n) * (1.0f / ((1.0 + s * s))));
  r = exp10(q);
  k = r * k;
  (c[(((185) - 1) * (8)) + (get_global_id(0))]) =
      (c[(((185) - 1) * (8)) + (get_global_id(0))]) * k;
  (d[(((185) - 1) * (8)) + (get_global_id(0))]) =
      (d[(((185) - 1) * (8)) + (get_global_id(0))]) * k;

  j = (e[(((21) - 1) * (8)) + (get_global_id(0))]) *
      ((ab) * (1.0f / ((c[(((190) - 1) * (8)) + (get_global_id(0))]))));
  k = ((j) * (1.0f / ((1.0 + j))));
  l = log10(fmax(j, t));
  m = 0.e0 * exp(((-g) * (1.0f / (1.e3f)))) +
      1.e0 * exp(((-g) * (1.0f / (1.31e3f)))) +
      exp(((-4.8097e4f) * (1.0f / (g))));
  n = log10(fmax(m, t));
  o = 0.75 - 1.27 * n;
  p = l - (0.4 + 0.67 * n);
  s = ((p) * (1.0f / ((o - 0.14 * p))));
  q = ((n) * (1.0f / ((1.0 + s * s))));
  r = exp10(q);
  k = r * k;
  (c[(((190) - 1) * (8)) + (get_global_id(0))]) =
      (c[(((190) - 1) * (8)) + (get_global_id(0))]) * k;
  (d[(((190) - 1) * (8)) + (get_global_id(0))]) =
      (d[(((190) - 1) * (8)) + (get_global_id(0))]) * k;
}