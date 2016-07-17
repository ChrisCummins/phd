__kernel void A(__global const float* a, __global const float* b,
                __global float* c, __global float* d, __global const float* e,
                const float f) {
  const float g = a[get_global_id(0)] * f;
  const float h = log((g));
  float i = 0.0;
  float j, k, l, m, n, o;
  float p, q, r;
  float s;

  const float t = 0x1.0p-126f;

  for (unsigned int u = 1; u <= 22; u++) {
    i += (b[(((u)-1) * (8)) + (get_global_id(0))]);
  }

  float v = i + (b[(((1) - 1) * (8)) + (get_global_id(0))]) +
            5.e0 * (b[(((6) - 1) * (8)) + (get_global_id(0))]) +
            (b[(((10) - 1) * (8)) + (get_global_id(0))]) +
            5.e-1 * (b[(((11) - 1) * (8)) + (get_global_id(0))]) +
            (b[(((12) - 1) * (8)) + (get_global_id(0))]) +
            2.e0 * (b[(((16) - 1) * (8)) + (get_global_id(0))]) +
            2.e0 * (b[(((14) - 1) * (8)) + (get_global_id(0))]) +
            2.e0 * (b[(((15) - 1) * (8)) + (get_global_id(0))]);
  float w = i + (b[(((1) - 1) * (8)) + (get_global_id(0))]) +
            5.e0 * (b[(((6) - 1) * (8)) + (get_global_id(0))]) +
            (b[(((10) - 1) * (8)) + (get_global_id(0))]) +
            5.e-1 * (b[(((11) - 1) * (8)) + (get_global_id(0))]) +
            (b[(((12) - 1) * (8)) + (get_global_id(0))]) +
            2.e0 * (b[(((16) - 1) * (8)) + (get_global_id(0))]) +
            1.5e0f * (b[(((14) - 1) * (8)) + (get_global_id(0))]) +
            1.5e0f * (b[(((15) - 1) * (8)) + (get_global_id(0))]);
  float x = i + (b[(((1) - 1) * (8)) + (get_global_id(0))]) +
            5.e0 * (b[(((6) - 1) * (8)) + (get_global_id(0))]) +
            (b[(((10) - 1) * (8)) + (get_global_id(0))]) +
            5.e-1 * (b[(((11) - 1) * (8)) + (get_global_id(0))]) +
            (b[(((12) - 1) * (8)) + (get_global_id(0))]) +
            2.e0 * (b[(((16) - 1) * (8)) + (get_global_id(0))]) +
            2.e0 * (b[(((14) - 1) * (8)) + (get_global_id(0))]) +
            2.e0 * (b[(((15) - 1) * (8)) + (get_global_id(0))]);

  j = (e[(((1) - 1) * (8)) + (get_global_id(0))]) *
      ((x) * (1.0f / ((c[(((16) - 1) * (8)) + (get_global_id(0))]))));
  k = ((j) * (1.0f / ((1.0 + j))));
  l = log10(fmax(j, t));
  m = 2.654e-1f * exp(((-g) * (1.0f / (9.4e1f)))) +
      7.346e-1f * exp(((-g) * (1.0f / (1.756e3f)))) +
      exp(((-5.182e3f) * (1.0f / (g))));
  n = log10(fmax(m, t));
  o = 0.75 - 1.27 * n;
  p = l - (0.4 + 0.67 * n);
  s = ((p) * (1.0f / ((o - 0.14 * p))));
  q = ((n) * (1.0f / ((1.0 + s * s))));
  r = exp10(q);
  k = r * k;
  (c[(((16) - 1) * (8)) + (get_global_id(0))]) =
      (c[(((16) - 1) * (8)) + (get_global_id(0))]) * k;
  (d[(((16) - 1) * (8)) + (get_global_id(0))]) =
      (d[(((16) - 1) * (8)) + (get_global_id(0))]) * k;

  j = (e[(((2) - 1) * (8)) + (get_global_id(0))]) *
      ((v) * (1.0f / ((c[(((31) - 1) * (8)) + (get_global_id(0))]))));
  k = ((j) * (1.0f / ((1.0 + j))));
  l = log10(fmax(j, t));
  m = 6.8e-2f * exp(((-g) * (1.0f / (1.97e2f)))) +
      9.32e-1f * exp(((-g) * (1.0f / (1.54e3f)))) +
      exp(((-1.03e4f) * (1.0f / (g))));
  n = log10(fmax(m, t));
  o = 0.75 - 1.27 * n;
  p = l - (0.4 + 0.67 * n);
  s = ((p) * (1.0f / ((o - 0.14 * p))));
  q = ((n) * (1.0f / ((1.0 + s * s))));
  r = exp10(q);
  k = r * k;
  (c[(((31) - 1) * (8)) + (get_global_id(0))]) =
      (c[(((31) - 1) * (8)) + (get_global_id(0))]) * k;
  (d[(((31) - 1) * (8)) + (get_global_id(0))]) =
      (d[(((31) - 1) * (8)) + (get_global_id(0))]) * k;

  j = (e[(((3) - 1) * (8)) + (get_global_id(0))]) *
      ((v) * (1.0f / ((c[(((39) - 1) * (8)) + (get_global_id(0))]))));
  k = ((j) * (1.0f / ((1.0 + j))));
  l = log10(fmax(j, t));
  m = 4.243e-1f * exp(((-g) * (1.0f / (2.37e2f)))) +
      5.757e-1f * exp(((-g) * (1.0f / (1.652e3f)))) +
      exp(((-5.069e3f) * (1.0f / (g))));
  n = log10(fmax(m, t));
  o = 0.75 - 1.27 * n;
  p = l - (0.4 + 0.67 * n);
  s = ((p) * (1.0f / ((o - 0.14 * p))));
  q = ((n) * (1.0f / ((1.0 + s * s))));
  r = exp10(q);
  k = r * k;
  (c[(((39) - 1) * (8)) + (get_global_id(0))]) =
      (c[(((39) - 1) * (8)) + (get_global_id(0))]) * k;
  (d[(((39) - 1) * (8)) + (get_global_id(0))]) =
      (d[(((39) - 1) * (8)) + (get_global_id(0))]) * k;

  j = (e[(((4) - 1) * (8)) + (get_global_id(0))]) *
      ((v) * (1.0f / ((c[(((41) - 1) * (8)) + (get_global_id(0))]))));
  k = ((j) * (1.0f / ((1.0 + j))));
  l = log10(fmax(j, t));
  m = 2.176e-1f * exp(((-g) * (1.0f / (2.71e2f)))) +
      7.824e-1f * exp(((-g) * (1.0f / (2.755e3f)))) +
      exp(((-6.57e3f) * (1.0f / (g))));
  n = log10(fmax(m, t));
  o = 0.75 - 1.27 * n;
  p = l - (0.4 + 0.67 * n);
  s = ((p) * (1.0f / ((o - 0.14 * p))));
  q = ((n) * (1.0f / ((1.0 + s * s))));
  r = exp10(q);
  k = r * k;
  (c[(((41) - 1) * (8)) + (get_global_id(0))]) =
      (c[(((41) - 1) * (8)) + (get_global_id(0))]) * k;
  (d[(((41) - 1) * (8)) + (get_global_id(0))]) =
      (d[(((41) - 1) * (8)) + (get_global_id(0))]) * k;

  j = (e[(((5) - 1) * (8)) + (get_global_id(0))]) *
      ((v) * (1.0f / ((c[(((48) - 1) * (8)) + (get_global_id(0))]))));
  k = ((j) * (1.0f / ((1.0 + j))));
  l = log10(fmax(j, t));
  m = 3.2e-1f * exp(((-g) * (1.0f / (7.8e1f)))) +
      6.8e-1f * exp(((-g) * (1.0f / (1.995e3f)))) +
      exp(((-5.59e3f) * (1.0f / (g))));
  n = log10(fmax(m, t));
  o = 0.75 - 1.27 * n;
  p = l - (0.4 + 0.67 * n);
  s = ((p) * (1.0f / ((o - 0.14 * p))));
  q = ((n) * (1.0f / ((1.0 + s * s))));
  r = exp10(q);
  k = r * k;
  (c[(((48) - 1) * (8)) + (get_global_id(0))]) =
      (c[(((48) - 1) * (8)) + (get_global_id(0))]) * k;
  (d[(((48) - 1) * (8)) + (get_global_id(0))]) =
      (d[(((48) - 1) * (8)) + (get_global_id(0))]) * k;

  j = (e[(((6) - 1) * (8)) + (get_global_id(0))]) *
      ((v) * (1.0f / ((c[(((56) - 1) * (8)) + (get_global_id(0))]))));
  k = ((j) * (1.0f / ((1.0 + j))));
  l = log10(fmax(j, t));
  m = 4.093e-1f * exp(((-g) * (1.0f / (2.75e2f)))) +
      5.907e-1f * exp(((-g) * (1.0f / (1.226e3f)))) +
      exp(((-5.185e3f) * (1.0f / (g))));
  n = log10(fmax(m, t));
  o = 0.75 - 1.27 * n;
  p = l - (0.4 + 0.67 * n);
  s = ((p) * (1.0f / ((o - 0.14 * p))));
  q = ((n) * (1.0f / ((1.0 + s * s))));
  r = exp10(q);
  k = r * k;
  (c[(((56) - 1) * (8)) + (get_global_id(0))]) =
      (c[(((56) - 1) * (8)) + (get_global_id(0))]) * k;
  (d[(((56) - 1) * (8)) + (get_global_id(0))]) =
      (d[(((56) - 1) * (8)) + (get_global_id(0))]) * k;

  j = (e[(((7) - 1) * (8)) + (get_global_id(0))]) *
      ((v) * (1.0f / ((c[(((71) - 1) * (8)) + (get_global_id(0))]))));
  k = ((j) * (1.0f / ((1.0 + j))));
  l = log10(fmax(j, t));
  m = 2.42e-1f * exp(((-g) * (1.0f / (9.4e1f)))) +
      7.58e-1f * exp(((-g) * (1.0f / (1.555e3f)))) +
      exp(((-4.2e3f) * (1.0f / (g))));
  n = log10(fmax(m, t));
  o = 0.75 - 1.27 * n;
  p = l - (0.4 + 0.67 * n);
  s = ((p) * (1.0f / ((o - 0.14 * p))));
  q = ((n) * (1.0f / ((1.0 + s * s))));
  r = exp10(q);
  k = r * k;
  (c[(((71) - 1) * (8)) + (get_global_id(0))]) =
      (c[(((71) - 1) * (8)) + (get_global_id(0))]) * k;
  (d[(((71) - 1) * (8)) + (get_global_id(0))]) =
      (d[(((71) - 1) * (8)) + (get_global_id(0))]) * k;

  j = (e[(((8) - 1) * (8)) + (get_global_id(0))]) *
      ((v) * (1.0f / ((c[(((78) - 1) * (8)) + (get_global_id(0))]))));
  k = ((j) * (1.0f / ((1.0 + j))));
  l = log10(fmax(j, t));
  m = 2.17e-1f * exp(((-g) * (1.0f / (7.4e1f)))) +
      7.83e-1f * exp(((-g) * (1.0f / (2.941e3f)))) +
      exp(((-6.964e3f) * (1.0f / (g))));
  n = log10(fmax(m, t));
  o = 0.75 - 1.27 * n;
  p = l - (0.4 + 0.67 * n);
  s = ((p) * (1.0f / ((o - 0.14 * p))));
  q = ((n) * (1.0f / ((1.0 + s * s))));
  r = exp10(q);
  k = r * k;
  (c[(((78) - 1) * (8)) + (get_global_id(0))]) =
      (c[(((78) - 1) * (8)) + (get_global_id(0))]) * k;
  (d[(((78) - 1) * (8)) + (get_global_id(0))]) =
      (d[(((78) - 1) * (8)) + (get_global_id(0))]) * k;

  j = (e[(((9) - 1) * (8)) + (get_global_id(0))]) *
      ((v) * (1.0f / ((c[(((89) - 1) * (8)) + (get_global_id(0))]))));
  k = ((j) * (1.0f / ((1.0 + j))));
  l = log10(fmax(j, t));
  m = 3.827e-1f * exp(((-g) * (1.0f / (1.3076e1f)))) +
      6.173e-1f * exp(((-g) * (1.0f / (2.078e3f)))) +
      exp(((-5.093e3f) * (1.0f / (g))));
  n = log10(fmax(m, t));
  o = 0.75 - 1.27 * n;
  p = l - (0.4 + 0.67 * n);
  s = ((p) * (1.0f / ((o - 0.14 * p))));
  q = ((n) * (1.0f / ((1.0 + s * s))));
  r = exp10(q);
  k = r * k;
  (c[(((89) - 1) * (8)) + (get_global_id(0))]) =
      (c[(((89) - 1) * (8)) + (get_global_id(0))]) * k;
  (d[(((89) - 1) * (8)) + (get_global_id(0))]) =
      (d[(((89) - 1) * (8)) + (get_global_id(0))]) * k;

  j = (e[(((10) - 1) * (8)) + (get_global_id(0))]) *
      ((v) * (1.0f / ((c[(((93) - 1) * (8)) + (get_global_id(0))]))));
  k = ((j) * (1.0f / ((1.0 + j))));
  l = log10(fmax(j, t));
  m = 4.675e-1f * exp(((-g) * (1.0f / (1.51e2f)))) +
      5.325e-1f * exp(((-g) * (1.0f / (1.038e3f)))) +
      exp(((-4.97e3f) * (1.0f / (g))));
  n = log10(fmax(m, t));
  o = 0.75 - 1.27 * n;
  p = l - (0.4 + 0.67 * n);
  s = ((p) * (1.0f / ((o - 0.14 * p))));
  q = ((n) * (1.0f / ((1.0 + s * s))));
  r = exp10(q);
  k = r * k;
  (c[(((93) - 1) * (8)) + (get_global_id(0))]) =
      (c[(((93) - 1) * (8)) + (get_global_id(0))]) * k;
  (d[(((93) - 1) * (8)) + (get_global_id(0))]) =
      (d[(((93) - 1) * (8)) + (get_global_id(0))]) * k;

  j = (e[(((11) - 1) * (8)) + (get_global_id(0))]) *
      ((w) * (1.0f / ((c[(((114) - 1) * (8)) + (get_global_id(0))]))));
  k = ((j) * (1.0f / ((1.0 + j))));
  (c[(((114) - 1) * (8)) + (get_global_id(0))]) =
      (c[(((114) - 1) * (8)) + (get_global_id(0))]) * k;
  (d[(((114) - 1) * (8)) + (get_global_id(0))]) =
      (d[(((114) - 1) * (8)) + (get_global_id(0))]) * k;

  j = (e[(((12) - 1) * (8)) + (get_global_id(0))]) *
      ((v) * (1.0f / ((c[(((115) - 1) * (8)) + (get_global_id(0))]))));
  k = ((j) * (1.0f / ((1.0 + j))));
  l = log10(fmax(j, t));
  m = -9.816e-1f * exp(((-g) * (1.0f / (5.3837e3f)))) +
      1.9816e0f * exp(((-g) * (1.0f / (4.2932e0f)))) +
      exp(((7.95e-2f) * (1.0f / (g))));
  n = log10(fmax(m, t));
  o = 0.75 - 1.27 * n;
  p = l - (0.4 + 0.67 * n);
  s = ((p) * (1.0f / ((o - 0.14 * p))));
  q = ((n) * (1.0f / ((1.0 + s * s))));
  r = exp10(q);
  k = r * k;
  (c[(((115) - 1) * (8)) + (get_global_id(0))]) =
      (c[(((115) - 1) * (8)) + (get_global_id(0))]) * k;
  (d[(((115) - 1) * (8)) + (get_global_id(0))]) =
      (d[(((115) - 1) * (8)) + (get_global_id(0))]) * k;
}