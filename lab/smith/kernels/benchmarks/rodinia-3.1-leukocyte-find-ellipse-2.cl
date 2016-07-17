__kernel void A(int a, int b, int c, int d, __constant float *e,
                __global float *f, __global float *g) {
  int h = c / 2;
  int i = d / 2;

  int j = get_global_id(0);
  int k = j % a;
  int l = j / a;

  if (l > b) return;

  float m = 0.0f;

  int n, o, p, q;

  if (l < b) {
    for (n = 0; n < c; n++) {
      q = k - h + n;

      if ((q >= 0) && (q < a)) {
        for (o = 0; o < d; o++) {
          p = l - i + o;

          if ((p >= 0) && (p < b) && (e[(n * d) + o] != 0)) {
            int r = (p * a) + q;
            float s = f[r];

            if (s > m) m = s;
          }
        }
      }
    }

    g[(k * b) + l] = m;
  }
}