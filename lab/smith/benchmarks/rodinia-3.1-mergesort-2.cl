__kernel void A(__global float4 *a, __global float4 *b, const int c, int d,
                __global int *e) {
  int f = get_global_id(0);

  int g = f / d;
  if (g >= (1024)) return;

  int h = f - g * d;
  int i = e[g] + h * c;

  int j = i + c / 2;
  global float4 *k;
  k = &(b[i]);

  if (i >= e[g + 1]) return;
  if (j >= e[g + 1]) {
    for (int l = 0; l < (e[g + 1] - i); l++) {
      k[l] = a[i + l];
    }
    return;
  }

  int m = 0;
  int n = 0;
  int o = 0;
  float4 p, q;
  p = a[i + m];
  q = a[j + n];

  while (true) {
    float4 r = a[i + m + 1];
    float4 s = a[j + n + 1];

    float4 t = B(p, q);
    float4 u = C(p, q);
    p = A(t);
    q = A(u);

    k[o++] = p;

    bool v;
    bool w;

    v = (m + 1 < c / 2);
    w = (n + 1 < c / 2) && (j + n + 1 < e[g + 1]);

    if (v) {
      if (w) {
        if (r.x < s.x) {
          m += 1;
          p = r;
        } else {
          n += 1;
          p = s;
        }
      } else {
        m += 1;
        p = r;
      }
    } else {
      if (w) {
        n += 1;
        p = s;
      } else {
        break;
      }
    }
  }
  k[o++] = q;
}