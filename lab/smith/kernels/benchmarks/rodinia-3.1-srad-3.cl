__kernel void A(long a, long b, int c, __global float* d, __global float* e, int f) {
  int g = get_group_id(0);
  int h = get_local_id(0);
  int i = (g * 256) + h;

  int j = 256 - (f * 256 - b);
  int k = 0;

  __local float l[256];
  __local float m[256];

  int n;

  if (i < b) {
    l[h] = d[i * c];
    m[h] = e[i * c];
  }

  barrier(1 | 2);

  if (j == 256) {
    for (n = 2; n <= 256; n = 2 * n) {
      if ((h + 1) % n == 0) {
        l[h] = l[h] + l[h - n / 2];
        m[h] = m[h] + m[h - n / 2];
      }

      barrier(1);
    }

    if (h == (256 - 1)) {
      d[g * c * 256] = l[h];
      e[g * c * 256] = m[h];
    }
  }

  else {
    if (g != (f - 1)) {
      for (n = 2; n <= 256; n = 2 * n) {
        if ((h + 1) % n == 0) {
          l[h] = l[h] + l[h - n / 2];
          m[h] = m[h] + m[h - n / 2];
        }

        barrier(1);
      }

      if (h == (256 - 1)) {
        d[g * c * 256] = l[h];
        e[g * c * 256] = m[h];
      }
    }

    else {
      for (n = 2; n <= 256; n = 2 * n) {
        if (j >= n) {
          k = n;
        }
      }

      for (n = 2; n <= k; n = 2 * n) {
        if ((h + 1) % n == 0 && h < k) {
          l[h] = l[h] + l[h - n / 2];
          m[h] = m[h] + m[h - n / 2];
        }

        barrier(1);
      }

      if (h == (k - 1)) {
        for (n = (g * 256) + k; n < (g * 256) + j; n++) {
          l[h] = l[h] + d[n];
          m[h] = m[h] + e[n];
        }

        d[g * c * 256] = l[h];
        e[g * c * 256] = m[h];
      }
    }
  }
}