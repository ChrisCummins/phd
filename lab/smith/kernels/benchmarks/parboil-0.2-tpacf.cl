__kernel void A(__global unsigned long* a, __global float* b, __constant float* c, int d, int e) {
  __global float* f = b + e * (d + 1);
  __global float* g = f + e * (d + 1);

  unsigned int h = get_group_id(0);
  unsigned int i = get_local_id(0);
  bool j = (h < (d + 1));

  __global float* k;
  __global float* l;
  __global float* m;
  __global float* n;
  __global float* o;
  __global float* p;

  __local unsigned int q[20][((256 / 32) * 16)];

  for (unsigned int r = 0; r < 20 * ((256 / 32) * 16); r += 256) {
    if ((r + i) < (20 * ((256 / 32) * 16))) {
      q[(r + i) / ((256 / 32) * 16)][(r + i) % ((256 / 32) * 16)] = 0;
    }
  }

  if (!j) {
    k = b;
    l = f;
    m = g;

    n = b + e * (h - d);
    o = f + e * (h - d);
    p = g + e * (h - d);
  } else {
    n = b + e * (h);
    o = f + e * (h);
    p = g + e * (h);

    k = n;
    l = o;
    m = p;
  }

  for (unsigned int s = 0; s < e; s += 256) {
    float t;
    float u;
    float v;

    if (i + s < e) {
      t = n[i + s];
      u = o[i + s];
      v = p[i + s];
    }

    for (unsigned int w = 0; w < e && (j ? w < s + 256 : 1); w++) {
      float x = k[w] * t + l[w] * u + m[w] * v;

      unsigned int y;

      unsigned int z = 0;
      unsigned int aa = 20;
      {
        unsigned int ab;

        while (aa > z + 1) {
          ab = (z + aa) / 2;
          if (x >= c[ab])
            aa = ab;
          else
            z = ab;
        }
        y = aa - 1;
      }

      unsigned int ac = i / (32 / 16);
      if ((x < c[z]) && (x >= c[aa]) && (!j || (i + s > w)) && ((i + s) < e)) {
        atom_inc(&(q[y][ac]));
      }
    }
  }

  unsigned int ad = i & ((((256 / 32) * 16) >> 1) - 1);
  unsigned int y = i / (((256 / 32) * 16) >> 1);
  for (unsigned int ae = ((256 / 32) * 16) >> 1; ae > 0; ae >>= 1) {
    for (unsigned int af = 0; af < 20; af += 256 / (((256 / 32) * 16) >> 1)) {
      barrier(1 | 2);
      if (ad < ae && af + y < 20) {
        unsigned long ag = q[af + y][ad] + q[af + y][ad + ae];
        q[af + y][ad] = ag;
      }
    }
  }

  barrier(1 | 2);

  __global unsigned long* ah = a + 20 * h;
  if (i < 20) {
    ah[i] = q[i][0];
  }
}