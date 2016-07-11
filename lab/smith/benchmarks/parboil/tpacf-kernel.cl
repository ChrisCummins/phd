__kernel void C(__global unsigned long* e, __global float* f,
                __constant float* g, int h, int i) {
  __global float* j = f + i * (h + 1);
  __global float* k = j + i * (h + 1);

  unsigned int l = get_group_id(0);
  unsigned int m = get_local_id(0);
  bool n = (l < (h + 1));

  __global float* o;
  __global float* p;
  __global float* q;
  __global float* r;
  __global float* s;
  __global float* t;

  __local unsigned int u[20][((256 / 32) * 16)];

  for (unsigned int v = 0; v < 20 * ((256 / 32) * 16); v += 256) {
    if ((v + m) < (20 * ((256 / 32) * 16))) {
      u[(v + m) / ((256 / 32) * 16)][(v + m) % ((256 / 32) * 16)] = 0;
    }
  }

  if (!n) {
    o = f;
    p = j;
    q = k;

    r = f + i * (l - h);
    s = j + i * (l - h);
    t = k + i * (l - h);
  } else {
    r = f + i * (l);
    s = j + i * (l);
    t = k + i * (l);

    o = r;
    p = s;
    q = t;
  }

  for (unsigned int w = 0; w < i; w += 256) {
    float x;
    float y;
    float z;

    if (m + w < i) {
      x = r[m + w];
      y = s[m + w];
      z = t[m + w];
    }

    for (unsigned int aa = 0; aa < i && (n ? aa < w + 256 : 1); aa++) {
      float ab = o[aa] * x + p[aa] * y + q[aa] * z;

      unsigned int ac;

      unsigned int ad = 0;
      unsigned int ae = 20;
      {
        unsigned int af;

        while (ae > ad + 1) {
          af = (ad + ae) / 2;
          if (ab >= g[af])
            ae = af;
          else
            ad = af;
        }
        ac = ae - 1;
      }

      unsigned int ag = m / (32 / 16);
      if ((ab < g[ad]) && (ab >= g[ae]) && (!n || (m + w > aa)) &&
          ((m + w) < i)) {
        atom_inc(&(u[ac][ag]));
      }
    }
  }

  unsigned int ah = m & ((((256 / 32) * 16) >> 1) - 1);
  unsigned int ac = m / (((256 / 32) * 16) >> 1);
  for (unsigned int ai = ((256 / 32) * 16) >> 1; ai > 0; ai >>= 1) {
    for (unsigned int aj = 0; aj < 20; aj += 256 / (((256 / 32) * 16) >> 1)) {
      barrier(1 | 2);
      if (ah < ai && aj + ac < 20) {
        unsigned long ak = u[aj + ac][ah] + u[aj + ac][ah + ai];
        u[aj + ac][ah] = ak;
      }
    }
  }

  barrier(1 | 2);

  __global unsigned long* al = e + 20 * l;
  if (m < 20) {
    al[m] = u[m][0];
  }
}
