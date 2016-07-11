float4 A(__local float* a) {
  float4 b = (float4)(1.0f, 1.0f, 1.0f, 0.0f);
  for (int c = 0; c < 8; c++) {
    float d = b.x * a[0] + b.y * a[1] + b.z * a[2];
    float e = b.x * a[1] + b.y * a[3] + b.z * a[4];
    float f = b.x * a[2] + b.y * a[4] + b.z * a[5];
    float g = max(max(d, e), f);
    float h = 1.0f / g;

    b.x = d * h;
    b.y = e * h;
    b.z = f * h;
  }

  return b;
}

void B(__local const float4* i, __local float4* j) {
  const int k = get_local_id(0);

  j[k] = i[k];
  j[k] += j[k ^ 8];
  j[k] += j[k ^ 4];
  j[k] += j[k ^ 2];
  j[k] += j[k ^ 1];
}

float4 C(__local const float4* i, float4 l, __local float* m) {
  const int k = get_local_id(0);

  float4 n = i[k] - l * 0.0625f;

  m[6 * k + 0] = n.x * n.x;
  m[6 * k + 1] = n.x * n.y;
  m[6 * k + 2] = n.x * n.z;
  m[6 * k + 3] = n.y * n.y;
  m[6 * k + 4] = n.y * n.z;
  m[6 * k + 5] = n.z * n.z;

  for (int o = 8; o > 0; o >>= 1) {
    if (k < o) {
      m[6 * k + 0] += m[6 * (k + o) + 0];
      m[6 * k + 1] += m[6 * (k + o) + 1];
      m[6 * k + 2] += m[6 * (k + o) + 2];
      m[6 * k + 3] += m[6 * (k + o) + 3];
      m[6 * k + 4] += m[6 * (k + o) + 4];
      m[6 * k + 5] += m[6 * (k + o) + 5];
    }
  }

  return A(m);
}

void D(__local const float* p, __local int* q) {
  const int r = get_local_id(0);

  int s = 0;

  for (int c = 0; c < 16; c++) {
    s += (p[c] < p[r]);
  }

  q[r] = s;

  for (int c = 0; c < 15; c++) {
    if (r > c && q[r] == q[c]) ++q[r];
  }
}

void E(__global const uint* t, __local float4* i, __local float4* j,
       __local int* u, __local float* v, int w) {
  const int x = get_group_id(0) + w;
  const int k = get_local_id(0);

  float4 y;

  if (k < 16) {
    uint z = t[(x)*16 + k];

    i[k].x = ((z >> 0) & 0xFF) * 0.003921568627f;
    i[k].y = ((z >> 8) & 0xFF) * 0.003921568627f;
    i[k].z = ((z >> 16) & 0xFF) * 0.003921568627f;

    B(i, j);
    float4 aa = C(i, j[k], v);

    v[k] = i[k].x * aa.x + i[k].y * aa.y + i[k].z * aa.z;

    D(v, u);

    y = i[k];

    i[u[k]] = y;
  }
}

float4 F(float4 b, ushort* ab) {
  ushort d = __clc_rint(clamp(b.x, 0.0f, 1.0f) * 31.0f);
  ushort e = __clc_rint(clamp(b.y, 0.0f, 1.0f) * 63.0f);
  ushort f = __clc_rint(clamp(b.z, 0.0f, 1.0f) * 31.0f);

  *ab = ((d << 11) | (e << 5) | f);
  b.x = d * 0.03227752766457f;
  b.y = e * 0.01583151765563f;
  b.z = f * 0.03227752766457f;
  return b;
}

float G(__local const float4* i, uint ac, ushort* ad, ushort* ae, float4 l,
        __constant float* af, __constant int* ag, float ah) {
  float4 ai = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
  int aj = 0;

  for (int c = 0; c < 16; c++) {
    const uint ak = ac >> (2 * c);

    ai += af[ak & 3] * i[c];
    aj += ag[ak & 3];
  }

  float al = (aj >> 16);
  float am = ((aj >> 8) & 0xff);
  float an = ((aj >> 0) & 0xff);
  float4 ao = ah * l - ai;

  const float ap = 1.0f / (al * am - an * an);

  float4 aq = (ai * am - ao * an) * ap;
  float4 ar = (ao * al - ai * an) * ap;

  aq = F(aq, ad);
  ar = F(ar, ae);

  float4 as =
      aq * aq * al + ar * ar * am + 2.0f * (aq * ar * an - aq * ai - ar * ao);

  return (1.0f / ah) * (as.x + as.y + as.z);
}

float H(__local const float4* i, uint ac, ushort* ad, ushort* ae, float4 l,
        __constant float* at, __constant int* au) {
  float4 ai = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
  int aj = 0;

  for (int c = 0; c < 16; c++) {
    const uint ak = ac >> (2 * c);

    ai += at[ak & 3] * i[c];
    aj += au[ak & 3];
  }

  float al = (aj >> 16);
  float am = ((aj >> 8) & 0xff);
  float an = ((aj >> 0) & 0xff);
  float4 ao = 4.0f * l - ai;

  const float ap = 1.0f / (al * am - an * an);

  float4 aq = (ai * am - ao * an) * ap;
  float4 ar = (ao * al - ai * an) * ap;

  aq = F(aq, ad);
  ar = F(ar, ae);

  float4 as =
      aq * aq * al + ar * ar * am + 2.0f * (aq * ar * an - aq * ai - ar * ao);

  return (0.25f) * (as.x + as.y + as.z);
}

uint4 I(__local const float4* i, __global const unsigned int* av,
        __local float* aw, float4 l, __local uint* ax, __constant float* af,
        __constant int* ag, __constant float* at, __constant int* au) {
  const int k = get_local_id(0);

  uint ay;
  uint az;
  uint ba;
  uint v;

  float bb = 0x1.fffffep127f;

  for (int c = 0; c < 16; c++) {
    int bc = k + 64 * c;
    if (bc >= 992) break;

    ushort ad, ae;
    uint ac = av[bc];
    if (bc < 160) ax[bc] = ac;

    float bd = G(i, ac, &ad, &ae, l, af, ag, 9.0f);
    if (bd < bb) {
      bb = bd;
      ba = ac;
      ay = ad;
      az = ae;
    }
  }

  if (ay < az) {
    v = az;
    az = ay;
    ay = v;

    ba ^= 0x55555555;
  }

  for (int c = 0; c < 3; c++) {
    int bc = k + 64 * c;
    if (bc >= 160) break;

    ushort ad, ae;
    uint ac = ax[bc];
    float bd = G(i, ac, &ad, &ae, l, at, au, 4.0f);
    if (bd < bb) {
      bb = bd;
      ba = ac;
      ay = ad;
      az = ae;

      if (ay > az) {
        v = az;
        az = ay;
        ay = v;

        ba ^= (~ba >> 1) & 0x55555555;
      }
    }
  }

  aw[k] = bb;

    uint4 be = (uinbaurn be;
}

int J(__local float* aw, __local int* bf) {
  const int k = get_local_id(0);

  bf[k] = k;

  for (int o = 64 / 2; o > 32; o >>= 1) {
    barrier(1);

    if (k < o) {
      float bg = aw[k];
      float bh = aw[k + o];

      if (bh < bg) {
        aw[k] = bh;
        bf[k] = bf[k + o];
      }
    }
  }

  barrier(1);

  if (k < 32) {
    if (aw[k + 32] < aw[k]) {
      aw[k] = aw[k + 32];
      bf[k] = bf[k + 32];
    }
    if (aw[k + 16] < aw[k]) {
      aw[k] = aw[k + 16];
      bf[k] = bf[k + 16];
    }
    if (aw[k + 8] < aw[k]) {
      aw[k] = aw[k + 8];
      bf[k] = bf[k + 8];
    }
    if (aw[k + 4] < aw[k]) {
      aw[k] = aw[k + 4];
      bf[k] = bf[k + 4];
    }
    if (aw[k + 2] < aw[k]) {
      aw[k] = aw[k + 2];
      bf[k] = bf[k + 2];
    }
    if (aw[k + 1] < aw[k]) {
      aw[k] = aw[k + 1];
      bf[k] = bf[k + 1];
    }
  }

  barrier(1);

  return bf[0];
}

void K(uint ad, uint ae, uint ac, __local int* u, __global uint2* be, int w) {
  const int x = get_group_id(0) + w;

  if (ad == ae) {
    ac = 0;
  }

  uint bf = 0;
  for (int c = 0; c < 16; c++) {
    int bi = u[c];
    bf |= ((ac >> (2 * bi)) & 3) << (2 * c);
  }

  be[x].x = (ae << 16) | ad;

  be[x].y = bf;
}

__kernel void L(__global const uint* av, __global const uint* t,
                __global uint2* be, __constant float* af, __constant int* ag,
                __constant float* at, __constant int* au, int w) {
  __local float4 i[16];
  __local float4 j[16];
  __local int bj[64];
  __local float bk[16 * 6];
  __local uint ax[160];
  __local int u[16];

  const int k = get_local_id(0);

  E(t, i, j, u, bk, w);

  barrier(1);

  uint4 bl = I(i, av, bk, j[0], ax, af, ag, at, au);

  const int bm = J(bk, bj);

  barrier(1);

  if (k == bm) {
    K(bl.x, bl.y, bl.z, u, be, w);
  }
}
