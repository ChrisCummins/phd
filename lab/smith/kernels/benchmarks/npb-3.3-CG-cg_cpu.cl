typedef bool logical;
__kernel void A(__global int *a, __global int *b, int c, int d) {
  int e = get_global_id(0);
  int f = get_global_size(0);
  int g = (d + f - 1) / f;
  int h = e * g;
  int i = h + g;
  if (i > d)
    i = d;

  for (int j = h; j < i; j++) {
    for (int k = b[j]; k < b[j + 1]; k++) {
      a[k] = a[k] - c;
    }
  }
}

__kernel void B(__global double *l) {
  int d = 14000 + 1;
  int e = get_global_id(0);
  int f = get_global_size(0);
  int g = (d + f - 1) / f;
  int h = e * g;
  int i = h + g;
  if (i > d)
    i = d;

  for (int j = h; j < i; j++) {
    l[j] = 1.0;
  }
}

__kernel void C(__global double *m, __global double *n, __global double *o, __global double *p, int d) {
  int e = get_global_id(0);
  int f = get_global_size(0);
  int g = (d + f - 1) / f;
  int h = e * g;
  int i = h + g;
  if (i > d)
    i = d;

  for (int j = h; j < i; j++) {
    m[j] = 0.0;
    n[j] = 0.0;
    o[j] = 0;
    p[j] = 0;
  }
}

__kernel void D(__global double *l, __global double *n, __global double *q, __global double *r, __local double *s, __local double *t, int d) {
  int e = get_global_id(0);
  int f = get_global_size(0);
  int g = (d + f - 1) / f;
  int h = e * g;
  int i = h + g;
  if (i > d)
    i = d;

  double u = 0.0;
  double v = 0.0;
  for (int j = h; j < i; j++) {
    u = u + l[j] * n[j];
    v = v + n[j] * n[j];
  }
  q[e] = u;
  r[e] = v;
}

__kernel void E(__global double *l, __global double *n, double v, int d) {
  int e = get_global_id(0);
  int f = get_global_size(0);
  int g = (d + f - 1) / f;
  int h = e * g;
  int i = h + g;
  if (i > d)
    i = d;

  for (int j = h; j < i; j++) {
    l[j] = v * n[j];
  }
}
__kernel void F(__global double *m, __global double *n, __global double *o, __global double *l, __global double *p) {
  int d = 14000 + 1;
  int e = get_global_id(0);
  int f = get_global_size(0);
  int g = (d + f - 1) / f;
  int h = e * g;
  int i = h + g;
  if (i > d)
    i = d;

  for (int j = h; j < i; j++) {
    m[j] = 0.0;
    n[j] = 0.0;
    double w = l[j];
    o[j] = w;
    p[j] = w;
  }
}

__kernel void G(__global double *o, __global double *x, int d) {
  int e = get_global_id(0);
  int f = get_global_size(0);
  int g = (d + f - 1) / f;
  int h = e * g;
  int i = h + g;
  if (i > d)
    i = d;

  double y = 0.0;
  for (int j = h; j < i; j++) {
    y = y + o[j] * o[j];
  }
  x[e] = y;
}

__kernel void H(__global int *b, __global double *z, __global double *p, __global int *a, __global double *m, int d) {
  int e = get_global_id(0);
  int f = get_global_size(0);
  int g = (d + f - 1) / f;
  int h = e * g;
  int i = h + g;
  if (i > d)
    i = d;

  for (int j = h; j < i; j++) {
    double aa = 0.0;
    for (int k = b[j]; k < b[j + 1]; k++) {
      aa = aa + z[k] * p[a[k]];
    }
    m[j] = aa;
  }
}

__kernel void I(__global double *p, __global double *m, __global double *ab, int d) {
  int e = get_global_id(0);
  int f = get_global_size(0);
  int g = (d + f - 1) / f;
  int h = e * g;
  int i = h + g;
  if (i > d)
    i = d;

  double ac = 0.0;
  for (int j = h; j < i; j++) {
    ac = ac + p[j] * m[j];
  }
  ab[e] = ac;
}

__kernel void J(__global double *p, __global double *m, __global double *o, __global double *n, __global double *x, double ad, int d) {
  int e = get_global_id(0);
  int f = get_global_size(0);
  int g = (d + f - 1) / f;
  int h = e * g;
  int i = h + g;
  if (i > d)
    i = d;

  double y = 0.0;
  for (int j = h; j < i; j++) {
    n[j] = n[j] + ad * p[j];
    o[j] = o[j] - ad * m[j];

    y = y + o[j] * o[j];
  }
  x[e] = y;
}

__kernel void K(__global double *p, __global double *o, const double ae, int d) {
  int e = get_global_id(0);
  int f = get_global_size(0);
  int g = (d + f - 1) / f;
  int h = e * g;
  int i = h + g;
  if (i > d)
    i = d;

  for (int j = h; j < i; j++) {
    p[j] = o[j] + ae * p[j];
  }
}

__kernel void L(__global int *b, __global double *z, __global double *n, __global int *a, __global double *o, int d) {
  int e = get_global_id(0);
  int f = get_global_size(0);
  int g = (d + f - 1) / f;
  int h = e * g;
  int i = h + g;
  if (i > d)
    i = d;

  for (int j = h; j < i; j++) {
    double aa = 0.0;
    for (int k = b[j]; k < b[j + 1]; k++) {
      aa = aa + z[k] * n[a[k]];
    }
    o[j] = aa;
  }
}

__kernel void M(__global double *l, __global double *o, __global double *af, int d) {
  int e = get_global_id(0);
  int f = get_global_size(0);
  int g = (d + f - 1) / f;
  int h = e * g;
  int i = h + g;
  if (i > d)
    i = d;

  double ag = 0.0;
  for (int j = h; j < i; j++) {
    double aa = l[j] - o[j];
    ag = ag + aa * aa;
  }
  af[e] = ag;
}