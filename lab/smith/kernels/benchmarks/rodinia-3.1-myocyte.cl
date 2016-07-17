__kernel void A(int a, __global float *b, __global float *c, __global float *d,
                __global float *e) {
  int f;
  int g;

  int h;
  int i;
  int j;

  float k;
  float l;
  float m;
  f = get_group_id(0);
  g = get_local_id(0);

  if (f == 0) {
    if (g == 0) {
      h = 0;

      Foo(a, b, c, h, d);
    }

  }

  else if (f == 1) {
    if (g == 0) {
      h = 46;
      i = 0;
      j = 0;
      k = b[35] * 1e3;

      B(a, b, c, h, d, i, e, j, k);

      h = 61;
      i = 5;
      j = 1;
      l = b[36] * 1e3;

      B(a, b, c, h, d, i, e, j, l);

      h = 76;
      i = 10;
      j = 2;
      m = b[37] * 1e3;

      B(a, b, c, h, d, i, e, j, m);
    }
  }
}