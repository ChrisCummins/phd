__kernel void A(__global float *a, __global float *b, __global float *c, float d, int e, int f, int g, float h, float i, float j, float k, float l, float m, float n) {
  float o = 80.0;

  int p = get_global_id(0);
  int q = get_global_id(1);
  int r = p + q * e;
  int s = e * f;

  int t = (p == 0) ? r : r - 1;
  int u = (p == e - 1) ? r : r + 1;
  int v = (q == 0) ? r : r - e;
  int w = (q == f - 1) ? r : r + e;

  float x, y, z;
  x = y = b[r];
  z = b[r + s];
  c[r] = n * y + i * b[t] + h * b[u] + k * b[w] + j * b[v] + m * x + l * z + d * a[r] + l * o;
  r += s;
  t += s;
  u += s;
  v += s;
  w += s;

  for (int aa = 1; aa < g - 1; ++aa) {
    x = y;
    y = z;
    z = b[r + s];
    c[r] = n * y + i * b[t] + h * b[u] + k * b[w] + j * b[v] + m * x + l * z + d * a[r] + l * o;
    r += s;
    t += s;
    u += s;
    v += s;
    w += s;
  }
  x = y;
  y = z;
  c[r] = n * y + i * b[t] + h * b[u] + k * b[w] + j * b[v] + m * x + l * z + d * a[r] + l * o;
  return;
}