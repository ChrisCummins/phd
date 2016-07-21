__kernel void A(__global float* const a, __global const float* const b, __constant float* const c, const int d, const int e, const int f, const int g) {
  bool h = true;
  const int i = get_global_id(0);
  const int j = get_global_id(1);
  const int k = get_local_id(0);
  const int l = get_local_id(1);
  const int m = get_local_size(0);
  const int n = get_local_size(1);
  __local float o[8 + 2 * 8][8 + 2 * 8];

  const int p = d + 2 * 8;
  const int q = p * (e + 2 * 8);

  int r = 0;
  int s = 0;

  r += 8 * p + 8 + g;

  r += j * p + i;

  float t[8];
  float u[8];
  float v;

  const int w = k + 8;
  const int x = l + 8;

  if (i >= d)
    h = false;
  if (j >= e)
    h = false;

  for (int y = 8 - 2; y >= 0; y--) {
    u[y] = b[r];
    r += q;
  }

  v = b[r];
  s = r;
  r += q;

  for (int y = 0; y < 8; y++) {
    t[y] = b[r];
    r += q;
  }

  for (int z = 0; z < f; z++) {
    for (int y = 8 - 1; y > 0; y--) u[y] = u[y - 1];
    u[0] = v;
    v = t[0];
    for (int y = 0; y < 8 - 1; y++) t[y] = t[y + 1];
    t[8 - 1] = b[r];

    r += q;
    s += q;
    barrier(1);
    if (l < 8) {
      o[l][w] = b[s - 8 * p];
      o[l + n + 8][w] = b[s + n * p];
    }

    if (k < 8) {
      o[x][k] = b[s - 8];
      o[x][k + m + 8] = b[s + m];
    }
    o[x][w] = v;
    barrier(1);

    float aa = c[0] * v;
    for (int y = 1; y <= 8; y++) {
      aa += c[y] * (t[y - 1] + u[y - 1] + o[x - y][w] + o[x + y][w] + o[x][w - y] + o[x][w + y]);
    }

    if (h)
      a[s] = aa;
  }
}