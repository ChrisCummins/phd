__kernel void A(__global uint2* a, __global uint* b, __global uint* c, uint d, uint e, uint f, __local uint* g) {
  __local uint h[16];

  uint i = get_group_id(0);
  uint j = get_local_id(0);
  uint k = get_local_size(0);

  uint2 l;

  l = a[get_global_id(0)];

  g[2 * j] = (l.x >> d) & 0xF;
  g[2 * j + 1] = (l.y >> d) & 0xF;

  if (j < 16) {
    h[j] = 0;
  }
  barrier(1);

  if ((j > 0) && (g[j] != g[j - 1])) {
    h[g[j]] = j;
  }
  if (g[j + k] != g[j + k - 1]) {
    h[g[j + k]] = j + k;
  }
  barrier(1);

  if (j < 16) {
    c[i * 16 + j] = h[j];
  }
  barrier(1);

  if ((j > 0) && (g[j] != g[j - 1])) {
    h[g[j - 1]] = j - h[g[j - 1]];
  }
  if (g[j + k] != g[j + k - 1]) {
    h[g[j + k - 1]] = j + k - h[g[j + k - 1]];
  }

  if (j == k - 1) {
    h[g[2 * k - 1]] = 2 * k - h[g[2 * k - 1]];
  }
  barrier(1);

  if (j < 16) {
    b[j * f + i] = h[j];
  }
}