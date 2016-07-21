__kernel void A(__global uint* a, __global uint* b, uint c, __local uint* d) {
  int e = get_local_id(0);

  int f = 0;
  int g = 1;

  d[f * c + e] = (e > 0) ? b[e - 1] : 0;

  for (int h = 1; h < c; h *= 2) {
    f = 1 - f;
    g = 1 - f;
    barrier(1);

    d[f * c + e] = d[g * c + e];

    if (e >= h)
      d[f * c + e] += d[g * c + e - h];
  }

  barrier(1);

  a[e] = d[f * c + e];
}