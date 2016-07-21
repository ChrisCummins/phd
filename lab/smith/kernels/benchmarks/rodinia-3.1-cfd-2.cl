__kernel void A(__global float* a, __constant float* b, int c) {
  const int d = get_global_id(0);
  if (d >= c)
    return;
  for (int e = 0; e < ((1 + 3) + 1); e++) a[d + e * c] = b[e];
}