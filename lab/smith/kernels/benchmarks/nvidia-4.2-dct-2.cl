__kernel void A(__global float *a, __global float *b, uint c, uint d, uint e) {
  __local float f[16][32 + 1];
  const uint g = get_local_id(0);
  const uint h = 8 * get_local_id(1);
  const uint i = g & (8 - 1);
  const uint j = get_group_id(0) * 32 + g;
  const uint k = get_group_id(1) * 16 + h;

  if ((j - i + 8 - 1 >= e) || (k + 8 - 1 >= d)) return;

  __local float *l = &f[h + 0][g + 0];
  __local float *m = &f[h + i][g - i];
  b += k * c + j;
  a += k * c + j;

  float n[8];
  for (uint o = 0; o < 8; o++) l[o * (32 + 1)] = b[o * c];

  for (uint o = 0; o < 8; o++) n[o] = m[o];
  B(n);
  for (uint o = 0; o < 8; o++) m[o] = n[o];

  for (uint o = 0; o < 8; o++) n[o] = l[o * (32 + 1)];
  B(n);
  for (uint o = 0; o < 8; o++) a[o * c] = n[o];
}