__kernel void A(unsigned int a, unsigned int b, unsigned int c, unsigned int d, int e, int f, int g, __global int *h, __global unsigned char *i, __global unsigned int *j) {
  int k = (get_group_id(0) * get_local_size(0)) + get_local_id(0);

  int l = k * g;
  unsigned char m[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  B(l, f, g, m);

  for (int n = 0; n < g && l + n < e; ++n) {
    unsigned int o[4];
    if (o[0] == a && o[1] == b && o[2] == c && o[3] == d) {
      *h = l + n;
      i[0] = m[0];
      i[1] = m[1];
      i[2] = m[2];
      i[3] = m[3];
      i[4] = m[4];
      i[5] = m[5];
      i[6] = m[6];
      i[7] = m[7];
      j[0] = o[0];
      j[1] = o[1];
      j[2] = o[2];
      j[3] = o[3];
    }
    ++m[0];
  }
}