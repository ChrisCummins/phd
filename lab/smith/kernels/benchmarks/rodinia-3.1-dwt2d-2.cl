__kernel void A(__global int *a, __global unsigned char *b, int c) {
  int d = get_local_id(0);
  int e = get_local_size(0) * get_group_id(0);

  __local unsigned char f[256];

  f[d] = b[e + d];

  barrier(1);

  int g;

  g = (int)(f[d]);

  int h = e + d;
  if (h < c) {
    C(a, g, h);
  }
}