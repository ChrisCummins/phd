__kernel void A(__global char* a, short b, int c) {
  const int d = get_global_id(0);
  if (d >= c)
    return;
  a[d] = b;
}