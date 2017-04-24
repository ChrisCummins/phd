__kernel void A(__global int* a, const int b) {
  int c = get_global_id(0);
  if (c < b)
    a[c] = b;
}