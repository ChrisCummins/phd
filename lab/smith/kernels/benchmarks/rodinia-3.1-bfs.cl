__kernel void A(__global char* a, __global char* b, __global char* c,
                __global char* d, const int e) {
  int f = get_global_id(0);
  if (f < e && b[f]) {
    a[f] = true;
    c[f] = true;
    *d = true;
    b[f] = false;
  }
}