__kernel void A(global uint *a, global uint *b, const int c) {
  int d = get_global_id(0);
  int e = c * ((1 << 10) * (1));
  int f = 0;

  for (int g = d; g < e; g += (1 << 10)) {
    int h = a[g];
    a[g] = f;
    f += h;
  }

  b[d] = f;
}