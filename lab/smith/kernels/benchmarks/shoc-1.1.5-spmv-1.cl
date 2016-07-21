__kernel void A(__global const float* restrict a, __global const float* restrict b, __global const int* restrict c, __global const int* restrict d, const int e, __global float* restrict f) {
  int g = get_global_id(0);

  if (g < e) {
    float h = 0;
    int i = d[g];
    int j = d[g + 1];
    for (int k = i; k < j; k++) {
      int l = c[k];

      h += a[k] * b[l];
    }
    f[g] = h;
  }
}
