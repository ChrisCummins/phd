__kernel void A(__global const float* restrict a, __global const float* restrict b, __global const int* restrict c, __global const int* restrict d, const int e, __global float* restrict f) {
  int g = get_global_id(0);

  if (g < e) {
    float h = 0.0;
    int i = d[g];
    for (int j = 0; j < i; j++) {
      int k = j * e + g;

      h += a[k] * b[c[k]];
    }
    f[g] = h;
  }
}
