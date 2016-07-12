__kernel void C(__global const float* restrict a,

                __global const float* restrict b,

                __global const int* restrict c, __global const int* restrict u,
                const int e, __global float* restrict f) {
  int h = get_global_id(0);

  if (h < e) {
    float v = 0.0;
    int w = u[h];
    for (int x = 0; x < w; x++) {
      int y = x * e + h;

      v += a[y] * b[c[y]];
    }
    f[h] = v;
  }
}
