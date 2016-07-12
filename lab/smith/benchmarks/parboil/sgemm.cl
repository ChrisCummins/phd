__kernel void A(__global const float *a, int b, __global const float *c, int d,
                __global float *e, int f, int g, float h, float i) {
  float j = 0.0f;
  int k = get_global_id(0);
  int l = get_global_id(1);

  for (int m = 0; m < g; ++m) {
    float n = a[k + m * b];
    float o = c[l + m * d];
    j += n * o;
  }
  e[k + l * f] = e[k + l * f] * i + h * j;
}