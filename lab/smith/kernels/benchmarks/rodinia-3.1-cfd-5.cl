__kernel void A(int a, int b, __global float* c, __global float* d, __global float* e, __global float* f) {
  const int g = get_global_id(0);
  if (g >= b)
    return;

  float h = e[g] / (float)(3 + 1 - a);

  d[g + 0 * b] = c[g + 0 * b] + h * f[g + 0 * b];
  d[g + (1 + 3) * b] = c[g + (1 + 3) * b] + h * f[g + (1 + 3) * b];
  d[g + (1 + 0) * b] = c[g + (1 + 0) * b] + h * f[g + (1 + 0) * b];
  d[g + (1 + 1) * b] = c[g + (1 + 1) * b] + h * f[g + (1 + 1) * b];
  d[g + (1 + 2) * b] = c[g + (1 + 2) * b] + h * f[g + (1 + 2) * b];
}