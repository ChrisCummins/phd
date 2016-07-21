__kernel void A(float a, float b, __global float* c, __global float* d, int e, int f, int g) {
  int h = get_global_id(0) + 1;
  int i = get_global_id(1) + 1;
  int j = get_global_id(2) + 1;

  if (h < e - 1) {
    d[((h) + e * ((i) + f * (j)))] = b * (c[((h) + e * ((i) + f * (j + 1)))] + c[((h) + e * ((i) + f * (j - 1)))] + c[((h) + e * ((i + 1) + f * (j)))] + c[((h) + e * ((i - 1) + f * (j)))] + c[((h + 1) + e * ((i) + f * (j)))] + c[((h - 1) + e * ((i) + f * (j)))]) - c[((h) + e * ((i) + f * (j)))] * a;
  }
}