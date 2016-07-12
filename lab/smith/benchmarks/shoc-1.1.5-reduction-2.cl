__kernel void B(__global float *a, __global float *b, unsigned int d) {
  float j = 0.0f;
  for (int f = 0; f < d; f++) {
    j += a[f];
  }
  b[0] = j;
}
