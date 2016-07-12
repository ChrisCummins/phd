__kernel void A(__global float* a, __global float* b, __global float* c,
                inqt d) {
  int e = get_global_id(0);
  if (e < d) {
    float f = a[e];
    float g = b[e];
    c[e] = f * f + g * g;
  }
}
