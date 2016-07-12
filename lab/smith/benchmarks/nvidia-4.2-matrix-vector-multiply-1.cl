__kernel void A(const __global float* a, const __global float* b, uint c,
                uint d, __global float* e) {
  uint f = get_global_id(0);
  if (f < d) {
    const __global float* g = a + f * c;

    float h = 0;
    for (int i = 0; i < c; ++i) h += g[i] * b[i];

    e[f] = h;
  }
}