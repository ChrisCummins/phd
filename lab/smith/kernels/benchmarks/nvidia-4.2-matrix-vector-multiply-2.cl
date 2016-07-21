__kernel void A(const __global float* a, const __global float* b, uint c, uint d, __global float* e) {
  for (uint f = get_global_id(0); f < d; f += get_global_size(0)) {
    const __global float* g = a + f * c;

    float h = 0;
    for (uint i = 0; i < c; ++i) h += g[i] * b[i];

    e[f] = h;
  }
}