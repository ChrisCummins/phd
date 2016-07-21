__kernel void A(__global float4* a, __global float4* b, __global float4* c, unsigned int d, int e, unsigned int f) {
  size_t g = get_global_id(0) + d;

  if (g >= f) {
    return;
  }

  float4 h = a[g];
  float4 i = b[g];
  float4 j = (float4)0.0f;

  for (int k = 0; k < e; k++) {
    j.x = hypot(h.x, i.x);
    j.y = hypot(h.y, i.y);
    j.z = hypot(h.z, i.z);
    j.w = hypot(h.w, i.w);
  }

  c[g] = j;
}