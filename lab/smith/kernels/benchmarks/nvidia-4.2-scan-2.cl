__kernel void A(__global uint *a, __global uint *b, __global uint *c, __local uint *d, uint e, uint f) {
  uint g = 0;
  if (get_global_id(0) < e)
    g = b[(4 * 256 - 1) + (4 * 256) * get_global_id(0)] + c[(4 * 256 - 1) + (4 * 256) * get_global_id(0)];

  uint h = B(g, d, f);

  if (get_global_id(0) < e)
    a[get_global_id(0)] = h;
}