__kernel void A(__global const float *a, __global const float *b, __global float *c, const int d, __local float *e) {
  __local float f;
  f = 0;

  __global float4 *g = (__global float4 *)a;
  __global float4 *h = (__global float4 *)c;
  int i = d / 4;

  int j = i / get_num_groups(0);
  int k = get_group_id(0) * j;

  int l = (get_group_id(0) == get_num_groups(0) - 1) ? i : k + j;

  int m = k + get_local_id(0);
  unsigned int n = k;

  float o = b[get_group_id(0)];

  while (n < l) {
    float4 p;
    if (m < l) {
      p = g[m];
    } else {
      p.x = 0.0f;
      p.y = 0.0f;
      p.z = 0.0f;
      p.w = 0.0f;
    }

    p.y += p.x;
    p.z += p.y;
    p.w += p.z;

    float q = B(p.w, e, 1);

    p.x += q + o;
    p.y += q + o;
    p.z += q + o;
    p.w += q + o;

    if (m < l) {
      h[m] = p;
    }

    barrier(1);
    if (get_local_id(0) == get_local_size(0) - 1) {
      f = p.w;
    }
    barrier(1);

    o = f;

    n += get_local_size(0);
    m += get_local_size(0);
  }
}