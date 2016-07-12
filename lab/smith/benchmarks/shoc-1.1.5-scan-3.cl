__kernel void D(__global const float *a, __global const float *b,
                __global float *p, const int c, __local float *d) {
  __local float q;
  q = 0;

  __global float4 *r = (__global float4 *)a;
  __global float4 *s = (__global float4 *)p;
  int t = c / 4;

  int e = t / get_num_groups(0);
  int f = get_group_id(0) * e;

  int g = (get_group_id(0) == get_num_groups(0) - 1) ? t : f + e;

  int i = f + get_local_id(0);
  unsigned int u = f;

  float v = b[get_group_id(0)];

  while (u < g) {
    float4 w;
    if (i < g) {
      w = r[i];
    } else {
      w.x = 0.0f;
      w.y = 0.0f;
      w.z = 0.0f;
      w.w = 0.0f;
    }

    w.y += w.x;
    w.z += w.y;
    w.w += w.z;

    float x = B(w.w, d, 1);

    w.x += x + v;
    w.y += x + v;
    w.z += x + v;
    w.w += x + v;

    if (i < g) {
      s[i] = w;
    }

    barrier(1);
    if (get_local_id(0) == get_local_size(0) - 1) {
      q = w.w;
    }
    barrier(1);

    v = q;

    u += get_local_size(0);
    i += get_local_size(0);
  }
}
