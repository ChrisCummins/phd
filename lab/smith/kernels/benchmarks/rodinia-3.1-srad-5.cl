__kernel void A(float a, int b, int c, long d, __global int* e, __global int* f,
                __global int* g, __global int* h, __global float* i,
                __global float* j, __global float* k, __global float* l,
                __global float* m, __global float* n) {
  int o = get_group_id(0);
  int p = get_local_id(0);
  int q = o * 256 + p;
  int r;
  int s;

  float t, u, v, w;
  float x;

  r = (q + 1) % b - 1;
  s = (q + 1) / b + 1 - 1;
  if ((q + 1) % b == 0) {
    r = b - 1;
    s = s - 1;
  }

  if (q < d) {
    t = m[q];
    u = m[f[r] + b * s];
    v = m[q];
    w = m[r + b * g[s]];

    x = t * i[q] + u * j[q] + v * l[q] + w * k[q];

    n[q] = n[q] + 0.25 * a * x;
  }
}