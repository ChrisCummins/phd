struct kValues {
  float Kx;
  float Ky;
  float Kz;
  float PhiMag;
};
__kernel void A(__global float* a, __global float* b, __global float* c,
                int d) {
  int e = get_global_id(0);
  if (e < d) {
    float f = a[e];
    float g = b[e];
    c[e] = f * f + g * g;
  }
}

__kernel void B(int d, int h, __global float* i, __global float* j,
                __global float* k, __global float* l, __global float* m,
                __global struct kValues* n) {
  float o[4];
  float p[4];
  float q[4];
  float r[4];
  float s[4];

  for (int t = 0; t < 4; t++) {
    int u = get_group_id(0) * 256 + 4 * get_local_id(0) + t;

    o[t] = i[u];
    p[t] = j[u];
    q[t] = k[u];
    r[t] = l[u];
    s[t] = m[u];
  }

  int v = 0;
  for (; (v < 1024) && (h < d); v++, h++) {
    float w = n[v].Kx;
    float x = n[v].Ky;
    float y = n[v].Kz;
    float z = n[v].PhiMag;

    for (int t = 0; t < 4; t++) {
      float aa = 6.2831853071795864769252867665590058f *
                 (w * o[t] + x * p[t] + y * q[t]);
      r[t] += z * cos(aa);
      s[t] += z * sin(aa);
    }
  }

  for (int t = 0; t < 4; t++) {
    int u = get_group_id(0) * 256 + 4 * get_local_id(0) + t;
    l[u] = r[t];
    m[u] = s[t];
  }
}