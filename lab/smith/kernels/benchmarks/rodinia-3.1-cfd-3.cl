__kernel void A(__global float* a, __global float* b, __global float* c, int d) {
  const int e = get_global_id(0);
  if (e >= d)
    return;

  float f = a[e + 0 * d];
  float3 g;
  g.x = a[e + (1 + 0) * d];
  g.y = a[e + (1 + 1) * d];
  g.z = a[e + (1 + 2) * d];

  float h = a[e + (1 + 3) * d];

  float3 i;
  B(f, g, &i);
  float j = C(i);

  float k = D(f, h, j);
  float l = E(f, k);

  c[e] = (float)(0.5f) / (sqrt(b[e]) * (sqrt(j) + l));
}