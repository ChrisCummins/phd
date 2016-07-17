__kernel void A(__global float4 *a, __global float4 *b, const int c,
                __global int *d, const float e, const float f, const float g,
                const int h) {
  uint i = get_global_id(0);

  float4 j = b[i];
  float4 k = {0.0f, 0.0f, 0.0f, 0.0f};

  int l = 0;
  while (l < c) {
    int m = d[l * h + i];

    float4 n = b[m];

    float o = j.x - n.x;
    float p = j.y - n.y;
    float q = j.z - n.z;
    float r = o * o + p * p + q * q;

    if (r < e) {
      r = 1.0f / r;
      float s = r * r * r;
      float t = r * s * (f * s - g);

      k.x += o * t;
      k.y += p * t;
      k.z += q * t;
    }
    l++;
  }

  a[i] = k;
}