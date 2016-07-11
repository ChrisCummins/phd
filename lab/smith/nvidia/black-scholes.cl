float A(float a);
void B(__global float *b, __global float *c, float d, float e, float f, float g,
       float h);

float A(float a) {
  const float i = 0.31938153f;
  const float j = -0.356563782f;
  const float k = 1.781477937f;
  const float l = -1.821255978f;
  const float m = 1.330274429f;
  const float n = 0.39894228040143267793994605993438f;

  float o = 1.0f / (1.0f + 0.2316419f * __clc_fabs(a));

  float p =
      n * exp(-0.5f * a * a) * (o * (i + o * (j + o * (k + o * (l + o * m)))));

  if (a > 0) p = 1.0f - p;

  return p;
}

void B(__global float *b, __global float *c, float d, float e, float f, float g,
       float h) {
  float q = sqrt(f);
  float r = (log(d / e) + (g + 0.5f * h * h) * f) / (h * q);
  float s = r - h * q;
  float t = A(r);
  float u = A(s);

  float v = exp(-g * f);
  *b = (d * t - e * v * u);
  *c = (e * v * (1.0f - u) - d * (1.0f - t));
}

__kernel void C(__global float *w, __global float *x, __global float *y,
                __global float *z, __global float *aa, float g, float h,
                unsigned int ab) {
  for (unsigned int ac = get_global_id(0); ac < ab; ac += get_global_size(0))
    B(&w[ac], &x[ac], y[ac], z[ac], aa[ac], g, h);
}
