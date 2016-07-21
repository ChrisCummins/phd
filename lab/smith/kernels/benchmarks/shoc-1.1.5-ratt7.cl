__kernel void A(__global const float* a, __global const float* b, __global float* c, __global const float* d, const float e) {
  const float f = a[get_global_id(0)] * e;
  const float g = log(f);

  const float h = 1e+20f;

  const float i = 8.31451e7f;
  const float j = 1.01325e6f;
  const float k = ((j) * (1.0f / ((i * (f)))));

  float l;

  l = ((((d[(((2) - 1) * (8)) + (get_global_id(0))]) * (d[(((26) - 1) * (8)) + (get_global_id(0))]) * k)) * (1.0f / ((d[(((27) - 1) * (8)) + (get_global_id(0))]))));
  (c[(((126) - 1) * (8)) + (get_global_id(0))]) = (b[(((126) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((2) - 1) * (8)) + (get_global_id(0))]) * (d[(((26) - 1) * (8)) + (get_global_id(0))]))) * (1.0f / (((d[(((1) - 1) * (8)) + (get_global_id(0))]) * (d[(((25) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((127) - 1) * (8)) + (get_global_id(0))]) = (b[(((127) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((2) - 1) * (8)) + (get_global_id(0))]) * (d[(((26) - 1) * (8)) + (get_global_id(0))]))) * (1.0f / (((d[(((12) - 1) * (8)) + (get_global_id(0))]) * (d[(((14) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((128) - 1) * (8)) + (get_global_id(0))]) = (b[(((128) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((3) - 1) * (8)) + (get_global_id(0))]) * (d[(((26) - 1) * (8)) + (get_global_id(0))]))) * (1.0f / (((d[(((5) - 1) * (8)) + (get_global_id(0))]) * (d[(((25) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((129) - 1) * (8)) + (get_global_id(0))]) = (b[(((129) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((3) - 1) * (8)) + (get_global_id(0))]) * (d[(((26) - 1) * (8)) + (get_global_id(0))]))) * (1.0f / (((d[(((10) - 1) * (8)) + (get_global_id(0))]) * (d[(((15) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((130) - 1) * (8)) + (get_global_id(0))]) = (b[(((130) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((5) - 1) * (8)) + (get_global_id(0))]) * (d[(((26) - 1) * (8)) + (get_global_id(0))]))) * (1.0f / (((d[(((6) - 1) * (8)) + (get_global_id(0))]) * (d[(((25) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((131) - 1) * (8)) + (get_global_id(0))]) = (b[(((131) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((2) - 1) * (8)) + (get_global_id(0))]) * (d[(((21) - 1) * (8)) + (get_global_id(0))]) * k)) * (1.0f / ((d[(((22) - 1) * (8)) + (get_global_id(0))]))));
  (c[(((132) - 1) * (8)) + (get_global_id(0))]) = (b[(((132) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((2) - 1) * (8)) + (get_global_id(0))]) * (d[(((21) - 1) * (8)) + (get_global_id(0))]))) * (1.0f / (((d[(((1) - 1) * (8)) + (get_global_id(0))]) * (d[(((19) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((133) - 1) * (8)) + (get_global_id(0))]) = (b[(((133) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((2) - 1) * (8)) + (get_global_id(0))]) * (d[(((21) - 1) * (8)) + (get_global_id(0))]))) * (1.0f / (((d[(((1) - 1) * (8)) + (get_global_id(0))]) * (d[(((20) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((134) - 1) * (8)) + (get_global_id(0))]) = (b[(((134) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((3) - 1) * (8)) + (get_global_id(0))]) * (d[(((21) - 1) * (8)) + (get_global_id(0))]))) * (1.0f / (((d[(((2) - 1) * (8)) + (get_global_id(0))]) * (d[(((26) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((135) - 1) * (8)) + (get_global_id(0))]) = (b[(((135) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((3) - 1) * (8)) + (get_global_id(0))]) * (d[(((21) - 1) * (8)) + (get_global_id(0))]))) * (1.0f / (((d[(((12) - 1) * (8)) + (get_global_id(0))]) * (d[(((14) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((136) - 1) * (8)) + (get_global_id(0))]) = (b[(((136) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((5) - 1) * (8)) + (get_global_id(0))]) * (d[(((21) - 1) * (8)) + (get_global_id(0))]))) * (1.0f / (((d[(((6) - 1) * (8)) + (get_global_id(0))]) * (d[(((19) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((137) - 1) * (8)) + (get_global_id(0))]) = (b[(((137) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((4) - 1) * (8)) + (get_global_id(0))]) * (d[(((21) - 1) * (8)) + (get_global_id(0))]))) * (1.0f / (((d[(((7) - 1) * (8)) + (get_global_id(0))]) * (d[(((19) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((138) - 1) * (8)) + (get_global_id(0))]) = (b[(((138) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((4) - 1) * (8)) + (get_global_id(0))]) * (d[(((21) - 1) * (8)) + (get_global_id(0))]))) * (1.0f / (((d[(((3) - 1) * (8)) + (get_global_id(0))]) * (d[(((27) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((139) - 1) * (8)) + (get_global_id(0))]) = (b[(((139) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((4) - 1) * (8)) + (get_global_id(0))]) * (d[(((21) - 1) * (8)) + (get_global_id(0))]))) * (1.0f / (((d[(((16) - 1) * (8)) + (get_global_id(0))]) * (d[(((17) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((140) - 1) * (8)) + (get_global_id(0))]) = (b[(((140) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((7) - 1) * (8)) + (get_global_id(0))]) * (d[(((21) - 1) * (8)) + (get_global_id(0))]))) * (1.0f / (((d[(((5) - 1) * (8)) + (get_global_id(0))]) * (d[(((27) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((141) - 1) * (8)) + (get_global_id(0))]) = (b[(((141) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((8) - 1) * (8)) + (get_global_id(0))]) * (d[(((21) - 1) * (8)) + (get_global_id(0))]))) * (1.0f / (((d[(((7) - 1) * (8)) + (get_global_id(0))]) * (d[(((22) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((142) - 1) * (8)) + (get_global_id(0))]) = (b[(((142) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((16) - 1) * (8)) + (get_global_id(0))]) * (d[(((21) - 1) * (8)) + (get_global_id(0))]))) * (1.0f / (((d[(((14) - 1) * (8)) + (get_global_id(0))]) * (d[(((22) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((143) - 1) * (8)) + (get_global_id(0))]) = (b[(((143) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((12) - 1) * (8)) + (get_global_id(0))]) * (d[(((21) - 1) * (8)) + (get_global_id(0))]))) * (1.0f / (((d[(((13) - 1) * (8)) + (get_global_id(0))]) * (d[(((19) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((144) - 1) * (8)) + (get_global_id(0))]) = (b[(((144) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((12) - 1) * (8)) + (get_global_id(0))]) * (d[(((21) - 1) * (8)) + (get_global_id(0))]) * k)) * (1.0f / ((d[(((30) - 1) * (8)) + (get_global_id(0))]))));
  (c[(((145) - 1) * (8)) + (get_global_id(0))]) = (b[(((145) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((12) - 1) * (8)) + (get_global_id(0))]) * (d[(((21) - 1) * (8)) + (get_global_id(0))]))) * (1.0f / (((d[(((2) - 1) * (8)) + (get_global_id(0))]) * (d[(((29) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((146) - 1) * (8)) + (get_global_id(0))]) = (b[(((146) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = (((d[(((27) - 1) * (8)) + (get_global_id(0))])) * (1.0f / (((d[(((12) - 1) * (8)) + (get_global_id(0))]) * (d[(((14) - 1) * (8)) + (get_global_id(0))]) * k))));
  (c[(((147) - 1) * (8)) + (get_global_id(0))]) = (b[(((147) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((2) - 1) * (8)) + (get_global_id(0))]) * (d[(((27) - 1) * (8)) + (get_global_id(0))]) * k)) * (1.0f / ((d[(((28) - 1) * (8)) + (get_global_id(0))]))));
  (c[(((148) - 1) * (8)) + (get_global_id(0))]) = (b[(((148) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((2) - 1) * (8)) + (get_global_id(0))]) * (d[(((27) - 1) * (8)) + (get_global_id(0))]))) * (1.0f / (((d[(((12) - 1) * (8)) + (get_global_id(0))]) * (d[(((16) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((149) - 1) * (8)) + (get_global_id(0))]) = (b[(((149) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((2) - 1) * (8)) + (get_global_id(0))]) * (d[(((27) - 1) * (8)) + (get_global_id(0))]))) * (1.0f / (((d[(((1) - 1) * (8)) + (get_global_id(0))]) * (d[(((26) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((150) - 1) * (8)) + (get_global_id(0))]) = (b[(((150) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);
}