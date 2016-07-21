__kernel void A(__global const float* a, __global const float* b, __global float* c, __global const float* d, const float e) {
  const float f = a[get_global_id(0)] * e;
  const float g = log(f);

  const float h = 1e+20f;

  const float i = 8.31451e7f;
  const float j = 1.01325e6f;
  const float k = ((j) * (1.0f / ((i * (f)))));

  float l;

  l = ((((d[(((4) - 1) * (8)) + (get_global_id(0))]) * (d[(((18) - 1) * (8)) + (get_global_id(0))]))) * (1.0f / (((d[(((7) - 1) * (8)) + (get_global_id(0))]) * (d[(((17) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((101) - 1) * (8)) + (get_global_id(0))]) = (b[(((101) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((2) - 1) * (8)) + (get_global_id(0))]) * (d[(((13) - 1) * (8)) + (get_global_id(0))]))) * (1.0f / (((d[(((1) - 1) * (8)) + (get_global_id(0))]) * (d[(((12) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((102) - 1) * (8)) + (get_global_id(0))]) = (b[(((102) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((3) - 1) * (8)) + (get_global_id(0))]) * (d[(((13) - 1) * (8)) + (get_global_id(0))]))) * (1.0f / (((d[(((5) - 1) * (8)) + (get_global_id(0))]) * (d[(((12) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((103) - 1) * (8)) + (get_global_id(0))]) = (b[(((103) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((5) - 1) * (8)) + (get_global_id(0))]) * (d[(((13) - 1) * (8)) + (get_global_id(0))]))) * (1.0f / (((d[(((6) - 1) * (8)) + (get_global_id(0))]) * (d[(((12) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((104) - 1) * (8)) + (get_global_id(0))]) = (b[(((104) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((9) - 1) * (8)) + (get_global_id(0))]) * (d[(((13) - 1) * (8)) + (get_global_id(0))]))) * (1.0f / (((d[(((2) - 1) * (8)) + (get_global_id(0))]) * (d[(((22) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((105) - 1) * (8)) + (get_global_id(0))]) = (b[(((105) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((10) - 1) * (8)) + (get_global_id(0))]) * (d[(((13) - 1) * (8)) + (get_global_id(0))]))) * (1.0f / (((d[(((12) - 1) * (8)) + (get_global_id(0))]) * (d[(((12) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((106) - 1) * (8)) + (get_global_id(0))]) = (b[(((106) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((11) - 1) * (8)) + (get_global_id(0))]) * (d[(((13) - 1) * (8)) + (get_global_id(0))]))) * (1.0f / (((d[(((12) - 1) * (8)) + (get_global_id(0))]) * (d[(((12) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((107) - 1) * (8)) + (get_global_id(0))]) = (b[(((107) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((2) - 1) * (8)) + (get_global_id(0))]) * (d[(((25) - 1) * (8)) + (get_global_id(0))]))) * (1.0f / (((d[(((11) - 1) * (8)) + (get_global_id(0))]) * (d[(((14) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((108) - 1) * (8)) + (get_global_id(0))]) = (b[(((108) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((3) - 1) * (8)) + (get_global_id(0))]) * (d[(((25) - 1) * (8)) + (get_global_id(0))]))) * (1.0f / (((d[(((2) - 1) * (8)) + (get_global_id(0))]) * (d[(((14) - 1) * (8)) + (get_global_id(0))]) * (d[(((14) - 1) * (8)) + (get_global_id(0))]) * k))));
  (c[(((109) - 1) * (8)) + (get_global_id(0))]) = (b[(((109) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((4) - 1) * (8)) + (get_global_id(0))]) * (d[(((25) - 1) * (8)) + (get_global_id(0))]))) * (1.0f / (((d[(((5) - 1) * (8)) + (get_global_id(0))]) * (d[(((14) - 1) * (8)) + (get_global_id(0))]) * (d[(((14) - 1) * (8)) + (get_global_id(0))]) * k))));
  (c[(((110) - 1) * (8)) + (get_global_id(0))]) = (b[(((110) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((9) - 1) * (8)) + (get_global_id(0))]) * (d[(((25) - 1) * (8)) + (get_global_id(0))]))) * (1.0f / (((d[(((14) - 1) * (8)) + (get_global_id(0))]) * (d[(((19) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((111) - 1) * (8)) + (get_global_id(0))]) = (b[(((111) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((10) - 1) * (8)) + (get_global_id(0))]) * (d[(((25) - 1) * (8)) + (get_global_id(0))]))) * (1.0f / (((d[(((14) - 1) * (8)) + (get_global_id(0))]) * (d[(((21) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((112) - 1) * (8)) + (get_global_id(0))]) = (b[(((112) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((25) - 1) * (8)) + (get_global_id(0))]) * (d[(((25) - 1) * (8)) + (get_global_id(0))]))) * (1.0f / (((d[(((14) - 1) * (8)) + (get_global_id(0))]) * (d[(((14) - 1) * (8)) + (get_global_id(0))]) * (d[(((19) - 1) * (8)) + (get_global_id(0))]) * k))));
  (c[(((113) - 1) * (8)) + (get_global_id(0))]) = (b[(((113) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = (((d[(((19) - 1) * (8)) + (get_global_id(0))])) * (1.0f / ((d[(((20) - 1) * (8)) + (get_global_id(0))]))));
  (c[(((114) - 1) * (8)) + (get_global_id(0))]) = (b[(((114) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = (((d[(((21) - 1) * (8)) + (get_global_id(0))])) * (1.0f / (((d[(((2) - 1) * (8)) + (get_global_id(0))]) * (d[(((19) - 1) * (8)) + (get_global_id(0))]) * k))));
  (c[(((115) - 1) * (8)) + (get_global_id(0))]) = (b[(((115) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((3) - 1) * (8)) + (get_global_id(0))]) * (d[(((19) - 1) * (8)) + (get_global_id(0))]))) * (1.0f / (((d[(((2) - 1) * (8)) + (get_global_id(0))]) * (d[(((25) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((116) - 1) * (8)) + (get_global_id(0))]) = (b[(((116) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((3) - 1) * (8)) + (get_global_id(0))]) * (d[(((19) - 1) * (8)) + (get_global_id(0))]))) * (1.0f / (((d[(((10) - 1) * (8)) + (get_global_id(0))]) * (d[(((14) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((117) - 1) * (8)) + (get_global_id(0))]) = (b[(((117) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((5) - 1) * (8)) + (get_global_id(0))]) * (d[(((19) - 1) * (8)) + (get_global_id(0))]))) * (1.0f / (((d[(((2) - 1) * (8)) + (get_global_id(0))]) * (d[(((26) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((118) - 1) * (8)) + (get_global_id(0))]) = (b[(((118) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((5) - 1) * (8)) + (get_global_id(0))]) * (d[(((19) - 1) * (8)) + (get_global_id(0))]))) * (1.0f / (((d[(((12) - 1) * (8)) + (get_global_id(0))]) * (d[(((14) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((119) - 1) * (8)) + (get_global_id(0))]) = (b[(((119) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((16) - 1) * (8)) + (get_global_id(0))]) * (d[(((19) - 1) * (8)) + (get_global_id(0))]))) * (1.0f / (((d[(((14) - 1) * (8)) + (get_global_id(0))]) * (d[(((21) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((120) - 1) * (8)) + (get_global_id(0))]) = (b[(((120) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((12) - 1) * (8)) + (get_global_id(0))]) * (d[(((19) - 1) * (8)) + (get_global_id(0))]) * k)) * (1.0f / ((d[(((29) - 1) * (8)) + (get_global_id(0))]))));
  (c[(((121) - 1) * (8)) + (get_global_id(0))]) = (b[(((121) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = (((d[(((19) - 1) * (8)) + (get_global_id(0))])) * (1.0f / ((d[(((20) - 1) * (8)) + (get_global_id(0))]))));
  (c[(((122) - 1) * (8)) + (get_global_id(0))]) = (b[(((122) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((3) - 1) * (8)) + (get_global_id(0))]) * (d[(((20) - 1) * (8)) + (get_global_id(0))]))) * (1.0f / (((d[(((10) - 1) * (8)) + (get_global_id(0))]) * (d[(((14) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((123) - 1) * (8)) + (get_global_id(0))]) = (b[(((123) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((5) - 1) * (8)) + (get_global_id(0))]) * (d[(((20) - 1) * (8)) + (get_global_id(0))]))) * (1.0f / (((d[(((2) - 1) * (8)) + (get_global_id(0))]) * (d[(((26) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((124) - 1) * (8)) + (get_global_id(0))]) = (b[(((124) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((4) - 1) * (8)) + (get_global_id(0))]) * (d[(((20) - 1) * (8)) + (get_global_id(0))]))) * (1.0f / (((d[(((10) - 1) * (8)) + (get_global_id(0))]) * (d[(((15) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((125) - 1) * (8)) + (get_global_id(0))]) = (b[(((125) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);
}