__kernel void A(__global const float* a, __global const float* b, __global float* c, __global const float* d, const float e) {
  float f = a[get_global_id(0)] * e;
  float g = log(f);

  const float h = 1e+20f;

  const float i = 8.31451e7f;
  const float j = 1.01325e6f;
  const float k = ((j) * (1.0f / ((i * (f)))));

  float l;

  l = ((((d[(((3) - 1) * (8)) + (get_global_id(0))]) * (d[(((8) - 1) * (8)) + (get_global_id(0))]))) * (1.0f / (((d[(((5) - 1) * (8)) + (get_global_id(0))]) * (d[(((7) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((26) - 1) * (8)) + (get_global_id(0))]) = (b[(((26) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((5) - 1) * (8)) + (get_global_id(0))]) * (d[(((8) - 1) * (8)) + (get_global_id(0))]))) * (1.0f / (((d[(((6) - 1) * (8)) + (get_global_id(0))]) * (d[(((7) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((27) - 1) * (8)) + (get_global_id(0))]) = (b[(((27) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((5) - 1) * (8)) + (get_global_id(0))]) * (d[(((8) - 1) * (8)) + (get_global_id(0))]))) * (1.0f / (((d[(((6) - 1) * (8)) + (get_global_id(0))]) * (d[(((7) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((28) - 1) * (8)) + (get_global_id(0))]) = (b[(((28) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((3) - 1) * (8)) + (get_global_id(0))]) * (d[(((14) - 1) * (8)) + (get_global_id(0))]) * k)) * (1.0f / ((d[(((15) - 1) * (8)) + (get_global_id(0))]))));
  (c[(((29) - 1) * (8)) + (get_global_id(0))]) = (b[(((29) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((5) - 1) * (8)) + (get_global_id(0))]) * (d[(((14) - 1) * (8)) + (get_global_id(0))]))) * (1.0f / (((d[(((2) - 1) * (8)) + (get_global_id(0))]) * (d[(((15) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((30) - 1) * (8)) + (get_global_id(0))]) = (b[(((30) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((1) - 1) * (8)) + (get_global_id(0))]) * (d[(((14) - 1) * (8)) + (get_global_id(0))]) * k)) * (1.0f / ((d[(((17) - 1) * (8)) + (get_global_id(0))]))));
  (c[(((31) - 1) * (8)) + (get_global_id(0))]) = (b[(((31) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((4) - 1) * (8)) + (get_global_id(0))]) * (d[(((14) - 1) * (8)) + (get_global_id(0))]))) * (1.0f / (((d[(((3) - 1) * (8)) + (get_global_id(0))]) * (d[(((15) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((32) - 1) * (8)) + (get_global_id(0))]) = (b[(((32) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((7) - 1) * (8)) + (get_global_id(0))]) * (d[(((14) - 1) * (8)) + (get_global_id(0))]))) * (1.0f / (((d[(((5) - 1) * (8)) + (get_global_id(0))]) * (d[(((15) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((33) - 1) * (8)) + (get_global_id(0))]) = (b[(((33) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((3) - 1) * (8)) + (get_global_id(0))]) * (d[(((9) - 1) * (8)) + (get_global_id(0))]))) * (1.0f / (((d[(((2) - 1) * (8)) + (get_global_id(0))]) * (d[(((14) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((34) - 1) * (8)) + (get_global_id(0))]) = (b[(((34) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((5) - 1) * (8)) + (get_global_id(0))]) * (d[(((9) - 1) * (8)) + (get_global_id(0))]))) * (1.0f / (((d[(((2) - 1) * (8)) + (get_global_id(0))]) * (d[(((16) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((35) - 1) * (8)) + (get_global_id(0))]) = (b[(((35) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((1) - 1) * (8)) + (get_global_id(0))]) * (d[(((9) - 1) * (8)) + (get_global_id(0))]))) * (1.0f / (((d[(((2) - 1) * (8)) + (get_global_id(0))]) * (d[(((10) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((36) - 1) * (8)) + (get_global_id(0))]) = (b[(((36) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((6) - 1) * (8)) + (get_global_id(0))]) * (d[(((9) - 1) * (8)) + (get_global_id(0))]))) * (1.0f / (((d[(((2) - 1) * (8)) + (get_global_id(0))]) * (d[(((17) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((37) - 1) * (8)) + (get_global_id(0))]) = (b[(((37) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((4) - 1) * (8)) + (get_global_id(0))]) * (d[(((9) - 1) * (8)) + (get_global_id(0))]))) * (1.0f / (((d[(((3) - 1) * (8)) + (get_global_id(0))]) * (d[(((16) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((38) - 1) * (8)) + (get_global_id(0))]) = (b[(((38) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((9) - 1) * (8)) + (get_global_id(0))]) * (d[(((14) - 1) * (8)) + (get_global_id(0))]) * k)) * (1.0f / ((d[(((25) - 1) * (8)) + (get_global_id(0))]))));
  (c[(((39) - 1) * (8)) + (get_global_id(0))]) = (b[(((39) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((9) - 1) * (8)) + (get_global_id(0))]) * (d[(((15) - 1) * (8)) + (get_global_id(0))]))) * (1.0f / (((d[(((14) - 1) * (8)) + (get_global_id(0))]) * (d[(((16) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((40) - 1) * (8)) + (get_global_id(0))]) = (b[(((40) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((2) - 1) * (8)) + (get_global_id(0))]) * (d[(((16) - 1) * (8)) + (get_global_id(0))]) * k)) * (1.0f / ((d[(((17) - 1) * (8)) + (get_global_id(0))]))));
  (c[(((41) - 1) * (8)) + (get_global_id(0))]) = (b[(((41) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((2) - 1) * (8)) + (get_global_id(0))]) * (d[(((16) - 1) * (8)) + (get_global_id(0))]))) * (1.0f / (((d[(((1) - 1) * (8)) + (get_global_id(0))]) * (d[(((14) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((42) - 1) * (8)) + (get_global_id(0))]) = (b[(((42) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((3) - 1) * (8)) + (get_global_id(0))]) * (d[(((16) - 1) * (8)) + (get_global_id(0))]))) * (1.0f / (((d[(((5) - 1) * (8)) + (get_global_id(0))]) * (d[(((14) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((43) - 1) * (8)) + (get_global_id(0))]) = (b[(((43) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((3) - 1) * (8)) + (get_global_id(0))]) * (d[(((16) - 1) * (8)) + (get_global_id(0))]))) * (1.0f / (((d[(((2) - 1) * (8)) + (get_global_id(0))]) * (d[(((15) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((44) - 1) * (8)) + (get_global_id(0))]) = (b[(((44) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((5) - 1) * (8)) + (get_global_id(0))]) * (d[(((16) - 1) * (8)) + (get_global_id(0))]))) * (1.0f / (((d[(((6) - 1) * (8)) + (get_global_id(0))]) * (d[(((14) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((45) - 1) * (8)) + (get_global_id(0))]) = (b[(((45) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((16) - 1) * (8)) + (get_global_id(0))]) * k)) * (1.0f / (((d[(((2) - 1) * (8)) + (get_global_id(0))]) * (d[(((14) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((46) - 1) * (8)) + (get_global_id(0))]) = (b[(((46) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((4) - 1) * (8)) + (get_global_id(0))]) * (d[(((16) - 1) * (8)) + (get_global_id(0))]))) * (1.0f / (((d[(((7) - 1) * (8)) + (get_global_id(0))]) * (d[(((14) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((47) - 1) * (8)) + (get_global_id(0))]) = (b[(((47) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((2) - 1) * (8)) + (get_global_id(0))]) * (d[(((10) - 1) * (8)) + (get_global_id(0))]) * k)) * (1.0f / ((d[(((12) - 1) * (8)) + (get_global_id(0))]))));
  (c[(((48) - 1) * (8)) + (get_global_id(0))]) = (b[(((48) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((1) - 1) * (8)) + (get_global_id(0))]) * (d[(((10) - 1) * (8)) + (get_global_id(0))]))) * (1.0f / (((d[(((2) - 1) * (8)) + (get_global_id(0))]) * (d[(((12) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((49) - 1) * (8)) + (get_global_id(0))]) = (b[(((49) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((3) - 1) * (8)) + (get_global_id(0))]) * (d[(((10) - 1) * (8)) + (get_global_id(0))]))) * (1.0f / (((d[(((2) - 1) * (8)) + (get_global_id(0))]) * (d[(((16) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((50) - 1) * (8)) + (get_global_id(0))]) = (b[(((50) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);
}