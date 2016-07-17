__kernel void A(__global const float* a, __global const float* b,
                __global float* c, __global const float* d, const float e) {
  const float f = a[get_global_id(0)] * e;
  const float g = log(f);

  const float h = 1e+20f;

  const float i = 8.31451e7f;
  const float j = 1.01325e6f;
  const float k = ((j) * (1.0f / ((i * (f)))));

  float l;

  l = ((((d[(((2) - 1) * (8)) + (get_global_id(0))]) *
         (d[(((4) - 1) * (8)) + (get_global_id(0))]))) *
       (1.0f / (((d[(((3) - 1) * (8)) + (get_global_id(0))]) *
                 (d[(((5) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((1) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((1) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((1) - 1) * (8)) + (get_global_id(0))]) *
         (d[(((3) - 1) * (8)) + (get_global_id(0))]))) *
       (1.0f / (((d[(((2) - 1) * (8)) + (get_global_id(0))]) *
                 (d[(((5) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((2) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((2) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((1) - 1) * (8)) + (get_global_id(0))]) *
         (d[(((5) - 1) * (8)) + (get_global_id(0))]))) *
       (1.0f / (((d[(((2) - 1) * (8)) + (get_global_id(0))]) *
                 (d[(((6) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((3) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((3) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((5) - 1) * (8)) + (get_global_id(0))]) *
         (d[(((5) - 1) * (8)) + (get_global_id(0))]))) *
       (1.0f / (((d[(((3) - 1) * (8)) + (get_global_id(0))]) *
                 (d[(((6) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((4) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((4) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((2) - 1) * (8)) + (get_global_id(0))]) *
         (d[(((2) - 1) * (8)) + (get_global_id(0))]) * k)) *
       (1.0f / ((d[(((1) - 1) * (8)) + (get_global_id(0))]))));
  (c[(((5) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((5) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((2) - 1) * (8)) + (get_global_id(0))]) *
         (d[(((2) - 1) * (8)) + (get_global_id(0))]) * k)) *
       (1.0f / ((d[(((1) - 1) * (8)) + (get_global_id(0))]))));
  (c[(((6) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((6) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((2) - 1) * (8)) + (get_global_id(0))]) *
         (d[(((2) - 1) * (8)) + (get_global_id(0))]) * k)) *
       (1.0f / ((d[(((1) - 1) * (8)) + (get_global_id(0))]))));
  (c[(((7) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((7) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((2) - 1) * (8)) + (get_global_id(0))]) *
         (d[(((2) - 1) * (8)) + (get_global_id(0))]) * k)) *
       (1.0f / ((d[(((1) - 1) * (8)) + (get_global_id(0))]))));
  (c[(((8) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((8) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((2) - 1) * (8)) + (get_global_id(0))]) *
         (d[(((5) - 1) * (8)) + (get_global_id(0))]) * k)) *
       (1.0f / ((d[(((6) - 1) * (8)) + (get_global_id(0))]))));
  (c[(((9) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((9) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((2) - 1) * (8)) + (get_global_id(0))]) *
         (d[(((3) - 1) * (8)) + (get_global_id(0))]) * k)) *
       (1.0f / ((d[(((5) - 1) * (8)) + (get_global_id(0))]))));
  (c[(((10) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((10) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((3) - 1) * (8)) + (get_global_id(0))]) *
         (d[(((3) - 1) * (8)) + (get_global_id(0))]) * k)) *
       (1.0f / ((d[(((4) - 1) * (8)) + (get_global_id(0))]))));
  (c[(((11) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((11) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((2) - 1) * (8)) + (get_global_id(0))]) *
         (d[(((4) - 1) * (8)) + (get_global_id(0))]) * k)) *
       (1.0f / ((d[(((7) - 1) * (8)) + (get_global_id(0))]))));
  (c[(((12) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((12) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((2) - 1) * (8)) + (get_global_id(0))]) *
         (d[(((4) - 1) * (8)) + (get_global_id(0))]) * k)) *
       (1.0f / ((d[(((7) - 1) * (8)) + (get_global_id(0))]))));
  (c[(((13) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((13) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((2) - 1) * (8)) + (get_global_id(0))]) *
         (d[(((4) - 1) * (8)) + (get_global_id(0))]) * k)) *
       (1.0f / ((d[(((7) - 1) * (8)) + (get_global_id(0))]))));
  (c[(((14) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((14) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((2) - 1) * (8)) + (get_global_id(0))]) *
         (d[(((4) - 1) * (8)) + (get_global_id(0))]) * k)) *
       (1.0f / ((d[(((7) - 1) * (8)) + (get_global_id(0))]))));
  (c[(((15) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((15) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((5) - 1) * (8)) + (get_global_id(0))]) *
         (d[(((5) - 1) * (8)) + (get_global_id(0))]) * k)) *
       (1.0f / ((d[(((8) - 1) * (8)) + (get_global_id(0))]))));
  (c[(((16) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((16) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((2) - 1) * (8)) + (get_global_id(0))]) *
         (d[(((7) - 1) * (8)) + (get_global_id(0))]))) *
       (1.0f / (((d[(((3) - 1) * (8)) + (get_global_id(0))]) *
                 (d[(((6) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((17) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((17) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((2) - 1) * (8)) + (get_global_id(0))]) *
         (d[(((7) - 1) * (8)) + (get_global_id(0))]))) *
       (1.0f / (((d[(((1) - 1) * (8)) + (get_global_id(0))]) *
                 (d[(((4) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((18) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((18) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((2) - 1) * (8)) + (get_global_id(0))]) *
         (d[(((7) - 1) * (8)) + (get_global_id(0))]))) *
       (1.0f / (((d[(((5) - 1) * (8)) + (get_global_id(0))]) *
                 (d[(((5) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((19) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((19) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((3) - 1) * (8)) + (get_global_id(0))]) *
         (d[(((7) - 1) * (8)) + (get_global_id(0))]))) *
       (1.0f / (((d[(((4) - 1) * (8)) + (get_global_id(0))]) *
                 (d[(((5) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((20) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((20) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((5) - 1) * (8)) + (get_global_id(0))]) *
         (d[(((7) - 1) * (8)) + (get_global_id(0))]))) *
       (1.0f / (((d[(((4) - 1) * (8)) + (get_global_id(0))]) *
                 (d[(((6) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((21) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((21) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((7) - 1) * (8)) + (get_global_id(0))]) *
         (d[(((7) - 1) * (8)) + (get_global_id(0))]))) *
       (1.0f / (((d[(((4) - 1) * (8)) + (get_global_id(0))]) *
                 (d[(((8) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((22) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((22) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((7) - 1) * (8)) + (get_global_id(0))]) *
         (d[(((7) - 1) * (8)) + (get_global_id(0))]))) *
       (1.0f / (((d[(((4) - 1) * (8)) + (get_global_id(0))]) *
                 (d[(((8) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((23) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((23) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((2) - 1) * (8)) + (get_global_id(0))]) *
         (d[(((8) - 1) * (8)) + (get_global_id(0))]))) *
       (1.0f / (((d[(((1) - 1) * (8)) + (get_global_id(0))]) *
                 (d[(((7) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((24) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((24) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((2) - 1) * (8)) + (get_global_id(0))]) *
         (d[(((8) - 1) * (8)) + (get_global_id(0))]))) *
       (1.0f / (((d[(((5) - 1) * (8)) + (get_global_id(0))]) *
                 (d[(((6) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((25) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((25) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);
}