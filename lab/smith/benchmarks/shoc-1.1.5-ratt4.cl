__kernel void A(__global const float* a, __global const float* b,
                __global float* c, __global const float* d, const float e) {
  float f = a[get_global_id(0)] * e;
  float g = log(f);

  const float h = 1e+20f;

  const float i = 8.31451e7f;
  const float j = 1.01325e6f;
  const float k = ((j) * (1.0f / ((i * (f)))));

  float l;

  l = ((((d[(((4) - 1) * (8)) + (get_global_id(0))]) *
         (d[(((10) - 1) * (8)) + (get_global_id(0))]))) *
       (1.0f / (((d[(((5) - 1) * (8)) + (get_global_id(0))]) *
                 (d[(((16) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((51) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((51) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((4) - 1) * (8)) + (get_global_id(0))]) *
         (d[(((10) - 1) * (8)) + (get_global_id(0))]))) *
       (1.0f / (((d[(((2) - 1) * (8)) + (get_global_id(0))]) *
                 (d[(((2) - 1) * (8)) + (get_global_id(0))]) *
                 (d[(((15) - 1) * (8)) + (get_global_id(0))]) * k))));
  (c[(((52) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((52) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((5) - 1) * (8)) + (get_global_id(0))]) *
         (d[(((10) - 1) * (8)) + (get_global_id(0))]))) *
       (1.0f / (((d[(((2) - 1) * (8)) + (get_global_id(0))]) *
                 (d[(((17) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((53) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((53) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((5) - 1) * (8)) + (get_global_id(0))]) *
         (d[(((10) - 1) * (8)) + (get_global_id(0))]))) *
       (1.0f / (((d[(((6) - 1) * (8)) + (get_global_id(0))]) *
                 (d[(((9) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((54) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((54) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((7) - 1) * (8)) + (get_global_id(0))]) *
         (d[(((10) - 1) * (8)) + (get_global_id(0))]))) *
       (1.0f / (((d[(((5) - 1) * (8)) + (get_global_id(0))]) *
                 (d[(((17) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((55) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((55) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((10) - 1) * (8)) + (get_global_id(0))]) *
         (d[(((14) - 1) * (8)) + (get_global_id(0))]) * k)) *
       (1.0f / ((d[(((26) - 1) * (8)) + (get_global_id(0))]))));
  (c[(((56) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((56) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((9) - 1) * (8)) + (get_global_id(0))]) *
         (d[(((10) - 1) * (8)) + (get_global_id(0))]))) *
       (1.0f / (((d[(((2) - 1) * (8)) + (get_global_id(0))]) *
                 (d[(((19) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((57) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((57) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((10) - 1) * (8)) + (get_global_id(0))]) *
         (d[(((10) - 1) * (8)) + (get_global_id(0))]))) *
       (1.0f / (((d[(((1) - 1) * (8)) + (get_global_id(0))]) *
                 (d[(((19) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((58) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((58) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = (((d[(((11) - 1) * (8)) + (get_global_id(0))])) *
       (1.0f / ((d[(((10) - 1) * (8)) + (get_global_id(0))]))));
  (c[(((59) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((59) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((2) - 1) * (8)) + (get_global_id(0))]) *
         (d[(((11) - 1) * (8)) + (get_global_id(0))]))) *
       (1.0f / (((d[(((1) - 1) * (8)) + (get_global_id(0))]) *
                 (d[(((9) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((60) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((60) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((3) - 1) * (8)) + (get_global_id(0))]) *
         (d[(((11) - 1) * (8)) + (get_global_id(0))]))) *
       (1.0f / (((d[(((1) - 1) * (8)) + (get_global_id(0))]) *
                 (d[(((14) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((61) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((61) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((3) - 1) * (8)) + (get_global_id(0))]) *
         (d[(((11) - 1) * (8)) + (get_global_id(0))]))) *
       (1.0f / (((d[(((2) - 1) * (8)) + (get_global_id(0))]) *
                 (d[(((16) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((62) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((62) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((5) - 1) * (8)) + (get_global_id(0))]) *
         (d[(((11) - 1) * (8)) + (get_global_id(0))]))) *
       (1.0f / (((d[(((2) - 1) * (8)) + (get_global_id(0))]) *
                 (d[(((17) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((63) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((63) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((1) - 1) * (8)) + (get_global_id(0))]) *
         (d[(((11) - 1) * (8)) + (get_global_id(0))]))) *
       (1.0f / (((d[(((2) - 1) * (8)) + (get_global_id(0))]) *
                 (d[(((12) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((64) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((64) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((4) - 1) * (8)) + (get_global_id(0))]) *
         (d[(((11) - 1) * (8)) + (get_global_id(0))]))) *
       (1.0f / (((d[(((2) - 1) * (8)) + (get_global_id(0))]) *
                 (d[(((5) - 1) * (8)) + (get_global_id(0))]) *
                 (d[(((14) - 1) * (8)) + (get_global_id(0))]) * k))));
  (c[(((65) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((65) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((4) - 1) * (8)) + (get_global_id(0))]) *
         (d[(((11) - 1) * (8)) + (get_global_id(0))]))) *
       (1.0f / (((d[(((6) - 1) * (8)) + (get_global_id(0))]) *
                 (d[(((14) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((66) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((66) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = (((d[(((11) - 1) * (8)) + (get_global_id(0))])) *
       (1.0f / ((d[(((10) - 1) * (8)) + (get_global_id(0))]))));
  (c[(((67) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((67) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = (((d[(((11) - 1) * (8)) + (get_global_id(0))])) *
       (1.0f / ((d[(((10) - 1) * (8)) + (get_global_id(0))]))));
  (c[(((68) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((68) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = (((d[(((11) - 1) * (8)) + (get_global_id(0))])) *
       (1.0f / ((d[(((10) - 1) * (8)) + (get_global_id(0))]))));
  (c[(((69) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((69) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((11) - 1) * (8)) + (get_global_id(0))]) *
         (d[(((15) - 1) * (8)) + (get_global_id(0))]))) *
       (1.0f / (((d[(((14) - 1) * (8)) + (get_global_id(0))]) *
                 (d[(((17) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((70) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((70) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((2) - 1) * (8)) + (get_global_id(0))]) *
         (d[(((17) - 1) * (8)) + (get_global_id(0))]) * k)) *
       (1.0f / ((d[(((18) - 1) * (8)) + (get_global_id(0))]))));
  (c[(((71) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((71) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((2) - 1) * (8)) + (get_global_id(0))]) *
         (d[(((17) - 1) * (8)) + (get_global_id(0))]))) *
       (1.0f / (((d[(((1) - 1) * (8)) + (get_global_id(0))]) *
                 (d[(((16) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((72) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((72) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((3) - 1) * (8)) + (get_global_id(0))]) *
         (d[(((17) - 1) * (8)) + (get_global_id(0))]))) *
       (1.0f / (((d[(((5) - 1) * (8)) + (get_global_id(0))]) *
                 (d[(((16) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((73) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((73) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((5) - 1) * (8)) + (get_global_id(0))]) *
         (d[(((17) - 1) * (8)) + (get_global_id(0))]))) *
       (1.0f / (((d[(((6) - 1) * (8)) + (get_global_id(0))]) *
                 (d[(((16) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((74) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((74) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((4) - 1) * (8)) + (get_global_id(0))]) *
         (d[(((17) - 1) * (8)) + (get_global_id(0))]))) *
       (1.0f / (((d[(((7) - 1) * (8)) + (get_global_id(0))]) *
                 (d[(((16) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((75) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((75) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);
}