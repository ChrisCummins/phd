__kernel void A(__global const float* a, __global const float* b,
                __global float* c, __global const float* d, const float e) {
  const float f = a[get_global_id(0)] * e;
  const float g = log(f);

  const float h = 1e+20f;

  const float i = 8.31451e7f;
  const float j = 1.01325e6f;
  const float k = ((j) * (1.0f / ((i * (f)))));

  float l;

  l = ((((d[(((7) - 1) * (8)) + (get_global_id(0))]) *
         (d[(((17) - 1) * (8)) + (get_global_id(0))]))) *
       (1.0f / (((d[(((8) - 1) * (8)) + (get_global_id(0))]) *
                 (d[(((16) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((76) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((76) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((9) - 1) * (8)) + (get_global_id(0))]) *
         (d[(((17) - 1) * (8)) + (get_global_id(0))]))) *
       (1.0f / (((d[(((2) - 1) * (8)) + (get_global_id(0))]) *
                 (d[(((26) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((77) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((77) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((2) - 1) * (8)) + (get_global_id(0))]) *
         (d[(((12) - 1) * (8)) + (get_global_id(0))]) * k)) *
       (1.0f / ((d[(((13) - 1) * (8)) + (get_global_id(0))]))));
  (c[(((78) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((78) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((3) - 1) * (8)) + (get_global_id(0))]) *
         (d[(((12) - 1) * (8)) + (get_global_id(0))]))) *
       (1.0f / (((d[(((2) - 1) * (8)) + (get_global_id(0))]) *
                 (d[(((17) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((79) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((79) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((5) - 1) * (8)) + (get_global_id(0))]) *
         (d[(((12) - 1) * (8)) + (get_global_id(0))]))) *
       (1.0f / (((d[(((6) - 1) * (8)) + (get_global_id(0))]) *
                 (d[(((10) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((80) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((80) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((5) - 1) * (8)) + (get_global_id(0))]) *
         (d[(((12) - 1) * (8)) + (get_global_id(0))]))) *
       (1.0f / (((d[(((6) - 1) * (8)) + (get_global_id(0))]) *
                 (d[(((11) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((81) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((81) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((4) - 1) * (8)) + (get_global_id(0))]) *
         (d[(((12) - 1) * (8)) + (get_global_id(0))]))) *
       (1.0f / (((d[(((3) - 1) * (8)) + (get_global_id(0))]) *
                 (d[(((18) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((82) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((82) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((4) - 1) * (8)) + (get_global_id(0))]) *
         (d[(((12) - 1) * (8)) + (get_global_id(0))]))) *
       (1.0f / (((d[(((5) - 1) * (8)) + (get_global_id(0))]) *
                 (d[(((17) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((83) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((83) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((7) - 1) * (8)) + (get_global_id(0))]) *
         (d[(((12) - 1) * (8)) + (get_global_id(0))]))) *
       (1.0f / (((d[(((4) - 1) * (8)) + (get_global_id(0))]) *
                 (d[(((13) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((84) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((84) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((7) - 1) * (8)) + (get_global_id(0))]) *
         (d[(((12) - 1) * (8)) + (get_global_id(0))]))) *
       (1.0f / (((d[(((5) - 1) * (8)) + (get_global_id(0))]) *
                 (d[(((18) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((85) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((85) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((8) - 1) * (8)) + (get_global_id(0))]) *
         (d[(((12) - 1) * (8)) + (get_global_id(0))]))) *
       (1.0f / (((d[(((7) - 1) * (8)) + (get_global_id(0))]) *
                 (d[(((13) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((86) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((86) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((9) - 1) * (8)) + (get_global_id(0))]) *
         (d[(((12) - 1) * (8)) + (get_global_id(0))]))) *
       (1.0f / (((d[(((2) - 1) * (8)) + (get_global_id(0))]) *
                 (d[(((21) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((87) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((87) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((12) - 1) * (8)) + (get_global_id(0))]) *
         (d[(((16) - 1) * (8)) + (get_global_id(0))]))) *
       (1.0f / (((d[(((13) - 1) * (8)) + (get_global_id(0))]) *
                 (d[(((14) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((88) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((88) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((12) - 1) * (8)) + (get_global_id(0))]) *
         (d[(((16) - 1) * (8)) + (get_global_id(0))]) * k)) *
       (1.0f / ((d[(((28) - 1) * (8)) + (get_global_id(0))]))));
  (c[(((89) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((89) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((12) - 1) * (8)) + (get_global_id(0))]) *
         (d[(((17) - 1) * (8)) + (get_global_id(0))]))) *
       (1.0f / (((d[(((13) - 1) * (8)) + (get_global_id(0))]) *
                 (d[(((16) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((90) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((90) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((10) - 1) * (8)) + (get_global_id(0))]) *
         (d[(((12) - 1) * (8)) + (get_global_id(0))]))) *
       (1.0f / (((d[(((2) - 1) * (8)) + (get_global_id(0))]) *
                 (d[(((22) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((91) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((91) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((11) - 1) * (8)) + (get_global_id(0))]) *
         (d[(((12) - 1) * (8)) + (get_global_id(0))]))) *
       (1.0f / (((d[(((2) - 1) * (8)) + (get_global_id(0))]) *
                 (d[(((22) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((92) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((92) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((12) - 1) * (8)) + (get_global_id(0))]) *
         (d[(((12) - 1) * (8)) + (get_global_id(0))]) * k)) *
       (1.0f / ((d[(((24) - 1) * (8)) + (get_global_id(0))]))));
  (c[(((93) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((93) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((12) - 1) * (8)) + (get_global_id(0))]) *
         (d[(((12) - 1) * (8)) + (get_global_id(0))]))) *
       (1.0f / (((d[(((2) - 1) * (8)) + (get_global_id(0))]) *
                 (d[(((23) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((94) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((94) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((12) - 1) * (8)) + (get_global_id(0))]) *
         (d[(((25) - 1) * (8)) + (get_global_id(0))]))) *
       (1.0f / (((d[(((14) - 1) * (8)) + (get_global_id(0))]) *
                 (d[(((22) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((95) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((95) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((2) - 1) * (8)) + (get_global_id(0))]) *
         (d[(((18) - 1) * (8)) + (get_global_id(0))]))) *
       (1.0f / (((d[(((1) - 1) * (8)) + (get_global_id(0))]) *
                 (d[(((17) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((96) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((96) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((2) - 1) * (8)) + (get_global_id(0))]) *
         (d[(((18) - 1) * (8)) + (get_global_id(0))]))) *
       (1.0f / (((d[(((5) - 1) * (8)) + (get_global_id(0))]) *
                 (d[(((12) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((97) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((97) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((2) - 1) * (8)) + (get_global_id(0))]) *
         (d[(((18) - 1) * (8)) + (get_global_id(0))]))) *
       (1.0f / (((d[(((6) - 1) * (8)) + (get_global_id(0))]) *
                 (d[(((11) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((98) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((98) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((3) - 1) * (8)) + (get_global_id(0))]) *
         (d[(((18) - 1) * (8)) + (get_global_id(0))]))) *
       (1.0f / (((d[(((5) - 1) * (8)) + (get_global_id(0))]) *
                 (d[(((17) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((99) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((99) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((5) - 1) * (8)) + (get_global_id(0))]) *
         (d[(((18) - 1) * (8)) + (get_global_id(0))]))) *
       (1.0f / (((d[(((6) - 1) * (8)) + (get_global_id(0))]) *
                 (d[(((17) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((100) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((100) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);
}
