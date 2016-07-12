__kernel void A(__global const float* a, __global const float* b,
                __global float* c, __global const float* d, const float e) {
  const float f = a[get_global_id(0)] * e;
  const float g = log(f);

  const float h = 1e+20f;

  const float i = 8.31451e7f;
  const float j = 1.01325e6f;
  const float k = ((j) * (1.0f / ((i * (f)))));

  float l;

  l = ((((d[(((3) - 1) * (8)) + (get_global_id(0))]) *
         (d[(((27) - 1) * (8)) + (get_global_id(0))]))) *
       (1.0f / (((d[(((5) - 1) * (8)) + (get_global_id(0))]) *
                 (d[(((26) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((151) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((151) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((5) - 1) * (8)) + (get_global_id(0))]) *
         (d[(((27) - 1) * (8)) + (get_global_id(0))]))) *
       (1.0f / (((d[(((6) - 1) * (8)) + (get_global_id(0))]) *
                 (d[(((26) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((152) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((152) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((4) - 1) * (8)) + (get_global_id(0))]) *
         (d[(((27) - 1) * (8)) + (get_global_id(0))]))) *
       (1.0f / (((d[(((7) - 1) * (8)) + (get_global_id(0))]) *
                 (d[(((26) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((153) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((153) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((4) - 1) * (8)) + (get_global_id(0))]) *
         (d[(((27) - 1) * (8)) + (get_global_id(0))]))) *
       (1.0f / (((d[(((5) - 1) * (8)) + (get_global_id(0))]) *
                 (d[(((14) - 1) * (8)) + (get_global_id(0))]) *
                 (d[(((17) - 1) * (8)) + (get_global_id(0))]) * k))));
  (c[(((154) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((154) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = (((d[(((22) - 1) * (8)) + (get_global_id(0))])) *
       (1.0f / (((d[(((1) - 1) * (8)) + (get_global_id(0))]) *
                 (d[(((20) - 1) * (8)) + (get_global_id(0))]) * k))));
  (c[(((155) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((155) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((2) - 1) * (8)) + (get_global_id(0))]) *
         (d[(((22) - 1) * (8)) + (get_global_id(0))]) * k)) *
       (1.0f / ((d[(((23) - 1) * (8)) + (get_global_id(0))]))));
  (c[(((156) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((156) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((2) - 1) * (8)) + (get_global_id(0))]) *
         (d[(((22) - 1) * (8)) + (get_global_id(0))]))) *
       (1.0f / (((d[(((1) - 1) * (8)) + (get_global_id(0))]) *
                 (d[(((21) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((157) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((157) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((3) - 1) * (8)) + (get_global_id(0))]) *
         (d[(((22) - 1) * (8)) + (get_global_id(0))]))) *
       (1.0f / (((d[(((5) - 1) * (8)) + (get_global_id(0))]) *
                 (d[(((21) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((158) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((158) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((3) - 1) * (8)) + (get_global_id(0))]) *
         (d[(((22) - 1) * (8)) + (get_global_id(0))]))) *
       (1.0f / (((d[(((12) - 1) * (8)) + (get_global_id(0))]) *
                 (d[(((16) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((159) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((159) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((3) - 1) * (8)) + (get_global_id(0))]) *
         (d[(((22) - 1) * (8)) + (get_global_id(0))]))) *
       (1.0f / (((d[(((10) - 1) * (8)) + (get_global_id(0))]) *
                 (d[(((17) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((160) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((160) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((5) - 1) * (8)) + (get_global_id(0))]) *
         (d[(((22) - 1) * (8)) + (get_global_id(0))]))) *
       (1.0f / (((d[(((6) - 1) * (8)) + (get_global_id(0))]) *
                 (d[(((21) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((161) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((161) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((4) - 1) * (8)) + (get_global_id(0))]) *
         (d[(((22) - 1) * (8)) + (get_global_id(0))]))) *
       (1.0f / (((d[(((7) - 1) * (8)) + (get_global_id(0))]) *
                 (d[(((21) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((162) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((162) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((7) - 1) * (8)) + (get_global_id(0))]) *
         (d[(((22) - 1) * (8)) + (get_global_id(0))]))) *
       (1.0f / (((d[(((5) - 1) * (8)) + (get_global_id(0))]) *
                 (d[(((28) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((163) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((163) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((16) - 1) * (8)) + (get_global_id(0))]) *
         (d[(((22) - 1) * (8)) + (get_global_id(0))]))) *
       (1.0f / (((d[(((14) - 1) * (8)) + (get_global_id(0))]) *
                 (d[(((23) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((164) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((164) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((10) - 1) * (8)) + (get_global_id(0))]) *
         (d[(((22) - 1) * (8)) + (get_global_id(0))]))) *
       (1.0f / (((d[(((2) - 1) * (8)) + (get_global_id(0))]) *
                 (d[(((29) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((165) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((165) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((11) - 1) * (8)) + (get_global_id(0))]) *
         (d[(((22) - 1) * (8)) + (get_global_id(0))]))) *
       (1.0f / (((d[(((13) - 1) * (8)) + (get_global_id(0))]) *
                 (d[(((20) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((166) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((166) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((11) - 1) * (8)) + (get_global_id(0))]) *
         (d[(((22) - 1) * (8)) + (get_global_id(0))]))) *
       (1.0f / (((d[(((2) - 1) * (8)) + (get_global_id(0))]) *
                 (d[(((29) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((167) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((167) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((12) - 1) * (8)) + (get_global_id(0))]) *
         (d[(((22) - 1) * (8)) + (get_global_id(0))]))) *
       (1.0f / (((d[(((13) - 1) * (8)) + (get_global_id(0))]) *
                 (d[(((21) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((168) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((168) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((12) - 1) * (8)) + (get_global_id(0))]) *
         (d[(((22) - 1) * (8)) + (get_global_id(0))]) * k)) *
       (1.0f / ((d[(((31) - 1) * (8)) + (get_global_id(0))]))));
  (c[(((169) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((169) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((2) - 1) * (8)) + (get_global_id(0))]) *
         (d[(((23) - 1) * (8)) + (get_global_id(0))]) * k)) *
       (1.0f / ((d[(((24) - 1) * (8)) + (get_global_id(0))]))));
  (c[(((170) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((170) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((2) - 1) * (8)) + (get_global_id(0))]) *
         (d[(((23) - 1) * (8)) + (get_global_id(0))]))) *
       (1.0f / (((d[(((1) - 1) * (8)) + (get_global_id(0))]) *
                 (d[(((22) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((171) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((171) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((3) - 1) * (8)) + (get_global_id(0))]) *
         (d[(((23) - 1) * (8)) + (get_global_id(0))]))) *
       (1.0f / (((d[(((12) - 1) * (8)) + (get_global_id(0))]) *
                 (d[(((17) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((172) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((172) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((3) - 1) * (8)) + (get_global_id(0))]) *
         (d[(((23) - 1) * (8)) + (get_global_id(0))]))) *
       (1.0f / (((d[(((2) - 1) * (8)) + (get_global_id(0))]) *
                 (d[(((28) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((173) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((173) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((4) - 1) * (8)) + (get_global_id(0))]) *
         (d[(((23) - 1) * (8)) + (get_global_id(0))]))) *
       (1.0f / (((d[(((7) - 1) * (8)) + (get_global_id(0))]) *
                 (d[(((22) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((174) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((174) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((7) - 1) * (8)) + (get_global_id(0))]) *
         (d[(((23) - 1) * (8)) + (get_global_id(0))]))) *
       (1.0f / (((d[(((4) - 1) * (8)) + (get_global_id(0))]) *
                 (d[(((24) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((175) - 1) * (8)) + (get_global_id(0))]) =
      (b[(((175) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);
}