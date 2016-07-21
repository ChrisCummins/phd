__kernel void A(__global const float* a, __global const float* b, __global float* c, __global const float* d, const float e) {
  const float f = a[get_global_id(0)] * e;
  const float g = log(f);

  const float h = 1e+20f;

  const float i = 8.31451e7f;
  const float j = 1.01325e6f;
  const float k = ((j) * (1.0f / ((i * (f)))));

  float l;

  l = ((((d[(((7) - 1) * (8)) + (get_global_id(0))]) * (d[(((23) - 1) * (8)) + (get_global_id(0))]))) * (1.0f / (((d[(((8) - 1) * (8)) + (get_global_id(0))]) * (d[(((22) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((176) - 1) * (8)) + (get_global_id(0))]) = (b[(((176) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((7) - 1) * (8)) + (get_global_id(0))]) * (d[(((23) - 1) * (8)) + (get_global_id(0))]))) * (1.0f / (((d[(((5) - 1) * (8)) + (get_global_id(0))]) * (d[(((12) - 1) * (8)) + (get_global_id(0))]) * (d[(((17) - 1) * (8)) + (get_global_id(0))]) * k))));
  (c[(((177) - 1) * (8)) + (get_global_id(0))]) = (b[(((177) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((8) - 1) * (8)) + (get_global_id(0))]) * (d[(((23) - 1) * (8)) + (get_global_id(0))]))) * (1.0f / (((d[(((7) - 1) * (8)) + (get_global_id(0))]) * (d[(((24) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((178) - 1) * (8)) + (get_global_id(0))]) = (b[(((178) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((16) - 1) * (8)) + (get_global_id(0))]) * (d[(((23) - 1) * (8)) + (get_global_id(0))]))) * (1.0f / (((d[(((14) - 1) * (8)) + (get_global_id(0))]) * (d[(((24) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((179) - 1) * (8)) + (get_global_id(0))]) = (b[(((179) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((2) - 1) * (8)) + (get_global_id(0))]) * (d[(((24) - 1) * (8)) + (get_global_id(0))]))) * (1.0f / (((d[(((1) - 1) * (8)) + (get_global_id(0))]) * (d[(((23) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((180) - 1) * (8)) + (get_global_id(0))]) = (b[(((180) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((3) - 1) * (8)) + (get_global_id(0))]) * (d[(((24) - 1) * (8)) + (get_global_id(0))]))) * (1.0f / (((d[(((5) - 1) * (8)) + (get_global_id(0))]) * (d[(((23) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((181) - 1) * (8)) + (get_global_id(0))]) = (b[(((181) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((5) - 1) * (8)) + (get_global_id(0))]) * (d[(((24) - 1) * (8)) + (get_global_id(0))]))) * (1.0f / (((d[(((6) - 1) * (8)) + (get_global_id(0))]) * (d[(((23) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((182) - 1) * (8)) + (get_global_id(0))]) = (b[(((182) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((11) - 1) * (8)) + (get_global_id(0))]) * (d[(((24) - 1) * (8)) + (get_global_id(0))]))) * (1.0f / (((d[(((12) - 1) * (8)) + (get_global_id(0))]) * (d[(((23) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((183) - 1) * (8)) + (get_global_id(0))]) = (b[(((183) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((12) - 1) * (8)) + (get_global_id(0))]) * (d[(((24) - 1) * (8)) + (get_global_id(0))]))) * (1.0f / (((d[(((13) - 1) * (8)) + (get_global_id(0))]) * (d[(((23) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((184) - 1) * (8)) + (get_global_id(0))]) = (b[(((184) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((2) - 1) * (8)) + (get_global_id(0))]) * (d[(((29) - 1) * (8)) + (get_global_id(0))]) * k)) * (1.0f / ((d[(((30) - 1) * (8)) + (get_global_id(0))]))));
  (c[(((185) - 1) * (8)) + (get_global_id(0))]) = (b[(((185) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((2) - 1) * (8)) + (get_global_id(0))]) * (d[(((29) - 1) * (8)) + (get_global_id(0))]))) * (1.0f / (((d[(((13) - 1) * (8)) + (get_global_id(0))]) * (d[(((20) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((186) - 1) * (8)) + (get_global_id(0))]) = (b[(((186) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((7) - 1) * (8)) + (get_global_id(0))]) * (d[(((29) - 1) * (8)) + (get_global_id(0))]))) * (1.0f / (((d[(((4) - 1) * (8)) + (get_global_id(0))]) * (d[(((30) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((187) - 1) * (8)) + (get_global_id(0))]) = (b[(((187) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((7) - 1) * (8)) + (get_global_id(0))]) * (d[(((29) - 1) * (8)) + (get_global_id(0))]))) * (1.0f / (((d[(((5) - 1) * (8)) + (get_global_id(0))]) * (d[(((17) - 1) * (8)) + (get_global_id(0))]) * (d[(((21) - 1) * (8)) + (get_global_id(0))]) * k))));
  (c[(((188) - 1) * (8)) + (get_global_id(0))]) = (b[(((188) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((16) - 1) * (8)) + (get_global_id(0))]) * (d[(((29) - 1) * (8)) + (get_global_id(0))]))) * (1.0f / (((d[(((14) - 1) * (8)) + (get_global_id(0))]) * (d[(((30) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((189) - 1) * (8)) + (get_global_id(0))]) = (b[(((189) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((2) - 1) * (8)) + (get_global_id(0))]) * (d[(((30) - 1) * (8)) + (get_global_id(0))]) * k)) * (1.0f / ((d[(((31) - 1) * (8)) + (get_global_id(0))]))));
  (c[(((190) - 1) * (8)) + (get_global_id(0))]) = (b[(((190) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((2) - 1) * (8)) + (get_global_id(0))]) * (d[(((30) - 1) * (8)) + (get_global_id(0))]))) * (1.0f / (((d[(((12) - 1) * (8)) + (get_global_id(0))]) * (d[(((22) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((191) - 1) * (8)) + (get_global_id(0))]) = (b[(((191) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((2) - 1) * (8)) + (get_global_id(0))]) * (d[(((30) - 1) * (8)) + (get_global_id(0))]))) * (1.0f / (((d[(((1) - 1) * (8)) + (get_global_id(0))]) * (d[(((29) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((192) - 1) * (8)) + (get_global_id(0))]) = (b[(((192) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((3) - 1) * (8)) + (get_global_id(0))]) * (d[(((30) - 1) * (8)) + (get_global_id(0))]))) * (1.0f / (((d[(((2) - 1) * (8)) + (get_global_id(0))]) * (d[(((12) - 1) * (8)) + (get_global_id(0))]) * (d[(((26) - 1) * (8)) + (get_global_id(0))]) * k))));
  (c[(((193) - 1) * (8)) + (get_global_id(0))]) = (b[(((193) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((3) - 1) * (8)) + (get_global_id(0))]) * (d[(((30) - 1) * (8)) + (get_global_id(0))]))) * (1.0f / (((d[(((16) - 1) * (8)) + (get_global_id(0))]) * (d[(((23) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((194) - 1) * (8)) + (get_global_id(0))]) = (b[(((194) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((3) - 1) * (8)) + (get_global_id(0))]) * (d[(((30) - 1) * (8)) + (get_global_id(0))]))) * (1.0f / (((d[(((5) - 1) * (8)) + (get_global_id(0))]) * (d[(((29) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((195) - 1) * (8)) + (get_global_id(0))]) = (b[(((195) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((5) - 1) * (8)) + (get_global_id(0))]) * (d[(((30) - 1) * (8)) + (get_global_id(0))]))) * (1.0f / (((d[(((6) - 1) * (8)) + (get_global_id(0))]) * (d[(((29) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((196) - 1) * (8)) + (get_global_id(0))]) = (b[(((196) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((7) - 1) * (8)) + (get_global_id(0))]) * (d[(((30) - 1) * (8)) + (get_global_id(0))]))) * (1.0f / (((d[(((8) - 1) * (8)) + (get_global_id(0))]) * (d[(((29) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((197) - 1) * (8)) + (get_global_id(0))]) = (b[(((197) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((12) - 1) * (8)) + (get_global_id(0))]) * (d[(((30) - 1) * (8)) + (get_global_id(0))]))) * (1.0f / (((d[(((13) - 1) * (8)) + (get_global_id(0))]) * (d[(((29) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((198) - 1) * (8)) + (get_global_id(0))]) = (b[(((198) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((2) - 1) * (8)) + (get_global_id(0))]) * (d[(((31) - 1) * (8)) + (get_global_id(0))]))) * (1.0f / (((d[(((12) - 1) * (8)) + (get_global_id(0))]) * (d[(((23) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((199) - 1) * (8)) + (get_global_id(0))]) = (b[(((199) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((2) - 1) * (8)) + (get_global_id(0))]) * (d[(((31) - 1) * (8)) + (get_global_id(0))]))) * (1.0f / (((d[(((1) - 1) * (8)) + (get_global_id(0))]) * (d[(((30) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((200) - 1) * (8)) + (get_global_id(0))]) = (b[(((200) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((3) - 1) * (8)) + (get_global_id(0))]) * (d[(((31) - 1) * (8)) + (get_global_id(0))]))) * (1.0f / (((d[(((17) - 1) * (8)) + (get_global_id(0))]) * (d[(((23) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((201) - 1) * (8)) + (get_global_id(0))]) = (b[(((201) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((5) - 1) * (8)) + (get_global_id(0))]) * (d[(((31) - 1) * (8)) + (get_global_id(0))]))) * (1.0f / (((d[(((6) - 1) * (8)) + (get_global_id(0))]) * (d[(((30) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((202) - 1) * (8)) + (get_global_id(0))]) = (b[(((202) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((4) - 1) * (8)) + (get_global_id(0))]) * (d[(((31) - 1) * (8)) + (get_global_id(0))]))) * (1.0f / (((d[(((7) - 1) * (8)) + (get_global_id(0))]) * (d[(((30) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((203) - 1) * (8)) + (get_global_id(0))]) = (b[(((203) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((7) - 1) * (8)) + (get_global_id(0))]) * (d[(((31) - 1) * (8)) + (get_global_id(0))]))) * (1.0f / (((d[(((5) - 1) * (8)) + (get_global_id(0))]) * (d[(((17) - 1) * (8)) + (get_global_id(0))]) * (d[(((23) - 1) * (8)) + (get_global_id(0))]) * k))));
  (c[(((204) - 1) * (8)) + (get_global_id(0))]) = (b[(((204) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((12) - 1) * (8)) + (get_global_id(0))]) * (d[(((31) - 1) * (8)) + (get_global_id(0))]))) * (1.0f / (((d[(((13) - 1) * (8)) + (get_global_id(0))]) * (d[(((30) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((205) - 1) * (8)) + (get_global_id(0))]) = (b[(((205) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);

  l = ((((d[(((21) - 1) * (8)) + (get_global_id(0))]) * (d[(((23) - 1) * (8)) + (get_global_id(0))]))) * (1.0f / (((d[(((12) - 1) * (8)) + (get_global_id(0))]) * (d[(((29) - 1) * (8)) + (get_global_id(0))])))));
  (c[(((206) - 1) * (8)) + (get_global_id(0))]) = (b[(((206) - 1) * (8)) + (get_global_id(0))]) * fmin(l, h);
}