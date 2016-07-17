__kernel void A(__global const float* a, __global const float* b,
                __global const float* c, __global float* d, const float e,
                const float f) {
  const float g = b[get_global_id(0)] * e;
  const float h = a[get_global_id(0)] * f;

  const float i = 0x1.0p-126f;

  float j, k;

  j = 0.0;

  (d[(((1) - 1) * (8)) + (get_global_id(0))]) = k =
      (c[(((1) - 1) * (8)) + (get_global_id(0))]) * 4.96046521e-1f;
  j += k;
  (d[(((2) - 1) * (8)) + (get_global_id(0))]) = k =
      (c[(((2) - 1) * (8)) + (get_global_id(0))]) * 9.92093043e-1f;
  j += k;
  (d[(((3) - 1) * (8)) + (get_global_id(0))]) = k =
      (c[(((3) - 1) * (8)) + (get_global_id(0))]) * 6.25023433e-2f;
  j += k;
  (d[(((4) - 1) * (8)) + (get_global_id(0))]) = k =
      (c[(((4) - 1) * (8)) + (get_global_id(0))]) * 3.12511716e-2f;
  j += k;
  (d[(((5) - 1) * (8)) + (get_global_id(0))]) = k =
      (c[(((5) - 1) * (8)) + (get_global_id(0))]) * 5.87980383e-2f;
  j += k;
  (d[(((6) - 1) * (8)) + (get_global_id(0))]) = k =
      (c[(((6) - 1) * (8)) + (get_global_id(0))]) * 5.55082499e-2f;
  j += k;
  (d[(((7) - 1) * (8)) + (get_global_id(0))]) = k =
      (c[(((7) - 1) * (8)) + (get_global_id(0))]) * 3.02968146e-2f;
  j += k;
  (d[(((8) - 1) * (8)) + (get_global_id(0))]) = k =
      (c[(((8) - 1) * (8)) + (get_global_id(0))]) * 2.93990192e-2f;
  j += k;
  (d[(((9) - 1) * (8)) + (get_global_id(0))]) = k =
      (c[(((9) - 1) * (8)) + (get_global_id(0))]) * 6.65112065e-2f;
  j += k;
  (d[(((10) - 1) * (8)) + (get_global_id(0))]) = k =
      (c[(((10) - 1) * (8)) + (get_global_id(0))]) * 6.23323639e-2f;
  j += k;
  (d[(((11) - 1) * (8)) + (get_global_id(0))]) = k =
      (c[(((11) - 1) * (8)) + (get_global_id(0))]) * 3.57008335e-2f;
  j += k;
  (d[(((12) - 1) * (8)) + (get_global_id(0))]) = k =
      (c[(((12) - 1) * (8)) + (get_global_id(0))]) * 2.27221341e-2f;
  j += k;
  (d[(((13) - 1) * (8)) + (get_global_id(0))]) = k =
      (c[(((13) - 1) * (8)) + (get_global_id(0))]) * 3.33039255e-2f;
  j += k;
  (d[(((14) - 1) * (8)) + (get_global_id(0))]) = k =
      (c[(((14) - 1) * (8)) + (get_global_id(0))]) * 3.84050525e-2f;
  j += k;
  (d[(((15) - 1) * (8)) + (get_global_id(0))]) = k =
      (c[(((15) - 1) * (8)) + (get_global_id(0))]) * 3.56453112e-2f;
  j += k;
  (d[(((16) - 1) * (8)) + (get_global_id(0))]) = k =
      (c[(((16) - 1) * (8)) + (get_global_id(0))]) * 3.32556033e-2f;
  j += k;
  (d[(((17) - 1) * (8)) + (get_global_id(0))]) = k =
      (c[(((17) - 1) * (8)) + (get_global_id(0))]) * 2.4372606e-2f;
  j += k;
  (d[(((18) - 1) * (8)) + (get_global_id(0))]) = k =
      (c[(((18) - 1) * (8)) + (get_global_id(0))]) * 2.37882046e-2f;
  j += k;
  (d[(((19) - 1) * (8)) + (get_global_id(0))]) = k =
      (c[(((19) - 1) * (8)) + (get_global_id(0))]) * 2.26996304e-2f;
  j += k;
  (d[(((20) - 1) * (8)) + (get_global_id(0))]) = k =
      (c[(((20) - 1) * (8)) + (get_global_id(0))]) * 2.43467162e-2f;
  j += k;
  (d[(((21) - 1) * (8)) + (get_global_id(0))]) = k =
      (c[(((21) - 1) * (8)) + (get_global_id(0))]) * 2.37635408e-2f;
  j += k;
  (d[(((22) - 1) * (8)) + (get_global_id(0))]) = k =
      (c[(((22) - 1) * (8)) + (get_global_id(0))]) * 3.56972032e-2f;
  j += k;

  j = ((h) * (1.0f / ((j * (g)*8.314510e7f))));

  for (unsigned l = 1; l <= 22; l++) {
    (d[(((l)-1) * (8)) + (get_global_id(0))]) =
        fmax((d[(((l)-1) * (8)) + (get_global_id(0))]), i) * j;
  }
}