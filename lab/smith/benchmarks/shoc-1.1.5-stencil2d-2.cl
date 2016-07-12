__kernel void F(__global float* y, __global float* z, const int aa, float ab,
                float ac, float ad, __local float* ae) {
  int a = get_group_id(0);
  int d = get_group_id(1);
  int af = get_num_groups(0);
  int ag = get_num_groups(1);
  int c = get_local_id(0);
  int f = get_local_id(1);
  int b = 16;
  int e = get_local_size(1);

  int ah = A(a, b, c);
  int ai = B(d, e, f);

  int aj = ag * e + 2;
  int ak = aj + (((aj % aa) == 0) ? 0 : (aa - (aj % aa)));
  int al = ak - 2;

  int am = e;
  for (int an = 0; an < (b + 2); an++) {
    int ao = C(c - 1 + an, f, am);
    int ap = C(ah - 1 + an, ai, al);
    ae[ao] = y[ap];
  }

  if (f == 0) {
    for (int an = 0; an < (b + 2); an++) {
      int ao = C(c - 1 + an, f - 1, am);
      int ap = C(ah - 1 + an, ai - 1, al);
      ae[ao] = y[ap];
    }
  } else if (f == (e - 1)) {
    for (int an = 0; an < (b + 2); an++) {
      int ao = C(c - 1 + an, f + 1, am);
      int ap = C(ah - 1 + an, ai + 1, al);
      ae[ao] = y[ap];
    }
  }

  barrier(1);

  for (int an = 0; an < b; an++) {
    int aq = C(c + an, f, am);
    int ar = C(c - 1 + an, f, am);
    int as = C(c + 1 + an, f, am);
    int at = C(c + an, f + 1, am);
    int au = C(c + an, f - 1, am);
    int av = C(c - 1 + an, f + 1, am);
    int aw = C(c + 1 + an, f + 1, am);
    int ax = C(c - 1 + an, f - 1, am);
    int ay = C(c + 1 + an, f - 1, am);

    float az = ae[aq];
    float ba = ae[ar] + ae[as] + ae[at] + ae[au];
    float bb = ae[av] + ae[aw] + ae[ax] + ae[ay];

    z[C(ah + an, ai, al)] = ab * az + ac * ba + ad * bb;
  }
}
