__kernel void A(__global float *a, __local float *b, __local float *c, __local float *d, int e, int f) {
  int g, h, i;
  int j;

  int k = get_group_id(0);
  int l = get_local_id(0);

  if (l < 64) {
    j = l;
    i = f * e + f;
    for (g = 0; g < 64 / 2; g++) {
      b[g * 64 + j] = a[i + j];
      i += e;
    }

    i = f * e + f;
    for (g = 0; g < 64; g++) {
      c[g * 64 + j] = a[i + (k + 1) * 64 + j];
      i += e;
    }

  } else {
    j = l - 64;

    i = (f + 64 / 2) * e + f;
    for (g = 64 / 2; g < 64; g++) {
      b[g * 64 + j] = a[i + j];
      i += e;
    }

    i = (f + (k + 1) * 64) * e + f;
    for (g = 0; g < 64; g++) {
      d[g * 64 + j] = a[i + j];
      i += e;
    }
  }
  barrier(1);

  if (l < 64) {
    j = l;
    for (g = 1; g < 64; g++) {
      for (h = 0; h < g; h++) c[g * 64 + j] -= b[g * 64 + h] * c[h * 64 + j];
    }
  } else {
    j = l - 64;
    for (g = 0; g < 64; g++) {
      for (h = 0; h < g; h++) d[j * 64 + g] -= d[j * 64 + h] * b[h * 64 + g];
      d[j * 64 + g] /= b[g * 64 + g];
    }
  }

  barrier(1);

  if (l < 64) {
    j = l;
    i = (f + 1) * e + f;
    for (g = 1; g < 64; g++) {
      a[i + (k + 1) * 64 + j] = c[g * 64 + j];
      i += e;
    }
  } else {
    j = l - 64;
    i = (f + (k + 1) * 64) * e + f;
    for (g = 0; g < 64; g++) {
      a[i + j] = d[g * 64 + j];
      i += e;
    }
  }
}