__kernel void A(__global float *a, __global float *b, unsigned int c,
                __local float *d) {
  unsigned int e = get_local_id(0);
  unsigned int f = get_global_id(0);

  d[e] = (f < c) ? a[f] : 0;

  barrier(1);

  for (unsigned int g = 1; g < get_local_size(0); g *= 2) {
    if ((e % (2 * g)) == 0) {
      d[e] += d[e + g];
    }
    barrier(1);
  }

  if (e == 0) b[get_group_id(0)] = d[0];
}

__kernel void B(__global float *a, __global float *b, unsigned int c,
                __local float *d) {
  unsigned int e = get_local_id(0);
  unsigned int f = get_global_id(0);

  d[e] = (f < c) ? a[f] : 0;

  barrier(1);

  for (unsigned int g = 1; g < get_local_size(0); g *= 2) {
    int h = 2 * g * e;

    if (h < get_local_size(0)) {
      d[h] += d[h + g];
    }

    barrier(1);
  }

  if (e == 0) b[get_group_id(0)] = d[0];
}

__kernel void C(__global float *a, __global float *b, unsigned int c,
                __local float *d) {
  unsigned int e = get_local_id(0);
  unsigned int f = get_global_id(0);

  d[e] = (f < c) ? a[f] : 0;

  barrier(1);

  for (unsigned int g = get_local_size(0) / 2; g > 0; g >>= 1) {
    if (e < g) {
      d[e] += d[e + g];
    }
    barrier(1);
  }

  if (e == 0) b[get_group_id(0)] = d[0];
}

__kernel void D(__global float *a, __global float *b, unsigned int c,
                __local float *d) {
  unsigned int e = get_local_id(0);
  unsigned int f = get_group_id(0) * (get_local_size(0) * 2) + get_local_id(0);

  d[e] = (f < c) ? a[f] : 0;
  if (f + get_local_size(0) < c) d[e] += a[f + get_local_size(0)];

  barrier(1);

  for (unsigned int g = get_local_size(0) / 2; g > 0; g >>= 1) {
    if (e < g) {
      d[e] += d[e + g];
    }
    barrier(1);
  }

  if (e == 0) b[get_group_id(0)] = d[0];
}

__kernel void E(__global float *a, __global float *b, unsigned int c,
                __local volatile float *d) {
  unsigned int e = get_local_id(0);
  unsigned int f = get_group_id(0) * (get_local_size(0) * 2) + get_local_id(0);

  d[e] = (f < c) ? a[f] : 0;
  if (f + get_local_size(0) < c) d[e] += a[f + get_local_size(0)];

  barrier(1);

  for (unsigned int g = get_local_size(0) / 2; g > 32; g >>= 1) {
    if (e < g) {
      d[e] += d[e + g];
    }
    barrier(1);
  }

  if (e < 32) {
    if (128 >= 64) {
      d[e] += d[e + 32];
    }
    if (128 >= 32) {
      d[e] += d[e + 16];
    }
    if (128 >= 16) {
      d[e] += d[e + 8];
    }
    if (128 >= 8) {
      d[e] += d[e + 4];
    }
    if (128 >= 4) {
      d[e] += d[e + 2];
    }
    if (128 >= 2) {
      d[e] += d[e + 1];
    }
  }

  if (e == 0) b[get_group_id(0)] = d[0];
}

__kernel void F(__global float *a, __global float *b, unsigned int c,
                __local volatile float *d) {
  unsigned int e = get_local_id(0);
  unsigned int f = get_group_id(0) * (get_local_size(0) * 2) + get_local_id(0);

  d[e] = (f < c) ? a[f] : 0;
  if (f + 128 < c) d[e] += a[f + 128];

  barrier(1);

  if (128 >= 512) {
    if (e < 256) {
      d[e] += d[e + 256];
    }
    barrier(1);
  }
  if (128 >= 256) {
    if (e < 128) {
      d[e] += d[e + 128];
    }
    barrier(1);
  }
  if (128 >= 128) {
    if (e < 64) {
      d[e] += d[e + 64];
    }
    barrier(1);
  }

  if (e < 32) {
    if (128 >= 64) {
      d[e] += d[e + 32];
    }
    if (128 >= 32) {
      d[e] += d[e + 16];
    }
    if (128 >= 16) {
      d[e] += d[e + 8];
    }
    if (128 >= 8) {
      d[e] += d[e + 4];
    }
    if (128 >= 4) {
      d[e] += d[e + 2];
    }
    if (128 >= 2) {
      d[e] += d[e + 1];
    }
  }

  if (e == 0) b[get_group_id(0)] = d[0];
}

__kernel void G(__global float *a, __global float *b, unsigned int c,
                __local volatile float *d) {
  unsigned int e = get_local_id(0);
  unsigned int f = get_group_id(0) * (get_local_size(0) * 2) + get_local_id(0);
  unsigned int i = 128 * 2 * get_num_groups(0);
  d[e] = 0;

  while (f < c) {
    d[e] += a[f];

    if (1 || f + 128 < c) d[e] += a[f + 128];
    f += i;
  }

  barrier(1);

  if (128 >= 512) {
    if (e < 256) {
      d[e] += d[e + 256];
    }
    barrier(1);
  }
  if (128 >= 256) {
    if (e < 128) {
      d[e] += d[e + 128];
    }
    barrier(1);
  }
  if (128 >= 128) {
    if (e < 64) {
      d[e] += d[e + 64];
    }
    barrier(1);
  }

  if (e < 32) {
    if (128 >= 64) {
      d[e] += d[e + 32];
    }
    if (128 >= 32) {
      d[e] += d[e + 16];
    }
    if (128 >= 16) {
      d[e] += d[e + 8];
    }
    if (128 >= 8) {
      d[e] += d[e + 4];
    }
    if (128 >= 4) {
      d[e] += d[e + 2];
    }
    if (128 >= 2) {
      d[e] += d[e + 1];
    }
  }

  if (e == 0) b[get_group_id(0)] = d[0];
}