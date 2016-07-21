__kernel void A(__global float *a, __global float *b, int c, int d, int e, __local float *f) {
  unsigned int g = get_global_id(0);
  unsigned int h = get_global_id(1);

  unsigned int i = h * d + g + c;
  if ((g + c < d) && (h < e)) {
    f[get_local_id(1) * (16 + 1) + get_local_id(0)] = b[i];
  }

  barrier(1);

  if ((g < e) && (h + c < d)) {
    a[i] = f[get_local_id(1) * (16 + 1) + get_local_id(0)];
  }
}