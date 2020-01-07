__kernel void vadd(
    __global float* a,
    __global float* b,
    __global float* c,
    const unsigned int count) {
  int i = get_global_id(0);
  if(i < count)
    c[i] = a[i] + b[i];
}
