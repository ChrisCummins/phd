__kernel void A(__global float4 *a, __global float4 *b, const int c) {
  int d = get_group_id(0);

  if (d * get_local_size(0) + get_local_id(0) < c / 4) {
    float4 e = a[d * get_local_size(0) + get_local_id(0)];
    b[d * get_local_size(0) + get_local_id(0)] = Ba(e);
  }
}