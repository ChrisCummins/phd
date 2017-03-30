__kernel void fn_A(__global int* A, const int B) {
  int C = get_global_id(0);
  if (C < B)
    A[C] = B;
}