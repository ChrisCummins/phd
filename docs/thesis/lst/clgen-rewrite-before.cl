// saxpy.cl - Compute kernel for SAXPY
#define DTYPE float
#define ALPHA(a) 3.5f * a
inline DTYPE ax(DTYPE x) { return ALPHA(x); }

kernel void saxpy( /* SAXPY kernel */
    global DTYPE *input1,
    global DTYPE *input2,
    const int nelem)
{
     unsigned int idx = get_global_id(0);
     if (idx < nelem) { // = ax + y
         input2[idx] += ax(input1[idx]); } }