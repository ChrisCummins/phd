// Sample 1: Valid OpenCL
//
// Initialise array with values.
__kernel void foobar(__global int* a, const size_t nelem) {
    int id = get_global_id(0);
    if (id < nelem)
        a[id] = nelem;
}
