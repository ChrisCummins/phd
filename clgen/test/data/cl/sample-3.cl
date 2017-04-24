#include "./sample-3.h"

__kernel void A(__global MY_DATA_TYPE* a) {
    int b = get_global_id(0);
    a[b] += 1.0f;
}
