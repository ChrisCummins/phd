__kernel void A(__global int* a) {
    int id = get_global_id(0);
    while (1)
        a[id] += 1;
}
