#include "convert_dataset.h"

int main() {
    float* data;
    int* data_row_ptr, *nz_count, *data_col_index;
    int *rows, cols, dim, nz_count_len, len;
    
    coo_to_jds(
        "fidapm05.mtx", // bcsstk32.mtx, fidapm05.mtx, jgl009.mtx
        4, // row padding
        4, // warp size
        2, // pack size
        1, // is mirrored?
        0, // binary matrix
        3, // debug level [0:2]
        &data, &data_row_ptr, &nz_count, &data_col_index,
        &rows, &cols, &dim, &len, &nz_count_len
    );
}