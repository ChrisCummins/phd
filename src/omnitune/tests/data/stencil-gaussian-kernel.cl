#ifndef float_MATRIX_T
typedef struct {
  __global float* data;
  unsigned int col_count;
} float_matrix_t;
#define float_MATRIX_T
#endif

#ifndef MATRIX_GET
#define get(m, y, x) m.data[(int)((y) * m.col_count + (x))]
#define MATRIX_GET
#endif
#ifndef MATRIX_SET
#define set(m, y, x, v) m.data[(int)((y) * m.col_count + (x))] = (v)
#define MATRIX_SET
#endif
#ifndef float_MATRIX_T
typedef struct {
  __global float* data;
  unsigned int col_count;
} float_matrix_t;
#define float_MATRIX_T
#endif

#ifndef MATRIX_GET
#define get(m, y, x) m.data[(int)((y) * m.col_count + (x))]
#define MATRIX_GET
#endif
#ifndef MATRIX_SET
#define set(m, y, x, v) m.data[(int)((y) * m.col_count + (x))] = (v)
#define MATRIX_SET
#endif
#define SCL_NORTH (5)
#define SCL_WEST  (5)
#define SCL_SOUTH (5)
#define SCL_EAST  (5)
#define SCL_TILE_WIDTH  (get_local_size(0) + SCL_WEST + SCL_EAST)
#define SCL_TILE_HEIGHT (get_local_size(1) + SCL_NORTH + SCL_SOUTH)
#define SCL_COL   (get_global_id(0))
#define SCL_ROW   (get_global_id(1))
#define SCL_L_COL (get_local_id(0))
#define SCL_L_ROW (get_local_id(1))
#define SCL_L_COL_COUNT (get_local_size(0))
#define SCL_L_ROW_COUNT (get_local_size(1))
#define SCL_L_ID (SCL_L_ROW * SCL_L_COL_COUNT + SCL_L_COL)
#define SCL_ROWS (SCL_ELEMENTS / SCL_COLS)
#define NEUTRAL 255


typedef float SCL_TYPE_0;
typedef float SCL_TYPE_1;

typedef struct {
    __local SCL_TYPE_1* data;
} input_matrix_t;

//In case, local memory is used
SCL_TYPE_1 getData(input_matrix_t* matrix, int x, int y){
    int offsetNorth = SCL_NORTH * SCL_TILE_WIDTH;
    int currentIndex = SCL_L_ROW * SCL_TILE_WIDTH + SCL_L_COL;
    int shift = x - y * SCL_TILE_WIDTH;

    return matrix->data[currentIndex+offsetNorth+shift+SCL_WEST];
}

int USR_FUNC(input_matrix_t* img, __global float* kernelVec, int range)
{
    float sum = 0.0;
    float norm = 0.0;
    int i, j;
    for (i = -range; i <= range; i++) {
        for (j = -range; j <= range; j++) {
            sum += getData(img, i, j) * kernelVec[(i + j) / 2 + range];
            norm += kernelVec[(i + j) / 2 + range];
        }
    }

    return sum / norm;
}

// --- SKELCL END USER CODE ---

#define STENCIL_PADDING_NEUTRAL         1
#define STENCIL_PADDING_NEAREST         0
#define STENCIL_PADDING_NEAREST_INITIAL 0


// The three different padding types affect the values loaded into the border
// regions. By defining macros to determine which value to return, we can
// save on a huge amount of conditional logic between the different padding
// types.
#if STENCIL_PADDING_NEAREST_INITIAL
#  define input(y, x)  SCL_IN[(y) * SCL_COLS + (x)]
#  define border(y, x) SCL_INITIAL[(y) * SCL_COLS + (x)]
#elif STENCIL_PADDING_NEAREST
#  define input(y, x)  SCL_IN[(y) * SCL_COLS + (x)]
#  define border(y, x) SCL_IN[(y) * SCL_COLS + (x)]
#elif STENCIL_PADDING_NEUTRAL
#  define input(y, x)  SCL_IN[(y) * SCL_COLS + (x)]
#  define border(y, x) neutral
#else
// Fall-through case, throw an error.
#  error Unrecognised padding type.
#endif

// Macro function to index flat SCL_LOCAL array.
#define local(y, x) SCL_LOCAL[(y) * SCL_TILE_WIDTH + (x)]

// Define a helper macro which accepts an \"a\" and \"b\" value, returning \"a\"
// if the padding type is neutral, else \"b\".
#if STENCIL_PADDING_NEUTRAL
#  define neutralPaddingIfElse(a, b) (a)
#else
#  define neutralPaddingIfElse(a, b) (b)
#endif

__kernel void SCL_STENCIL(__global SCL_TYPE_1* SCL_IN,
                          __global SCL_TYPE_1* SCL_OUT,
                          __global SCL_TYPE_0* SCL_INITIAL,
                          __local SCL_TYPE_1* SCL_LOCAL,
                          const unsigned int SCL_ELEMENTS,
                          const unsigned int SCL_COLS, __global float* kernelVec, int range) {
#if STENCIL_PADDING_NEUTRAL
        const SCL_TYPE_0 neutral = NEUTRAL;
#endif
        const unsigned int col   = get_global_id(0);
        const unsigned int l_col = get_local_id(0);
        const unsigned int row   = get_global_id(1);
        const unsigned int l_row = get_local_id(1);

        input_matrix_t Mm;
        Mm.data = SCL_LOCAL;
        int i;

        if (l_row == 0 && row < SCL_ELEMENTS / SCL_COLS) {
                const unsigned int SCL_WORKGROUP = SCL_ROWS / get_local_size(1);
                const unsigned int SCL_REST      = SCL_ROWS % get_local_size(1);

                int withinCols     = col < SCL_COLS;
                int withinColsWest = col - SCL_WEST < SCL_COLS;
                int withinColsEast = col + SCL_EAST < SCL_COLS;

                if (row == 0) {
                        for (i = 0; i < SCL_NORTH; i++)
                                if (withinCols)  local(i, l_col + SCL_WEST) = border(0, col);
                                else             local(i, l_col + SCL_WEST) = border(0, SCL_COLS - 1);

                        for (i = SCL_NORTH; i < SCL_TILE_HEIGHT; i++)
                                if (withinCols)  local(i, l_col + SCL_WEST) = input(i - SCL_NORTH, col);
                                else             local(i, l_col + SCL_WEST) = border(i + 1 - SCL_NORTH, -1);

                        if (col < SCL_WEST) {
                                for (i = 0; i < neutralPaddingIfElse(SCL_TILE_HEIGHT, SCL_NORTH); i++)
                                        local(i, l_col) = border(0, 0);
#if !STENCIL_PADDING_NEUTRAL
                                for (i = SCL_NORTH; i < SCL_TILE_HEIGHT; i++)
                                        local(i, l_col) = input(i - SCL_NORTH, 0);
#endif
                        } else if (l_col < SCL_WEST) {
                                for (i = 0; i < neutralPaddingIfElse(SCL_TILE_HEIGHT, SCL_NORTH); i++)
                                        if (withinColsWest)  local(i, l_col) = border(0, col - SCL_WEST);
                                        else                 local(i, l_col) = border(0, SCL_COLS - 1);
                                for (i = SCL_NORTH; i < SCL_TILE_HEIGHT; i++)
                                        if (withinColsWest)  local(i, l_col) = input(i - SCL_NORTH, col - SCL_WEST);
                                        else                 local(i, l_col) = border(i + 1 - SCL_NORTH, -1);
                        }

                        if (col >= SCL_COLS - SCL_EAST) {
                                for (i = 0; i < SCL_TILE_HEIGHT; i++)
                                        local(i, l_col + SCL_WEST + SCL_EAST) = border(0, SCL_COLS - 1);
#if !STENCIL_PADDING_NEUTRAL
                                for (i = SCL_NORTH; i < SCL_TILE_HEIGHT; i++)
                                        local(i, l_col + SCL_WEST + SCL_EAST) = border(i + 1 - SCL_NORTH, -1);
#endif
                        } else if (l_col >= get_local_size(0) - SCL_EAST) {
                                for (i = 0; i < SCL_NORTH; i++)
                                        if (withinColsEast)  local(i, l_col + SCL_WEST + SCL_EAST) = border(0, col + SCL_EAST);
                                        else                 local(i, l_col + SCL_WEST + SCL_EAST) = border(0, SCL_COLS - 1);

                                for (i = SCL_NORTH; i < SCL_TILE_HEIGHT; i++)
                                        if (withinColsEast)  local(i, l_col + SCL_WEST + SCL_EAST) = input(i - SCL_NORTH, col + SCL_EAST);
                                        else                 local(i, l_col + SCL_WEST + SCL_EAST) = border(i + 1 - SCL_NORTH, -1);
                        }
                } else if (row - SCL_NORTH + SCL_TILE_HEIGHT < SCL_ROWS) {
                        for (i = 0; i < SCL_TILE_HEIGHT; i++)
                                if (withinCols)  local(i, l_col + SCL_WEST) = input(row + i - SCL_NORTH, col);
                                else             local(i, l_col + SCL_WEST) = border(row + i + 1 - SCL_NORTH, -1);

                        if (col < SCL_WEST) {
                                for (i = 0; i < SCL_TILE_HEIGHT; i++)
                                        local(i, l_col) = border(row + i - SCL_NORTH, 0);
                        } else if (l_col < SCL_WEST) {
                                for (i = 0; i < SCL_TILE_HEIGHT; i++)
                                        if (withinColsWest)  local(i, l_col) = input(row + i - SCL_NORTH, col - SCL_WEST);
                                        else                 local(i, l_col) = border(row + i + 1 - SCL_NORTH, -1);
                        }

                        if (col >= SCL_COLS - SCL_EAST) {
                                for (i = 0; i < SCL_TILE_HEIGHT; i++)
                                        local(i, l_col + SCL_WEST + SCL_EAST) = border(row + 1 + i - SCL_NORTH, -1);
                        } else if (l_col >= get_local_size(0) - SCL_EAST) {
                                for (i = 0; i < SCL_TILE_HEIGHT; i++)
                                        if (withinColsEast)  local(i, l_col + SCL_WEST + SCL_EAST) = input(row + i - SCL_NORTH, col + SCL_EAST);
                                        else                 local(i, l_col + SCL_WEST + SCL_EAST) = border(row + i + 1 - SCL_NORTH, -1);
                        }
                }
                else {
                        for (i = 0; i < SCL_TILE_HEIGHT; i++) {
                                const int withinRows = row + i - SCL_NORTH < SCL_ROWS;
                                if      ( withinCols &&  withinRows)  local(i, l_col + SCL_WEST) = input(row + i - SCL_NORTH, col);
                                else if ( withinCols && !withinRows)  local(i, l_col + SCL_WEST) = border(0, SCL_ELEMENTS - SCL_COLS + col);
                                else if (!withinCols &&  withinRows)  local(i, l_col + SCL_WEST) = border(row + 1 + i - SCL_NORTH, -1);
                                else                                  local(i, l_col + SCL_WEST) = border(0, SCL_ELEMENTS - 1);
                        }

                        if (col < SCL_WEST) {
                                for (i = 0; i < SCL_TILE_HEIGHT; i++) {
                                        const int withinRows = row + i - SCL_NORTH < SCL_ROWS;
                                        if (withinRows)  local(i, l_col) = border(row + i - SCL_NORTH, 0);
                                        else             local(i, l_col) = border(0, SCL_ELEMENTS - SCL_COLS);
                                }
                        } else if (l_col < SCL_WEST) {
                                for (i = 0; i < SCL_TILE_HEIGHT; i++) {
                                        const int withinRows = row + i - SCL_NORTH < SCL_ROWS;
                                        if      ( withinColsWest &&  withinRows)  local(i, l_col) = input(row + i - SCL_NORTH, col - SCL_WEST);
                                        else if ( withinColsWest && !withinRows)  local(i, l_col) = border(0, SCL_ELEMENTS - SCL_COLS + col - SCL_WEST);
                                        else if (!withinColsWest &&  withinRows)  local(i, l_col) = border(row + 1 + i - SCL_NORTH, -1);
                                        else                                      local(i, l_col) = border(0, SCL_ELEMENTS - 1);
                                }
                        }

                        if (col >= SCL_COLS - SCL_EAST) {
                                for (i = 0; i < SCL_TILE_HEIGHT; i++) {
                                        const int withinRows = row + i - SCL_NORTH < SCL_ROWS;
                                        if (withinRows)  local(i, l_col + SCL_WEST + SCL_EAST) = border(row + 1 + i - SCL_NORTH, -1);
                                        else             local(i, l_col + SCL_WEST + SCL_EAST) = border(0, SCL_ELEMENTS - 1);
                                }
                        } else if (l_col >= get_local_size(0) - SCL_EAST) {
                                for (i = 0; i < SCL_TILE_HEIGHT; i++) {
                                        const int withinRows = row + i - SCL_NORTH < SCL_ROWS;
                                        if      ( withinColsEast &&  withinRows)  local(i, l_col + SCL_WEST + SCL_EAST) = input(row + i - SCL_NORTH, col + SCL_EAST);
                                        else if ( withinColsEast && !withinRows)  local(i, l_col + SCL_WEST + SCL_EAST) = border(0, SCL_ELEMENTS - SCL_COLS + col + SCL_EAST);
                                        else if (!withinColsEast &&  withinRows)  local(i, l_col + SCL_WEST + SCL_EAST) = border(row + 1 + i - SCL_NORTH, -1);
                                        else                                      local(i, l_col + SCL_WEST + SCL_EAST) = border(0, SCL_ELEMENTS - 1);
                                }
                        }
                }
        }

        // Clear up after ourselves.
#undef input
#undef border
#undef local

        barrier(CLK_LOCAL_MEM_FENCE);

        if (row < SCL_ELEMENTS / SCL_COLS && col < SCL_COLS)
                SCL_OUT[row * SCL_COLS + col] = USR_FUNC(&Mm, kernelVec, range);
}


 
