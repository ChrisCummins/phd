// A shim header providing common definitions.
//
// Coarse grained control is provided over what is defined using include guards.
// To prevent the definition of unsupported storage classes and qualifiers:
//   -DCLGEN_OPENCL_SHIM_NO_UNSUPPORTED_STORAGE_CLASSES_AND_QUALIFIERS
// To prevent the definition of common types:
//   -DCLGEN_OPENCL_SHIM_NO_COMMON_TYPES
// To prevent the definition of common constants:
//   -DCLGEN_OPENCL_SHIM_NO_COMMON_CONSTANTS
//
// Copyright (c) 2016, 2017, 2018, 2019 Chris Cummins.
//
// clgen is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// clgen is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with clgen.  If not, see <https://www.gnu.org/licenses/>.

// Unsupported OpenCL storage classes and qualifiers.
#ifndef CLGEN_OPENCL_SHIM_NO_UNSUPPORTED_STORAGE_CLASSES_AND_QUALIFIERS
#define static
#define generic
#define AS
#endif  // CLGEN_OPENCL_SHIM_NO_UNSUPPORTED_STORAGE_CLASSES_AND_QUALIFIERS


// Common types.
#ifndef CLGEN_OPENCL_SHIM_NO_COMMON_TYPES
#define CONVT float
#define DATA_TYPE float
#define DATATYPE float
#define FLOAT_T float
#define FLOAT_TYPE float
#define FPTYPE float
#define hmc_float float
#define inType float
#define outType float
#define real float
#define REAL float
#define Ty float
#define TyOut float
#define TYPE float
#define VALTYPE float
#define VALUE_TYPE float
#define VECTYPE float
#define WORKTYPE float
#define hmc_complex float2
#define mixed2 float2
#define real2 float2
#define REAL2 float2
#define mixed3 float3
#define real3 float3
#define REAL3 float3
#define FPVECTYPE float4
#define mixed4 float4
#define real4 float4
#define REAL4 float4
#define T4 float4
#define BITMAP_INDEX_TYPE int
#define INDEX_TYPE int
#define Ix int
#define KParam int
#define Tp int
#define Pixel int3
#define uint32_t unsigned int
#endif  // CLGEN_OPENCL_SHIM_NO_COMMON_TYPES

// Common constants
#ifndef CLGEN_OPENCL_SHIM_NO_COMMON_CONSTANTS
#define ACCESSES 16
#define AVER 2
#define BETA 0.5
#define BINS_PER_BLOCK 8
#define BITMAP_SIZE 1024
#define BLACK 0
#define BLK_X 8
#define BLK_Y 8
#define BLOCK 32
#define BLOCK_DIM 2
#define BLOCK_SIZE 64
#define BLOCK_SIZE_WITH_PAD 64
#define BLOCK_X 8
#define BLOCK_Y 8
#define BLOCKSNUM 64
#define BUCKETS 8
#define CHARS 16
#define CLASS 'A'  // Used in npb-3.3
#define COLS 64
#define COLUMNS_BLOCKDIM_X 16
#define COLUMNS_BLOCKDIM_Y 8
#define COLUMNS_HALO_STEPS 1
#define COLUMNS_RESULT_STEPS 4
#define CONCURRENT_THREADS 128
#define CUTOFF_VAL 0.5
#define DEF_DIM 3
#define DIAMETER 16
#define DIM 3
#define DIMX 128
#define DIMY 64
#define DIMZ 8
#define DIRECTIONS 4
#define DISPATCH_SIZE 64
#define ELEMENTS 1024
#define EPSILON 0.5
#define EUCLID 1
#define EXPRESSION
#define EXTRA 4
#define FILTER_LENGTH 128
#define FLEN 100
#define FOCALLENGTH 100
#define FORCE_WORK_GROUP_SIZE 32
#define GAUSS_RADIUS 5
#define GLOBALSIZE_LOG2 10
#define GROUP 128
#define GROUPSIZE 128
#define HEIGHT 128
#define HISTOGRAM64_WORKGROUP_SIZE 64
#define IMAGEH 512
#define IMAGEW 1024
#define INITLUT 0
#define INITSTEP 0
#define INPUT_WIDTH 256
#define INVERSE -1
#define ITERATIONS 1000
#define KAPPA 4
#define KERNEL_RADIUS 8
#define KEY 8
#define KITERSNUM 64
#define KVERSION 1
#define LAMBDA .5
#define LENGTH 1024
#define LIGHTBUFFERDIM 128
#define LOCAL_H 8
#define LOCAL_MEM_SIZE 2048
#define LOCAL_MEMORY_BANKS 16
#define LOCAL_SIZE 128
#define LOCAL_SIZE_LIMIT 1024
#define LOCAL_W 128
#define LOCALSIZE_LOG2 5
#define LOG2_WARP_SIZE 5
#define LOOKUP_GAP 16
#define LROWS 16
#define LSIZE 64  // Used in npb-3.3
#define LUTSIZE 1024
#define LUTSIZE_LOG2 10
#define LWS 128
#define M_PI 3.14
#define MASS 100
#define MAX 100
#define MAX_PARTITION_SIZE 1024
#define MAXWORKX 8
#define MAXWORKY 8
#define MERGE_WORKGROUP_SIZE 32
#define MOD 16
#define MT_RNG_COUNT 8
#define MULT 4
#define N_CELL_ENTRIES 128
#define N_GP 8
#define N_PER_THREAD 16
#define NCHAINLENGTH 100
#define NDIM 4
#define NEEDMEAN 1
#define NMIXTURES
#define NROUNDS 100
#define NSIEVESIZE 64
#define NSPACE 512
#define NSPIN 8
#define NTIME 100
#define NUM_OF_THREADS 1024
#define NUMBER_THREADS 32
#define OFFSET 2
#define ONE 1
#define OP_WARPSIZE 32
#define PADDING 8
#define PADDINGX 4
#define PADDINGY 2
#define PI 3.14
#define PRESCAN_THREADS 128  /* Used in parboil-0.2 histo */
#define PULSELOCALOFFSET 8
#define PULSEOFF 16
#define QPEX 1
#define QUEUE_SIZE 128
#define RADIUS 8
#define RADIUSX 16
#define RADIUSY 16
#define RADIX 2
#define REGION_WIDTH 16
#define RESULT_SIZE 512
#define ROWS_BLOCKDIM_X 16
#define ROWS_BLOCKDIM_Y 4
#define ROWS_HALO_STEPS 1
#define ROWS_RESULT_STEPS 4
#define ROWSIZE 128
#define SAT .5
#define SCALE
#define SCREENHEIGHT 1920
#define SCREENWIDTH 1080
#define SHAREMASK 0
#define SIGN 1
#define SIMD_WIDTH 32
#define SIMDROWSIZE
#define SINGLE_PRECISION 1
#define SIZE 1024
#define SLICE 32
#define STACK_SIZE 1024
#define STEP 8
#define SUBWAVE_SIZE 32
#define TDIR 1
#define THREADBUNCH 32
#define THREADS 2048
#define THREADS_H 16
#define THREADS_W 128
#define THREADS_X 128
#define THREADS_Y 16
#define THRESHOLD 0.5
#define TILE_COLS 16
#define TILE_COLS 16
#define TILE_DIM 16
#define TILE_HEIGHT 16
#define TILE_M 16
#define TILE_N 16
#define TILE_ROWS 16
#define TILE_SIZE 16
#define TILE_TB_HEIGHT 16
#define TILE_WIDTH 16
#define TILEH 16
#define TILESH 64
#define TILESW 16
#define TILEW 16
#define TRANSPOSEX 1
#define TRANSPOSEY -1
#define TREE_DEPTH 3
#define TX 8
#define TY 16
#define UNROLL 8
#define VECSIZE 128
#define VOLSPACE 4
#define WARP_COUNT 8
#define WARPS_PER_GROUP 8
#define WDEPTH 16
#define WDIM 2
#define WG_H 8
#define WG_SIZE 128
#define WG_SIZE_X 32
#define WG_SIZE_Y 8
#define WG_W 32
#define WIDTH 16
#define WINDOW 16
#define WORK_GROUP_SIZE 256
#define WORK_ITEMS 128
#define WORKGROUP_SIZE 256
#define WORKGROUPSIZE 256
#define WORKSIZE 128
#define WSIZE 1024
#define XDIR 0
#define XSIZE 128
#define YDIR 1
#define YSIZE 64
#define ZDIR 2
#define ZERO 0
#define ZSIZE 128
#endif  // CLGEN_OPENCL_SHIM_NO_COMMON_CONSTANTS
