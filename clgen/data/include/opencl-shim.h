// Enable OpenCL features and implementation.
#ifndef CLGEN_FEATURES
#define cl_clang_storage_class_specifiers
#define cl_khr_fp64
#include <clc/clc.h>
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define __CL_VERSION_1_0__
#define __CL_VERSION_1_1__
#define __CL_VERSION_1_2__
#define __ENDIAN_LITTLE__
#define __FAST_RELAXED_MATH__
#define __IMAGE_SUPPORT__
#define __OPENCL_VERSION__ 1
#endif  /* CLGEN_FEATURES */

// Unsupported OpenCL storage classes and qualifiers.
#define static
#define generic
#define AS

// Common typedefs
typedef float CONVT;
typedef float DATA_TYPE;
typedef float DATATYPE;
typedef float FLOAT_T;
typedef float FLOAT_TYPE;
typedef float FPTYPE;
typedef float hmc_float;
typedef float inType;
typedef float outType;
typedef float real;
typedef float REAL;
/* typedef float T; */
typedef float Ty;
typedef float TyOut;
typedef float TYPE;
typedef float VALTYPE;
typedef float VALUE_TYPE;
typedef float VECTYPE;
typedef float WORKTYPE;
typedef float2 hmc_complex;
typedef float2 mixed2;
typedef float2 real2;
typedef float2 REAL2;
typedef float3 mixed3;
typedef float3 real3;
typedef float3 REAL3;
typedef float4 FPVECTYPE;
typedef float4 mixed4;
typedef float4 real4;
typedef float4 REAL4;
typedef float4 T4;
typedef int BITMAP_INDEX_TYPE;
typedef int INDEX_TYPE;
typedef int Ix;
typedef int KParam;
typedef int Tp;
typedef int3 Pixel;
typedef unsigned int uint32_t;

// Common constants
#define ACCESSES 16
#define AVER 2
#define BETA 0.5
#define BINS_PER_BLOCK 8
#define BITMAP_SIZE 1024
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
