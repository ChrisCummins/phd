/*
 * potential lattice is decomposed into size 8^3 lattice point "regions"
 *
 * THIS IMPLEMENTATION:  one thread per lattice point
 * thread block size 128 gives 4 thread blocks per region
 * kernel is invoked for each x-y plane of regions,
 * where gridDim.x is 4*(x region dimension) so that blockIdx.x 
 * can absorb the z sub-region index in its 2 lowest order bits
 *
 * Regions are stored contiguously in memory in row-major order
 *
 * The bins have to not only cover the region, but they need to surround
 * the outer edges so that region sides and corners can still use
 * neighbor list stencil.  The binZeroAddr is actually a shifted pointer into
 * the bin array (binZeroAddr = binBaseAddr + (c*binDim_y + c)*binDim_x + c)
 * where c = ceil(cutoff / binsize).  This allows for negative offsets to
 * be added to myBinIndex.
 *
 * The (0,0,0) spatial origin corresponds to lower left corner of both
 * regionZeroAddr and binZeroAddr.  The atom coordinates are translated
 * during binning to enforce this assumption.
 */

#include "macros.h"

// OpenCL 1.1 support for int3 is not uniform on all implementations, so
// we use int4 instead.  Only the 'x', 'y', and 'z' fields of xyz are used.
typedef int4 xyz;

__kernel void opencl_cutoff_potential_lattice(
    int binDim_x,
    int binDim_y,
    __global float4 *binBaseAddr,
    int offset,
    float h,                /* lattice spacing */
    float cutoff2,          /* square of cutoff distance */
    float inv_cutoff2,
    __global float *regionZeroAddr,  /* address of lattice regions starting at origin */
    int zRegionIndex,
    __constant int *NbrListLen,
    __constant xyz *NbrList
    )
{
  __global float4* binZeroAddr = binBaseAddr + offset;

  __local float AtomBinCache[BIN_CACHE_MAXLEN * BIN_DEPTH * 4];
  __global float *mySubRegionAddr;
  __local xyz myBinIndex;

  /* thread id */
  const int tid = (get_local_id(2)*8 + get_local_id(1))*8 + get_local_id(0);

  /* neighbor index */
  int nbrid;

  /* this is the start of the sub-region indexed by tid */
  mySubRegionAddr = regionZeroAddr + ((zRegionIndex*get_num_groups(1)
	+ get_group_id(1))*(get_num_groups(0)>>2) + (get_group_id(0)>>2))*REGION_SIZE
	+ (get_group_id(0)&3)*SUB_REGION_SIZE;

  /* spatial coordinate of this lattice point */
  float x = (8 * (get_group_id(0) >> 2) + get_local_id(0)) * h;
  float y = (8 * get_group_id(1) + get_local_id(1)) * h;
  float z = (8 * zRegionIndex + 2*(get_group_id(0)&3) + get_local_id(2)) * h;

  int totalbins = 0;
  int numbins;

  /* bin number determined by center of region */
  myBinIndex.x = (int) floor((8 * (get_group_id(0) >> 2) + 4) * h * BIN_INVLEN);
  myBinIndex.y = (int) floor((8 * get_group_id(1) + 4) * h * BIN_INVLEN);
  myBinIndex.z = (int) floor((8 * zRegionIndex + 4) * h * BIN_INVLEN);

  /* first neighbor in list for me to cache */
  nbrid = (tid >> 4);

  numbins = BIN_CACHE_MAXLEN;

  float energy = 0.f;
  for (totalbins = 0;  totalbins < *NbrListLen;  totalbins += numbins) {
    int bincnt;

    /* start of where to write in shared memory */
    int startoff = BIN_SIZE * (tid >> 4);

    /* each half-warp to cache up to 4 atom bins */
    for (bincnt = 0;  bincnt < 4 && nbrid < *NbrListLen;  bincnt++, nbrid += 8) {
      int i = myBinIndex.x + NbrList[nbrid].x;
      int j = myBinIndex.y + NbrList[nbrid].y;
      int k = myBinIndex.z + NbrList[nbrid].z;

      /* determine global memory location of atom bin */
      __global float *p_global = ((__global float *) binZeroAddr)
       + (((k*binDim_y) + j)*binDim_x + i) * BIN_SIZE;

      /* coalesced read from global memory -
       * retain same ordering in shared memory for now */
      int tidmask = tid & 15;
      int binIndex = startoff + bincnt*8*BIN_SIZE;

      AtomBinCache[binIndex + tidmask   ] = p_global[tidmask   ];
      AtomBinCache[binIndex + tidmask+16] = p_global[tidmask+16];
    }
    barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);    

    /* no warp divergence */
    if (totalbins + BIN_CACHE_MAXLEN > *NbrListLen) {
      numbins = *NbrListLen - totalbins;
    }

    for (bincnt = 0;  bincnt < numbins;  bincnt++) {
      int i;
      float r2;

      for (i = 0;  i < BIN_DEPTH;  i++) {
        float ax = AtomBinCache[bincnt * BIN_SIZE + i*4];
        float ay = AtomBinCache[bincnt * BIN_SIZE + i*4 + 1];
        float az = AtomBinCache[bincnt * BIN_SIZE + i*4 + 2];
        float aq = AtomBinCache[bincnt * BIN_SIZE + i*4 + 3];
        if (0.f == aq) break;  /* no more atoms in bin */
        r2 = (ax - x) * (ax - x) + (ay - y) * (ay - y) + (az - z) * (az - z);
        if (r2 < cutoff2) {
          float s = (1.f - r2 * inv_cutoff2);
          energy += aq * rsqrt(r2) * s * s;
        }
      } /* end loop over atoms in bin */
    } /* end loop over cached atom bins */
    barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);

  } /* end loop over neighbor list */

  /* store into global memory */
  mySubRegionAddr[tid] = energy;
}
