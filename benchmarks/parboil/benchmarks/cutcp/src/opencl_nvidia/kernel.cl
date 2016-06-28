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

// Define a type for a 3D coordinate.  Only 3 vector components are needed.
// Using int4 type because int3 support is missing on some platforms.
typedef int4 xyz;

__kernel void opencl_cutoff_potential_lattice6overlap(
    int binDim_x,
    int binDim_y,
    __global float4 *binBaseAddr,
    int offset,
    float h,                /* lattice spacing */
    float cutoff2,          /* square of cutoff distance */
    float inv_cutoff2,
    __global ener_t *regionZeroAddr, /* address of lattice regions starting at origin */
    int zRegionIndex,
    __constant int *NbrListLen,
    __constant xyz *NbrList
    )
{
  __global float4* binZeroAddr = binBaseAddr + offset;

  __local float AtomBinCache[BIN_CACHE_MAXLEN * BIN_DEPTH * 4];
  __global ener_t *myRegionAddr;
  __local xyz myBinIndex;

  const int xRegionIndex = get_group_id(0);
  const int yRegionIndex = get_group_id(1);
  
  /* thread id */
  const int tid = (get_local_id(2)*get_local_size(1)+get_local_id(1))*get_local_size(0)+get_local_id(0);

  /* neighbor index */
  int nbrid;

  /* this is the start of the sub-region indexed by tid */
  myRegionAddr = regionZeroAddr + ((zRegionIndex*get_num_groups(1)
	+ yRegionIndex)*get_num_groups(0) + xRegionIndex)*REGION_SIZE;
    
  /* spatial coordinate of this lattice point */
  float x = (8 * xRegionIndex + get_local_id(0)) * h;  
  float y = (8 * yRegionIndex + get_local_id(1)) * h;
  float z = (8 * zRegionIndex + get_local_id(2)) * h;

  int totalbins = 0;
  int numbins;

  /* bin number determined by center of region */
  myBinIndex.x = (int) floor((8 * xRegionIndex + 4) * h * BIN_INVLEN);
  myBinIndex.y = (int) floor((8 * yRegionIndex + 4) * h * BIN_INVLEN);
  myBinIndex.z = (int) floor((8 * zRegionIndex + 4) * h * BIN_INVLEN);

  /* first neighbor in list for me to cache */
  nbrid = (tid >> 4);

  numbins = BIN_CACHE_MAXLEN;

#ifndef NEIGHBOR_COUNT
  ener_t energy0 = 0.f;
  ener_t energy1 = 0.f;
  ener_t energy2 = 0.f;
  ener_t energy3 = 0.f;
#else
  ener_t energy0 = 0, energy1 = 0, energy2 = 0, energy3 = 0;
#endif

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
       + (((mul24(k, binDim_y) + j)*binDim_x + i) << BIN_SHIFT);

      /* coalesced read from global memory -
       * retain same ordering in shared memory for now */
      int binIndex = startoff + (bincnt << (3 + BIN_SHIFT));
      int tidmask = tid & 15;

      AtomBinCache[binIndex + tidmask   ] = p_global[tidmask   ];
      AtomBinCache[binIndex + tidmask+16] = p_global[tidmask+16];
    }

    barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);

    /* no warp divergence */
    if (totalbins + BIN_CACHE_MAXLEN > *NbrListLen) {
      numbins = *NbrListLen - totalbins;
    }

    int stopbin = (numbins << BIN_SHIFT);
    for (bincnt = 0; bincnt < stopbin; bincnt+=BIN_SIZE) {
      int i;

      for (i = 0;  i < BIN_DEPTH;  i++) {

        int off = bincnt + (i<<2);

        float aq = AtomBinCache[off + 3];
        if (0.f == aq) 
           break;  /* no more atoms in bin */

        float dx = AtomBinCache[off    ] - x;
        float dz = AtomBinCache[off + 2] - z;
        float dxdz2 = dx*dx + dz*dz;
        float dy = AtomBinCache[off + 1] - y;
        float r2 = dy*dy + dxdz2;

#ifndef NEIGHBOR_COUNT
        if (r2 < cutoff2)
	{
          float s = (1.f - r2 * inv_cutoff2);
          energy0 += aq * rsqrt(r2) * s * s;
        }
#else
	 energy0 += (r2 < cutoff2);
#endif
        dy -= 2.0f*h;
        r2 = dy*dy + dxdz2;

#ifndef NEIGHBOR_COUNT
	if (r2 < cutoff2)
	{
          float s = (1.f - r2 * inv_cutoff2);
          energy1 += aq * rsqrt(r2) * s * s;
        }
#else
	energy1 += (r2 < cutoff2);
#endif
        dy -= 2.0f*h;
        r2 = dy*dy + dxdz2;
#ifndef NEIGHBOR_COUNT
        if (r2 < cutoff2)
	{
          float s = (1.f - r2 * inv_cutoff2);
          energy2 += aq * rsqrt(r2) * s * s;
        }
#else
	energy2 += (r2 < cutoff2);
#endif
        dy -= 2.0f*h;
        r2 = dy*dy + dxdz2;
#ifndef NEIGHBOR_COUNT
        if (r2 < cutoff2)
	{
          float s = (1.f - r2 * inv_cutoff2);
          energy3 += aq * rsqrt(r2) * s * s;
        }
#else
	energy3 += (r2 < cutoff2);
#endif
      } /* end loop over atoms in bin */
    } /* end loop over cached atom bins */
    barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);

  } /* end loop over neighbor list */

  /* store into global memory */
  myRegionAddr[(tid>>4)*64 + (tid&15)     ] = energy0;
  myRegionAddr[(tid>>4)*64 + (tid&15) + 16] = energy1;
  myRegionAddr[(tid>>4)*64 + (tid&15) + 32] = energy2;
  myRegionAddr[(tid>>4)*64 + (tid&15) + 48] = energy3;
}
