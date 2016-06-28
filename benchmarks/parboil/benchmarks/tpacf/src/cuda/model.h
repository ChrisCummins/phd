/***************************************************************************
 *cr
 *cr            (C) Copyright 2007 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/
#ifndef __MODEL_H__
#define __MODEL_H__

#include <parboil.h>

#define D2R M_PI/180.0
#define R2D 180.0/M_PI
#define R2AM 60.0*180.0/M_PI

#define bins_per_dec 5
#define min_arcmin 1.0
#define max_arcmin 10000.0

#define NUM_BINS 20

#define SINGLE_PRECISION 1

#if SINGLE_PRECISION
  #define REAL float
#else
  #define REAL double
#endif

typedef unsigned long hist_t;

struct spherical 
{
  REAL ra, dec;  // latitude, longitude pair
};
 
struct cartesian 
{
  REAL x, y, z;  // cartesian coodrinates
};

int readdatafile(char *fname, struct cartesian *data, int npoints);

void initBinB(struct pb_TimerSet *timers);

#endif
