/***************************************************************************
 *cr
 *cr            (C) Copyright 2007 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/
#ifndef __MODEL_H__
#define __MODEL_H__

#define D2R M_PI/180.0
#define R2D 180.0/M_PI
#define R2AM 60.0*180.0/M_PI

#define bins_per_dec 5
#define min_arcmin 1.0
#define max_arcmin 10000.0

#define NUM_BINS 20
#define BLOCK_SIZE 256

typedef unsigned long hist_t;

struct spherical 
{
  float ra, dec;  // latitude, longitude pair
};
 
struct cartesian 
{
  float x, y, z;  // cartesian coodrinates
};

int readdatafile(char *fname, struct cartesian *data, int npoints);

char* readFile(const char*);

#define CHECK_ERROR(errorMessage)           \
  if(clStatus != CL_SUCCESS)                \
  {                                         \
     printf("Error: %s!\n",errorMessage);   \
     printf("Line: %d\n",__LINE__);         \
     exit(1);                               \
  }

#endif
