/***************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

#ifndef __FILEH__
#define __FILEH__

#ifdef __cplusplus
extern "C"
#endif

void outputData(char* fName, float *h_A0,int nx,int ny,int nz);
char* readFile(const char* fileName);

#endif
