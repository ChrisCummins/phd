/***************************************************************************
 *
 *            (C) Copyright 2010 The Board of Trustees of the
 *                        University of Illinois
 *                         All Rights Reserved
 *
 ***************************************************************************/

#define UINT32_MAX 4294967295
#define UINT8_MAX 255
#define NUM_PROCS 8

#define min(x,y) ((x<y)?x:y)
#define max(x,y) ((x>y)?x:y)

void dump_histo_img(unsigned char* histo, unsigned int height, unsigned int width, const char *filename);
