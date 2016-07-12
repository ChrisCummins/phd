/***************************************************************************
 *cr
 *cr            (C) Copyright 2007 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/
#include <sys/time.h>
#include <stdio.h>
#include <math.h>
#include <strings.h>
#include <math.h>

#include "model.h"

int readdatafile(char *fname, struct cartesian *data, int npoints)
{
  FILE *infile;
  int lcount = 0;
  REAL ra, dec;

  if ((infile = fopen(fname, "r")) == NULL)
    {
      fprintf(stderr, "Unable to open data file %s for reading\n", fname);
      return lcount;
    }

  for (lcount = 0; lcount < npoints; lcount++)
    {
      #if SINGLE_PRECISION
      if (fscanf(infile, "%f %f", &ra, &dec) != 2)
      #else
      if (fscanf(infile, "%lf %lf", &ra, &dec) != 2)
      #endif
	break;

      {
        // data conversion
        REAL rarad = D2R * ra;
        REAL decrad = D2R * dec;
        REAL cd = cos(decrad);
	
	data[lcount].x = cos(rarad) * cd;
	data[lcount].y = sin(rarad) * cd;
	data[lcount].z = sin(decrad);
      }
    }

  fclose(infile);
  
  return lcount;
}

