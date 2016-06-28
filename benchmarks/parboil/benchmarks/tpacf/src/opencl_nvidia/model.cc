/***************************************************************************
 *cr
 *cr            (C) Copyright 2007 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/
#include <CL/cl.h>
#include <sys/time.h>
#include <stdio.h>
#include <math.h>
#include <strings.h>
#include <math.h>
#include <parboil.h>

#include "model.h"

unsigned int NUM_SETS;
unsigned int NUM_ELEMENTS;

int readdatafile(char *fname, struct cartesian *data, int npoints)
{
  FILE *infile;
  int lcount = 0;
  float ra, dec;

  if ((infile = fopen(fname, "r")) == NULL)
    {
      fprintf(stderr, "Unable to open data file %s for reading\n", fname);
      return lcount;
    }

  for (lcount = 0; lcount < npoints; lcount++)
    {
      if (fscanf(infile, "%f %f", &ra, &dec) != 2)
	break;

      {
        // data conversion
        float rarad = D2R * ra;
        float decrad = D2R * dec;
        float cd = cos(decrad);
	
	data[lcount].x = cos(rarad) * cd;
	data[lcount].y = sin(rarad) * cd;
	data[lcount].z = sin(decrad);
      }
    }

  fclose(infile);
  
  return lcount;
}

char* readFile(const char* fileName)
{
        FILE* fp;
        fp = fopen(fileName,"r");
        if(fp == NULL)
        {
                printf("Error: Cannot open kernel file for reading!\n");
                exit(1);
        }

        fseek(fp,0,SEEK_END);
        long size = ftell(fp);
        rewind(fp);

        char* buffer = (char*)malloc(sizeof(char)*(size+1));
        if(buffer  == NULL)
        {
                printf("Error: Cannot allocated buffer for file contents!\n");
                fclose(fp);
                exit(1);
        }

        size_t res = fread(buffer,1,size,fp);
        if(res != size)
        {
                printf("Error: Cannot read kernel file contents!\n");
                fclose(fp);
                exit(1);
        }

	buffer[size] = 0;
        fclose(fp);
        return buffer;
}
