/***************************************************************************
 *
 *            (C) Copyright 2010 The Board of Trustees of the
 *                        University of Illinois
 *                         All Rights Reserved
 *
 ***************************************************************************/

#include <parboil.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <smmintrin.h>
#include <emmintrin.h>

#include "FauxBlock.h"
#include "util.h"

/******************************************************************************
* Implementation: Threaded
* Details:
* This implementations is a multi-threaded, SSE version of histogram. The
* span of the data into the histogram is first determined, then every thread
* takes an equal portion of the input and computes a partial histogram for it.
* Finally all the partial copies are combined using SSE intrinsics to generate
* the final histogram.
******************************************************************************/

MAKE_FUNC_7_ARGS(static, histo_scan2, int, rank, int, numThreads, int, img_height, int, img_width, unsigned int*, input, unsigned int*, Min, unsigned int*, Max)
{
    int stride = img_height/numThreads;
    int start = rank*stride;
    int end = (rank == (numThreads-1))? img_height-(numThreads-1)*stride : stride;

    unsigned int minVar = UINT32_MAX;
    unsigned int maxVar = 0;

    int i, j;
    for (j = start; j < start+end; ++j)
    {
        for (i = 0; i < img_width; ++i)
        {
            minVar = min(minVar,input[j*img_width+i]);
            maxVar = max(maxVar,input[j*img_width+i]);
        }
    }

    Min[rank] = minVar;
    Max[rank] = maxVar;
}

MAKE_FUNC_10_ARGS(static, histo_thread2, int, rank, int, numThreads, int, img_height, int, img_width, unsigned int*, input, int, histo_height, int, histo_width, unsigned char*, bins, unsigned int*, Min, unsigned int*, Max)
{
    int stride     = img_height/numThreads;
    unsigned int start = rank*stride;
    unsigned int end   = (rank == (numThreads-1))? (start+img_height-(numThreads-1)*stride): (start+stride);

    int min = Min[0]&~(15);
    int max = (((Max[0]-min+(numThreads*16))/(numThreads*16))*(numThreads*16));

    memset(bins+min, 0, max*sizeof(unsigned char));

    int i, j; for (j = start; j < end; ++j)
    {
        for (i = 0; i < img_width; ++i)
        {
            const unsigned int value = input[j*img_width+i];
            // Increment the appropriate bin, but do not roll-over the max value
            if (bins[value] < UINT8_MAX){
                ++bins[value];
            }
        }
    }
}

MAKE_FUNC_7_ARGS(static, histo_merge2, int, rank, int, numThreads, int, outSize, __m128i*, out, unsigned char**, bins, unsigned int*, Min, unsigned int*, Max)
{
    int minVar = Min[0] & ~(15);
    int stride = ((Max[0]-minVar+(numThreads*16))/(numThreads*16))*16;
    int maxVar = minVar + stride*numThreads;

    int topStride = minVar/numThreads;
    int topStart = rank*topStride;
    int topEnd;

    int botStride = (outSize-maxVar)/numThreads;
    int botStart = maxVar -1 + rank*botStride;
    int botEnd;

    int start = ((minVar+rank*stride)*sizeof(unsigned char))/sizeof(__m128i);
    int end = start + (stride*sizeof(unsigned char))/sizeof(__m128i);

    if (rank == numThreads-1){
        topEnd = topStart + minVar - (numThreads-1)*topStride;
        botEnd = botStart + (outSize-maxVar) - (numThreads-1)*botStride;
    } else {
        topEnd = topStart + topStride;
        botEnd = botStart + botStride;
    }

    memset(((char*)out)+topStart, 0, (topEnd-topStart)*sizeof(char));
    memset(((char*)out)+botStart, 0, (botEnd-botStart)*sizeof(char));

    int i, j;
    for (i= start; i < end; i++){
        __m128i acc = _mm_load_si128(((__m128i*)(bins[0]))+i);
        for (j=1; j<numThreads; j++){
            __m128i b = _mm_load_si128(((__m128i*)(bins[j]))+i);
            acc = _mm_adds_epu8(acc,b);
        }
        _mm_store_si128 (out+i, acc);
    }
}

int main(int argc, char* argv[]) {
  struct pb_TimerSet timers;
  struct pb_Parameters *parameters;

  parameters = pb_ReadParameters(&argc, argv);
  if (!parameters)
    return -1;

  if(!parameters->inpFiles[0]){
    fputs("Input file expected\n", stderr);
    return -1;
  }

  pb_InitializeTimerSet(&timers);
  pb_SwitchToTimer(&timers, pb_TimerID_IO);

  int numIterations;
  int numThreads = 2;

  if (argc >= 2){
    numIterations = atoi(argv[1]);
  } else {
    fputs("Expected at least one command line argument\n", stderr);
    return -1;
  }

  if (argc >= 3){
    numThreads = atoi(argv[2]);
    printf("Number of threads = %d\n", numThreads);
  } else {
    printf("Number of threads = %d (default)\n", numThreads);
  }

  unsigned int img_width, img_height;
  unsigned int histo_width, histo_height;

  FILE* f = fopen(parameters->inpFiles[0],"rb");
  int result = 0;

  result += fread(&img_width,    sizeof(unsigned int), 1, f);
  result += fread(&img_height,   sizeof(unsigned int), 1, f);
  result += fread(&histo_width,  sizeof(unsigned int), 1, f);
  result += fread(&histo_height, sizeof(unsigned int), 1, f);

  if (result != 4){
    fputs("Error reading input and output dimensions from file\n", stderr);
    return -1;
  }

  unsigned int* img = (unsigned int*) malloc (img_width*img_height*sizeof(unsigned int));
  unsigned char* histo = (unsigned char*) calloc (histo_width*histo_height, sizeof(unsigned char));

  result = fread(img, sizeof(unsigned int), img_width*img_height, f);

  fclose(f);

  if (result != img_width*img_height){
    fputs("Error reading input array from file\n", stderr);
    return -1;
  }

  int new_size = (((histo_height*histo_width)+(numThreads*16-1))/(numThreads*16))*(numThreads*16);

  unsigned char** bins = (unsigned char**) malloc (numThreads*sizeof(unsigned char*));
  faux_block_t* blk  = (faux_block_t*) malloc (numThreads*sizeof(faux_block_t));
  faux_block_t* blk1 = (faux_block_t*) malloc (numThreads*sizeof(faux_block_t));
  faux_block_t* blk2 = (faux_block_t*) malloc (numThreads*sizeof(faux_block_t));
  unsigned int* Min  = (unsigned int*) malloc (numThreads*sizeof(unsigned int));
  unsigned int* Max  = (unsigned int*) malloc (numThreads*sizeof(unsigned int));
  __m128i* out  = (__m128i*) calloc (new_size,sizeof(unsigned char));

  int iter, i;
  for (iter = 0; iter < 1000; iter++){
    memset(histo,0,histo_height*histo_width*sizeof(unsigned char));
    for (i = 0; i < img_width*img_height; ++i) {
      const unsigned int value = img[i];
      if (histo[value] < UINT8_MAX) {
        ++histo[value];
      }
    }
  }

  for (i=0; i< numThreads;i++){
    bins[i] = (unsigned char*) calloc (histo_height*histo_width,sizeof(unsigned char));
    blk[i]  = make_histo_scan2_block(i, numThreads, img_height, img_width, img, Min, Max);
    blk1[i] = make_histo_thread2_block(i,numThreads,img_height,img_width,img,histo_height,histo_width,bins[i],Min,Max);
    blk2[i] = make_histo_merge2_block(i,numThreads,new_size,out,bins,Min,Max);
  }

  pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);

  for (iter = 0; iter < 1000; iter++) {
    memset(out, 0, new_size);

    faux_block_run (blk,numThreads);

    for (i=1;i<numThreads;i++){
      Min[0] = min(Min[0],Min[i]);
      Max[0] = max(Max[0],Max[i]);
    }

    faux_block_run (blk1,numThreads);
    faux_block_run (blk2,numThreads);  
  }

  pb_SwitchToTimer(&timers, pb_TimerID_IO);

  for (i=0; i < histo_height*histo_width; i++){
    histo[i] = ((unsigned char*)out)[i];
  }

  for (i=0; i<numThreads;i++){
    free(bins[i]);
  }
  free(bins);
  free(blk);
  free(blk1);
  free(blk2);
  free(Min);
  free(Max);

  if (parameters->outFile) {
    dump_histo_img(histo, histo_height, histo_width, parameters->outFile);
  }

  pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);

  free(img);
  free(histo);

  pb_SwitchToTimer(&timers, pb_TimerID_NONE);

  printf("\n");
  pb_PrintTimerSet(&timers);
  pb_FreeParameters(parameters);

  return 0;
}
