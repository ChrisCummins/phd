#include <libcecl.h>
/**********************************************************************
Copyright ©2015 Advanced Micro Devices, Inc. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

•	Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
•	Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or
 other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
 DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
********************************************************************/

#include "Shared.h"
#include "Log.h"
#include "Timer.h"

#define  WS 64      // work group size

#define SUCCESS 0
#define FAILURE 1
#define EXPECTED_FAILURE 2


int nLoops;         // overall number of timing loops
int nRepeats;       // # of repeats for each transfer step
int nKernelsBatch;  // time batch of kernels
int nSkip;          // to discount lazy allocation effects, etc.
int nKLoops;        // repeat inside kernel to show peak mem B/W
int status = 0;

int nBytes;         // input and output buffer size
int nThreads;       // number of GPU work items
int nItems;         // number of 32-bit 4-vectors for GPU kernel
int nAlign;         // safe bet for most PCs

int nBytesResult;

bool printLog;
bool doHost;
int  whichTest;
int  nWF;

TestLog *tlog;
bool vFailure=false;

void *memIn,
     *memOut,
     *memResult;

cl_mem inputBuffer,
       resultBuffer;

void usage()
{
            std::cout << "\nOptions:\n\n";
            std::cout << "   -d <n>             number of GPU device\n" ;
            std::cout << "   -nl <n>            number of timing loops\n";
            std::cout << "   -nr <n>            repeat each timing <n> times\n";
            std::cout << "   -nk <n>            number of loops in kernel\n";
            std::cout << "   -nkb <n>           number of kernel launches per batch\n";
            std::cout << "   -nb <n>            buffer size in bytes\n";
            std::cout << "   -nw <n>            # of wave fronts per SIMD\n";
            std::cout << "                      (default: 7)\n";
            std::cout << "   -l                 print complete timing log\n";
            std::cout << "   -s <n>             skip first <n> timings for average\n";
            std::cout << "                      (default: 1)\n";
            std::cout << "   -if <n>            input flags\n";
            std::cout << "                      (ok to use multiple):\n";

            for( int i = 0; i < nFlags; i++ )
               std::cout << "        " << i << "  " << flags[i].s << std::endl;
            std::cout << "\n";

            std::cout << "   -h                 print this message\n\n";

            exit(SUCCESS);
}

void parseOptions(int argc, char * argv[])
{
    while(--argc) 
    {
        if( strcmp(argv[argc], "-nl") == 0 )
            nLoops = atoi( argv[ argc + 1 ] );

        if( strcmp(argv[argc], "-nb") == 0 )
            nBytes = atoi( argv[ argc + 1 ] );

        if( strcmp(argv[argc], "-nr") == 0 )
            nRepeats = atoi( argv[ argc + 1 ] );

        if( strcmp(argv[argc], "-nk") == 0 )
           nKLoops = atoi( argv[ argc + 1 ] );

        if( strcmp(argv[argc], "-nkb") == 0 )
           nKernelsBatch = atoi( argv[ argc + 1 ] );

        if( strcmp(argv[argc], "-nw") == 0 )
           nWF = atoi( argv[ argc + 1 ] );

        if( strcmp(argv[argc], "-s") == 0 )
            nSkip = atoi( argv[ argc + 1 ] );

        if( strcmp(argv[argc], "-l") == 0 )
            printLog = true;

        if( strcmp(argv[argc], "-if") == 0 )
        {
            int f = atoi( argv[ argc + 1 ] );
            if( f < nFlags )
                inFlags |= flags[ f ].f;
        }

        if( strcmp(argv[argc], "-d") == 0 )
        {
            devnum = atoi( argv[ argc + 1 ] );
        }

        if( strcmp(argv[argc], "-h") == 0 )
           usage();
    }

    cl_mem_flags f = CL_MEM_READ_ONLY |
                     CL_MEM_WRITE_ONLY |
                     CL_MEM_READ_WRITE;

    if( (inFlags & f) == 0 )
             inFlags |= CL_MEM_READ_ONLY;

    nSkip = nLoops > nSkip ? nSkip : 0;
}

void timedBufMappedWrite( cl_command_queue queue,
                          cl_mem buf,
                          unsigned char v )
{
    CPerfCounter t1, t2, t3;
    cl_int ret;
    cl_event ev;
    void *ptr;
    cl_map_flags mapFlag = CL_MAP_READ | CL_MAP_WRITE;

    t1.Reset();
    t2.Reset();
    t3.Reset();

    t1.Start();

    ptr = (void * ) CECL_MAP_BUFFER( queue,
                                        buf,
                                        CL_FALSE,
                                        mapFlag,
                                        0,
                                        nBytes,
                                        0, NULL, 
                                        &ev,
                                        &ret );
    ASSERT_CL_RETURN( ret );

    clFlush( queue );
    spinForEventsComplete( 1, &ev );

    t1.Stop();

    t2.Start();

    memset( ptr, v, nBytes );

    t2.Stop();

    t3.Start();

    ret = clEnqueueUnmapMemObject( queue,
                                   buf,
                                   (void *) ptr,
                                   0, NULL, &ev );
    ASSERT_CL_RETURN( ret );

    clFlush( queue );
    spinForEventsComplete( 1, &ev );

    t3.Stop();
}

void timedKernel( cl_command_queue queue,
                  cl_kernel        kernel,
                  cl_mem           bufSrc,
                  cl_mem           bufDst,
                  unsigned char    v,
                  bool quiet )
{
     cl_int ret;
     cl_event ev = 0;
     CPerfCounter t; 

     cl_uint nItemsPerThread = nItems / nThreads;

     size_t global_work_size[2] = { nThreads, 0 };
     size_t local_work_size[2] = { WS, 0 };

     cl_uint val = 0;

     for(int i = 0; i < sizeof( cl_uint ); i++)
        val |= v << (i * 8);

     CECL_SET_KERNEL_ARG( kernel, 0, sizeof(void *),  (void *) &bufSrc );
     CECL_SET_KERNEL_ARG( kernel, 1, sizeof(void *),  (void *) &bufDst );
     CECL_SET_KERNEL_ARG( kernel, 2, sizeof(cl_uint), (void *) &nItemsPerThread);
     CECL_SET_KERNEL_ARG( kernel, 3, sizeof(cl_uint), (void *) &val);
     CECL_SET_KERNEL_ARG( kernel, 4, sizeof(cl_uint), (void *) &nKLoops);

     t.Reset();
     t.Start();

     for( int r = 0; r < nKernelsBatch; r++ )
     {
        ret = CECL_ND_RANGE_KERNEL( queue,
                                      kernel,
                                      1,
                                      NULL,
                                      global_work_size,
                                      local_work_size,
                                      0, NULL, &ev );
         ASSERT_CL_RETURN( ret );
     }

     clFlush( queue );
     spinForEventsComplete( 1, &ev );

     t.Stop();

     if( !quiet )
         tlog->Timer( "%32s  %lf s\n", "CECL_ND_RANGE_KERNEL():", 
                      t.GetElapsedTime() / nKernelsBatch, nBytes, nKLoops );
}

void timedReadKernelVerify( cl_command_queue queue,
                            cl_kernel        kernel,
                            cl_mem           bufSrc,
                            cl_mem           bufRes,
                            unsigned char    v,
                            bool             quiet )
{
    cl_int ret;
    cl_event ev;

    timedKernel( queue, kernel, bufSrc, bufRes, v, quiet );

    ret = CECL_READ_BUFFER( queue,
                               bufRes,
                               CL_FALSE,
                               0,
                               nBytesResult,
                               memResult,
                               0, NULL,
                               &ev );
    ASSERT_CL_RETURN( ret );

    clFlush( queue );
    spinForEventsComplete( 1, &ev );

     cl_uint sum = 0;

     for(int i = 0; i < nThreads / WS; i++)
         sum += ((cl_uint *) memResult)[i];

     bool verify;

     if( sum == nBytes / sizeof(cl_uint) )
         verify = true;
     else
     {
         verify = false;
         vFailure = true;
     }
 
    if( !quiet )
    {
      if( verify )
        tlog->Msg( "%s\n", "\nPassed" );
      else
        tlog->Error( "%s\n", "\nFailed" );
    }
}

void createBuffers()
{
   // host memory buffers

    // return if a error or expected error has occured	
    if(status != SUCCESS) 
    return;
    

#ifdef _WIN32
   memIn =      (void *) _aligned_malloc( nBytes, nAlign );
   memOut =     (void *) _aligned_malloc( nBytes, nAlign );
   memResult =  (void *) _aligned_malloc( nBytesResult, nAlign );
#else
   memIn =      (void *) memalign( nAlign, nBytes );
   memOut =     (void *) memalign( nAlign, nBytes );
   memResult =  (void *) memalign( nBytesResult, nBytesResult );
#endif

   if( memIn == NULL ||
       memOut == NULL ||
       memResult == NULL ) 
   {
       fprintf( stderr, "%s:%d: error: %s\n", \
                __FILE__, __LINE__, "could not allocate host buffers\n" );
       exit(FAILURE);
   }

   // CL buffers

   cl_int ret;
   void *hostPtr = NULL;

   if( inFlags & CL_MEM_USE_HOST_PTR ||
       inFlags & CL_MEM_COPY_HOST_PTR )
       hostPtr = memIn;

   inputBuffer = CECL_BUFFER( context,
                                 inFlags,
                                 nBytes,
                                 hostPtr, &ret );

   ASSERT_CL_RETURN( ret );

   resultBuffer = CECL_BUFFER( context,
                                  CL_MEM_READ_WRITE,
                                  nBytesResult,
                                  NULL, &ret );
   ASSERT_CL_RETURN( ret );
}

void cleanup()
{
	// Releases OpenCL resources (Context, Memory etc.)
    cl_int ret;

	ret = clReleaseKernel(read_kernel);
    ASSERT_CL_RETURN( ret );

	ret = clReleaseKernel(write_kernel);
    ASSERT_CL_RETURN( ret );

	ret = clReleaseProgram(program);
    ASSERT_CL_RETURN( ret );

	ret = clReleaseMemObject(inputBuffer);
	ASSERT_CL_RETURN( ret );

	ret = clReleaseMemObject(resultBuffer);
	ASSERT_CL_RETURN( ret );

	ret = clReleaseCommandQueue(queue);
    ASSERT_CL_RETURN( ret );

    ret = clReleaseContext(context);
    ASSERT_CL_RETURN( ret );
}

void printHeader()
{
// return if a error or expected error has occured	
    if(status != SUCCESS) 
    return;   

    std::cout <<"\nDevice " << devnum << ":            " << devname << std::endl;

#ifdef _WIN32
   std::cout << "Build:               _WINxx"; 
#ifdef _DEBUG
   std::cout << " DEBUG";
#else
   std::cout << " release";
#endif
   std::cout << "\n" ;
#else
#ifdef NDEBUG
    std::cout <<"Build:               release\n";
#else
    std::cout <<"Build:               DEBUG\n";
#endif
#endif

   std::cout << "GPU work items:      " << nThreads << std::endl;
   std::cout << "Buffer size:         " << nBytes << std::endl;
   std::cout << "Timing loops:        " << nLoops << std::endl; 
   std::cout << "Repeats:             " << nRepeats << std::endl;
   std::cout << "Kernel loops:        " << nKLoops << std::endl;
   std::cout << "Batch size:          " << nKernelsBatch << std::endl;

   std::cout << "inputBuffer:         " << std::endl;

   for( int i = 0; i < nFlags; i++ )
      if( inFlags & flags[i].f )
          std::cout << flags[i].s << std::endl;

   std::cout <<"\n\n";
}

void printResults()
{
   if( printLog ) 
      tlog->printLog();

   tlog->printSummary( nSkip );

   std::cout << "\n" ;
   fflush( stdout );
}

void runKernelLaunchTest()
{
   CPerfCounter t;
   int  nl = nLoops;
   
    // return if an error or expected error has already occured
    if(status)
		return;

   t.Reset(); t.Start();

   while( nl-- )
   {
      tlog->loopMarker();

      for(int i = 0; i < nRepeats; i++)
          timedBufMappedWrite( queue, inputBuffer, nl & 0xff );

      tlog->Msg( "\n%s\n\n", "GPU kernel read of inputBuffer" );

      for(int i = 0; i < nRepeats; i++)
          timedReadKernelVerify( queue, read_kernel, inputBuffer, resultBuffer, nl & 0xff, true );

      tlog->Msg( "%s\n", "" );
   }
}

void initDefaults()
{
   devnum =         0;
   nLoops =         1000;
   nKernelsBatch =  1;
   nRepeats =       1;
   nSkip =          2; 
   nKLoops =        1;  

   nBytes =         131072;
   nAlign =         4096;
   printLog =       false;
   doHost =         true;
   whichTest =      0;
   nWF =            7;
}

void computeGlobals()
{
    // educated guess of optimal work size

    int minBytes = WS * sizeof( cl_uint ) * 4;

    nBytes = ( nBytes / minBytes ) * minBytes;
    nBytes = nBytes < minBytes ? minBytes : nBytes;
    nItems = nBytes / ( 4 * sizeof(cl_uint));
    
    int maxThreads = nBytes / ( 4 * sizeof( cl_uint ) );

    nThreads = deviceMaxComputeUnits * nWF * WS;
 
    if( nThreads > maxThreads )
        nThreads = maxThreads;
    else
    {
        while( nItems % nThreads != 0 )
            nThreads += WS;
    }

    nBytesResult = ( nThreads / WS ) * sizeof(cl_uint);
}

int main(int argc, char **argv)
{
    initDefaults();
    parseOptions( argc, argv );

    tlog = new TestLog( nLoops * nRepeats * 50 );

    initCL( (char *) "KernelLaunch_Kernels.cl" );

    computeGlobals();
    printHeader();
    createBuffers();
    runKernelLaunchTest();
    printResults();

	cleanup();

    if(vFailure || (status == FAILURE))
    {
		std::cerr << "Failed!" << std::endl;
        return FAILURE;
    }
    else	// Passed or Expected Failure
    {
		if(status == SUCCESS)
			std::cout << "Passed!" << std::endl;
        return SUCCESS;
    }
}
