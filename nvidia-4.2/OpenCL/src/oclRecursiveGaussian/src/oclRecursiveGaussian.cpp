#include <cecl.h>
/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

// Standard utilities and system includes, plus project specific items
//*****************************************************************************
// OpenGL Graphics Includes
#include <GL/glew.h>
#ifdef UNIX
    #include <GL/glxew.h>
#endif
#if defined (_WIN32)
    #include <GL/wglew.h>
#endif
#if defined(__APPLE__) || defined(MACOSX)
    #include <OpenGL/OpenGL.h>
    #include <GLUT/glut.h>
#else
    #include <GL/freeglut.h>
#endif

// Includes
#include <stdio.h>
#include <stdlib.h>
#include <memory>
#include <iostream>
#include <cassert>

// Project Includes
#include "oclRecursiveGaussian.h"

// Shared QA Test Includes
#include <shrQATest.h>

#ifndef min
#define min(a,b) (a < b ? a : b)
#endif

int *pArgc = NULL;
char **pArgv = NULL;

// Defines and globals for recursive gaussian processing demo
//*****************************************************************************
#define REFRESH_DELAY	  10 //ms

float fSigma = 10.0f;               // filter sigma (blur factor)
int iOrder = 0;                     // filter order
int iTransposeBlockDim = 16;        // initial height and width dimension of 2D transpose workgroup 
int iNumThreads = 64;	            // number of threads per block for Gaussian

// Image data vars
const char* cImageFile = "StoneRGB.ppm";
unsigned int uiImageWidth = 1920;   // Image width
unsigned int uiImageHeight = 1080;  // Image height
unsigned int* uiInput = NULL;       // Host buffer to hold input image data
unsigned int* uiTemp = NULL;        // Host buffer to hold intermediate image data
unsigned int* uiOutput = NULL;      // Host buffer to hold output image data

// OpenGL, Display Window and GUI Globals
int iGLUTWindowHandle;              // handle to the GLUT window
int iGLUTMenuHandle;                // handle to the GLUT menu
int iGraphicsWinPosX = 0;           // GL Window X location
int iGraphicsWinPosY = 0;           // GL Window Y location
int iGraphicsWinWidth = 1024;       // GL Window width
int iGraphicsWinHeight = ((float)uiImageHeight / (float)uiImageWidth) * iGraphicsWinWidth;  // GL Windows height
float fZoom = 1.0f;                 // pixel display zoom   
int iFrameCount = 0;                // FPS count for averaging
int iFrameTrigger = 90;             // FPS trigger for sampling
int iFramesPerSec = 45;             // frames per second
double dProcessingTime = 0.0;       // Computation time accumulator
bool bFullScreen = false;           // state var for full screen mode or not
GLint iVsyncState;                  // state var to cache startup Vsync setting

// app configuration parms
const char* cProcessor [] = {"OpenCL GPU", "Host C++ CPU"};
bool bFilter = true;                // state var for whether filter is enaged or not
int iProcFlag = 0;                  // 0 = GPU, 1 = CPU
shrBOOL bNoPrompt = shrFALSE;		// false = normal GL loop, true = Finite period of GL loop (a few seconds)
shrBOOL bQATest = shrFALSE;			// false = normal GL loop, true = run No-GL test sequence
int iTestSets = 3;                  // # of loop set retriggers before auto exit when bNoPrompt = shrTrue  

// OpenCL vars
const char* clSourcefile = "RecursiveGaussian.cl";  // OpenCL kernel source file
char* cPathAndName = NULL;          // var for full paths to data, src, etc.
char* cSourceCL;                    // Buffer to hold source for compilation 
cl_platform_id cpPlatform;          // OpenCL platform
cl_context cxGPUContext;            // OpenCL context
cl_command_queue cqCommandQueue;    // OpenCL command que
cl_device_id* cdDevices = NULL;     // device list
cl_uint uiNumDevsUsed = 1;          // Number of devices used in this sample 
cl_program cpProgram;               // OpenCL program
cl_kernel ckSimpleRecursiveRGBA;    // OpenCL Kernel for simple recursion
cl_kernel ckRecursiveGaussianRGBA;  // OpenCL Kernel for gaussian recursion
cl_kernel ckTranspose;              // OpenCL for transpose
cl_mem cmDevBufIn;                  // OpenCL device memory input buffer object
cl_mem cmDevBufTemp;                // OpenCL device memory temp buffer object
cl_mem cmDevBufOut;                 // OpenCL device memory output buffer object
size_t szBuffBytes;                 // Size of main image buffers
size_t szGaussGlobalWork;           // global # of work items in single dimensional range
size_t szGaussLocalWork;            // work group # of work items in single dimensional range
size_t szTransposeGlobalWork[2];    // global # of work items in 2 dimensional range
size_t szTransposeLocalWork[2];     // work group # of work items in a 2 dimensional range
size_t szParmDataBytes;			    // Byte size of context information
size_t szKernelLength;			    // Byte size of kernel code
cl_int ciErrNum;		            // Error code var
const char* cExecutableName;

// Forward Function declarations
//*****************************************************************************
// OpenCL functionality
double GPUGaussianFilterRGBA(GaussParms* pGP);
void GPUGaussianSetCommonArgs(GaussParms* pGP);

// OpenGL functionality
void InitGL(int* argc, char** argv);
void DeInitGL();
void DisplayGL();
void Reshape(int w, int h);
void Idle(void);
void KeyboardGL(unsigned char key, int x, int y);
void MenuGL(int i);
void timerEvent(int value);

// Helpers
void TestNoGL();
void TriggerFPSUpdate();
void ShowMenuItems();
void Cleanup(int iExitCode);
void (*pCleanup)(int) = &Cleanup;

// Main program
//*****************************************************************************
int main(int argc, char** argv)
{
	pArgc = &argc;
	pArgv = argv;

	shrQAStart(argc, argv);

    // Start logs 
	cExecutableName = argv[0];
    shrSetLogFileName ("oclRecursiveGaussian.txt");
    shrLog("%s Starting (using %s)...\n\n", argv[0], clSourcefile); 

    // Get command line args for quick test or QA test, if provided
    bNoPrompt = shrCheckCmdLineFlag(argc, (const char**)argv, "noprompt");
    bQATest   = shrCheckCmdLineFlag(argc, (const char**)argv, "qatest");

    // Menu items
	if (!(bQATest))
    {
        ShowMenuItems();
    }

    // Find the path from the exe to the image file and load the image
    cPathAndName = shrFindFilePath(cImageFile, argv[0]);
    oclCheckErrorEX(cPathAndName != NULL, shrTRUE, pCleanup);
    ciErrNum = shrLoadPPM4ub(cPathAndName, (unsigned char **)&uiInput, &uiImageWidth, &uiImageHeight);
    oclCheckErrorEX(ciErrNum, shrTRUE, pCleanup);
    shrLog("Image Width = %i, Height = %i, bpp = %i\n\n", uiImageWidth, uiImageHeight, sizeof(unsigned int)<<3);

    // Allocate intermediate and output host image buffers
    szBuffBytes = uiImageWidth * uiImageHeight * sizeof (unsigned int);
    uiTemp = (unsigned int*)malloc(szBuffBytes);
    uiOutput = (unsigned int*)malloc(szBuffBytes);
    shrLog("Allocate Host Image Buffers...\n"); 

    // Initialize OpenGL items (if not No-GL QA test)
	if (!(bQATest))
	{
		InitGL(&argc, argv);
	}
	shrLog("%sInitGL...\n", bQATest ? "Skipping " : "Calling "); 

    // Get the NVIDIA platform
    ciErrNum = oclGetPlatformID(&cpPlatform);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    shrLog("clGetPlatformID...\n"); 

    //Get all the devices
    cl_uint uiNumDevices = 0;           // Number of devices available
    cl_uint uiTargetDevice = 0;	        // Default Device to compute on
    cl_uint uiNumComputeUnits;          // Number of compute units (SM's on NV GPU)
    shrLog("Get the Device info and select Device...\n");
    ciErrNum = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_CPU, 0, NULL, &uiNumDevices);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    cdDevices = (cl_device_id *)malloc(uiNumDevices * sizeof(cl_device_id) );
    ciErrNum = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_CPU, uiNumDevices, cdDevices, NULL);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    // Set target device and Query number of compute units on uiTargetDevice
    shrLog(" # of Devices Available = %u\n", uiNumDevices); 
    if(shrGetCmdLineArgumentu(argc, (const char**)argv, "device", &uiTargetDevice)== shrTRUE) 
    {
        uiTargetDevice = CLAMP(uiTargetDevice, 0, (uiNumDevices - 1));
    }
    shrLog(" Using Device %u: ", uiTargetDevice); 
    oclPrintDevName(LOGBOTH, cdDevices[uiTargetDevice]);  
    ciErrNum = clGetDeviceInfo(cdDevices[uiTargetDevice], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(uiNumComputeUnits), &uiNumComputeUnits, NULL);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    shrLog("\n # of Compute Units = %u\n\n", uiNumComputeUnits); 

    //Create the context
    cxGPUContext = clCreateContext(0, uiNumDevsUsed, &cdDevices[uiTargetDevice], NULL, NULL, &ciErrNum);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    shrLog("clCreateContext...\n\n"); 

    // Create a command-queue 
    cqCommandQueue = CECL_CREATE_COMMAND_QUEUE(cxGPUContext, cdDevices[uiTargetDevice], 0, &ciErrNum);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    shrLog("CECL_CREATE_COMMAND_QUEUE...\n\n"); 

    // Allocate the OpenCL source, intermediate and result buffer memory objects on the device GMEM
    cmDevBufIn = CECL_BUFFER(cxGPUContext, CL_MEM_READ_ONLY, szBuffBytes, NULL, &ciErrNum);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    cmDevBufTemp = CECL_BUFFER(cxGPUContext, CL_MEM_READ_WRITE, szBuffBytes, NULL, &ciErrNum);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    cmDevBufOut = CECL_BUFFER(cxGPUContext, CL_MEM_WRITE_ONLY, szBuffBytes, NULL, &ciErrNum);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    shrLog("CECL_BUFFER (Input, Intermediate and Output buffers, device GMEM)...\n"); 

    // Read the OpenCL kernel source in from file
    free(cPathAndName);
    cPathAndName = shrFindFilePath(clSourcefile, argv[0]);
    oclCheckErrorEX(cPathAndName != NULL, shrTRUE, pCleanup);
    cSourceCL = oclLoadProgSource(cPathAndName, "// My comment\n", &szKernelLength);
    oclCheckErrorEX(cSourceCL != NULL, shrTRUE, pCleanup);
    shrLog("oclLoadProgSource...\n"); 

    // Create the program 
    cpProgram = CECL_PROGRAM_WITH_SOURCE(cxGPUContext, 1, (const char **)&cSourceCL, &szKernelLength, &ciErrNum);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    shrLog("CECL_PROGRAM_WITH_SOURCE...\n"); 

    // Setup build options string 
    //--------------------------------
    // Add mad option 
    std::string sBuildOpts = " -cl-fast-relaxed-math"; 

    // Clamp to edge option
    #ifdef CLAMP_TO_EDGE 
        sBuildOpts  += " -D CLAMP_TO_EDGE";
    #endif

    // mac
    #ifdef MAC
        sBuildOpts  += " -DMAC";
    #endif

    // Build the program 
    ciErrNum = CECL_PROGRAM(cpProgram, 0, NULL, sBuildOpts.c_str(), NULL, NULL);
    if (ciErrNum != CL_SUCCESS)
    {
        // If build problem, write out standard ciErrNum, Build Log and PTX, then cleanup and exit
        shrLogEx(LOGBOTH | ERRORMSG, ciErrNum, STDERROR);
        oclLogBuildInfo(cpProgram, cdDevices[uiTargetDevice]);
        oclLogPtx(cpProgram, cdDevices[uiTargetDevice], "oclRecursiveGaussian.ptx");
        shrQAFinish(argc, (const char **)argv, QA_FAILED);
        Cleanup(EXIT_FAILURE);
    }
    shrLog("CECL_PROGRAM...\n"); 

    // Create kernels
    ckSimpleRecursiveRGBA = CECL_KERNEL(cpProgram, "SimpleRecursiveRGBA", &ciErrNum);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    ckRecursiveGaussianRGBA = CECL_KERNEL(cpProgram, "RecursiveGaussianRGBA", &ciErrNum);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    ckTranspose = CECL_KERNEL(cpProgram, "Transpose", &ciErrNum);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    shrLog("CECL_KERNEL (Rows, Columns, Transpose)...\n\n"); 

    // check/reset work group size
    size_t wgSize;
    ciErrNum = clGetKernelWorkGroupInfo(ckTranspose, cdDevices[uiTargetDevice], CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &wgSize, NULL);
    if (wgSize == 64)
    {
        iTransposeBlockDim = 8;
    }

    // Set unchanging local work sizes for gaussian kernels and transpose kernel
    szGaussLocalWork = iNumThreads;
    szTransposeLocalWork[0] = iTransposeBlockDim;
    szTransposeLocalWork[1] = iTransposeBlockDim;

    // init filter coefficients
    PreProcessGaussParms (fSigma, iOrder, &oclGP);

    // set common kernel args
    GPUGaussianSetCommonArgs (&oclGP);

    // init running timers
    shrDeltaT(0);   // timer 0 used for computation timing 
    shrDeltaT(1);   // timer 1 used for fps computation

    // Start main GLUT rendering loop for processing and rendering, 
	// or otherwise run No-GL Q/A test sequence
	if (!(bQATest))
	{
		glutMainLoop();
	}
	else 
	{
		TestNoGL();
	}

    shrQAFinish(argc, (const char **)argv, QA_PASSED);
    Cleanup(EXIT_SUCCESS);
}

// Function to set common kernel args that only change outside of GLUT loop
//*****************************************************************************
void GPUGaussianSetCommonArgs(GaussParms* pGP)
{
    // common Gaussian args
    #if USE_SIMPLE_FILTER
        // Set the Common Argument values for the simple Gaussian kernel
        ciErrNum |= CECL_SET_KERNEL_ARG(ckSimpleRecursiveRGBA, 4, sizeof(float), (void*)&pGP->ema);
        oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    #else
        // Set the Common Argument values for the Gaussian kernel
        ciErrNum |= CECL_SET_KERNEL_ARG(ckRecursiveGaussianRGBA, 4, sizeof(float), (void*)&pGP->a0);
        ciErrNum |= CECL_SET_KERNEL_ARG(ckRecursiveGaussianRGBA, 5, sizeof(float), (void*)&pGP->a1);
        ciErrNum |= CECL_SET_KERNEL_ARG(ckRecursiveGaussianRGBA, 6, sizeof(float), (void*)&pGP->a2);
        ciErrNum |= CECL_SET_KERNEL_ARG(ckRecursiveGaussianRGBA, 7, sizeof(float), (void*)&pGP->a3);
        ciErrNum |= CECL_SET_KERNEL_ARG(ckRecursiveGaussianRGBA, 8, sizeof(float), (void*)&pGP->b1);
        ciErrNum |= CECL_SET_KERNEL_ARG(ckRecursiveGaussianRGBA, 9, sizeof(float), (void*)&pGP->b2);
        ciErrNum |= CECL_SET_KERNEL_ARG(ckRecursiveGaussianRGBA, 10, sizeof(float), (void*)&pGP->coefp);
        ciErrNum |= CECL_SET_KERNEL_ARG(ckRecursiveGaussianRGBA, 11, sizeof(float), (void*)&pGP->coefn);
        oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

     #endif

    // Set common transpose Argument values 
    ciErrNum |= CECL_SET_KERNEL_ARG(ckTranspose, 4, sizeof(unsigned int) * iTransposeBlockDim * (iTransposeBlockDim + 1), NULL );
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
}

// 8-bit RGBA Gaussian filter for GPU on a 2D image using OpenCL
//*****************************************************************************
double GPUGaussianFilterRGBA(GaussParms* pGP)
{
    // var for kernel timing
    double dKernelTime = 0.0;

    // Copy input data from host to device
    ciErrNum = CECL_WRITE_BUFFER(cqCommandQueue, cmDevBufIn, CL_TRUE, 0, szBuffBytes, uiInput, 0, NULL, NULL);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    // sync host and start timer
    clFinish(cqCommandQueue);
    shrDeltaT(0);

    // Set Gaussian global work dimensions, then set variable args and process in 1st dimension
    szGaussGlobalWork = shrRoundUp((int)szGaussLocalWork, uiImageWidth); 
    #if USE_SIMPLE_FILTER
        // Set simple Gaussian kernel variable arg values
        ciErrNum = CECL_SET_KERNEL_ARG(ckSimpleRecursiveRGBA, 0, sizeof(cl_mem), (void*)&cmDevBufIn);
        ciErrNum |= CECL_SET_KERNEL_ARG(ckSimpleRecursiveRGBA, 1, sizeof(cl_mem), (void*)&cmDevBufTemp);
        ciErrNum |= CECL_SET_KERNEL_ARG(ckSimpleRecursiveRGBA, 2, sizeof(unsigned int), (void*)&uiImageWidth);
        ciErrNum |= CECL_SET_KERNEL_ARG(ckSimpleRecursiveRGBA, 3, sizeof(unsigned int), (void*)&uiImageHeight);
        oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

        // Launch simple Gaussian kernel on the data in one dimension
        ciErrNum = CECL_ND_RANGE_KERNEL(cqCommandQueue, ckSimpleRecursiveRGBA, 1, NULL, &szGaussGlobalWork, &szGaussLocalWork, 0, NULL, NULL);
        oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    #else
        // Set full Gaussian kernel variable arg values
        ciErrNum = CECL_SET_KERNEL_ARG(ckRecursiveGaussianRGBA, 0, sizeof(cl_mem), (void*)&cmDevBufIn);
        ciErrNum |= CECL_SET_KERNEL_ARG(ckRecursiveGaussianRGBA, 1, sizeof(cl_mem), (void*)&cmDevBufTemp);
        ciErrNum |= CECL_SET_KERNEL_ARG(ckRecursiveGaussianRGBA, 2, sizeof(unsigned int), (void*)&uiImageWidth);
        ciErrNum |= CECL_SET_KERNEL_ARG(ckRecursiveGaussianRGBA, 3, sizeof(unsigned int), (void*)&uiImageHeight);
        oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

        // Launch full Gaussian kernel on the data in one dimension
        ciErrNum = CECL_ND_RANGE_KERNEL(cqCommandQueue, ckRecursiveGaussianRGBA, 1, NULL, &szGaussGlobalWork, &szGaussLocalWork, 0, NULL, NULL);
        oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
     #endif

    // Set transpose global work dimensions and variable args 
    szTransposeGlobalWork[0] = shrRoundUp((int)szTransposeLocalWork[0], uiImageWidth); 
    szTransposeGlobalWork[1] = shrRoundUp((int)szTransposeLocalWork[1], uiImageHeight); 
    ciErrNum = CECL_SET_KERNEL_ARG(ckTranspose, 0, sizeof(cl_mem), (void*)&cmDevBufTemp);
    ciErrNum |= CECL_SET_KERNEL_ARG(ckTranspose, 1, sizeof(cl_mem), (void*)&cmDevBufOut);
    ciErrNum |= CECL_SET_KERNEL_ARG(ckTranspose, 2, sizeof(unsigned int), (void*)&uiImageWidth);
    ciErrNum |= CECL_SET_KERNEL_ARG(ckTranspose, 3, sizeof(unsigned int), (void*)&uiImageHeight);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    // Launch transpose kernel in 1st direction
    ciErrNum = CECL_ND_RANGE_KERNEL(cqCommandQueue, ckTranspose, 2, NULL, szTransposeGlobalWork, szTransposeLocalWork, 0, NULL, NULL);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    // Reset Gaussian global work dimensions and variable args, then process in 2nd dimension
    // note width and height parameters flipped due to transpose
    szGaussGlobalWork = shrRoundUp((int)szGaussLocalWork, uiImageHeight); 
    #if USE_SIMPLE_FILTER
        // set simple Gaussian kernel arg values
        ciErrNum = CECL_SET_KERNEL_ARG(ckSimpleRecursiveRGBA, 0, sizeof(cl_mem), (void*)&cmDevBufOut);
        ciErrNum |= CECL_SET_KERNEL_ARG(ckSimpleRecursiveRGBA, 1, sizeof(cl_mem), (void*)&cmDevBufTemp);
        ciErrNum |= CECL_SET_KERNEL_ARG(ckSimpleRecursiveRGBA, 2, sizeof(unsigned int), (void*)&uiImageHeight);
        ciErrNum |= CECL_SET_KERNEL_ARG(ckSimpleRecursiveRGBA, 3, sizeof(unsigned int), (void*)&uiImageWidth);
        oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

        // Launch simple Gaussian kernel on the data in the other dimension
        ciErrNum = CECL_ND_RANGE_KERNEL(cqCommandQueue, ckSimpleRecursiveRGBA, 1, NULL, &szGaussGlobalWork, &szGaussLocalWork, 0, NULL, NULL);
        oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    #else
        // Set full Gaussian kernel arg values
        ciErrNum = CECL_SET_KERNEL_ARG(ckRecursiveGaussianRGBA, 0, sizeof(cl_mem), (void*)&cmDevBufOut);
        ciErrNum |= CECL_SET_KERNEL_ARG(ckRecursiveGaussianRGBA, 1, sizeof(cl_mem), (void*)&cmDevBufTemp);
        ciErrNum |= CECL_SET_KERNEL_ARG(ckRecursiveGaussianRGBA, 2, sizeof(unsigned int), (void*)&uiImageHeight);
        ciErrNum |= CECL_SET_KERNEL_ARG(ckRecursiveGaussianRGBA, 3, sizeof(unsigned int), (void*)&uiImageWidth);
        oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
 
        // Launch full Gaussian kernel on the data in the other dimension
        ciErrNum = CECL_ND_RANGE_KERNEL(cqCommandQueue, ckRecursiveGaussianRGBA, 1, NULL, &szGaussGlobalWork, &szGaussLocalWork, 0, NULL, NULL);
        oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
     #endif

    // Reset transpose global work dimensions and variable args 
    // note width and height parameters flipped due to 1st transpose
    szTransposeGlobalWork[0] = shrRoundUp((int)szTransposeLocalWork[0], uiImageHeight); 
    szTransposeGlobalWork[1] = shrRoundUp((int)szTransposeLocalWork[1], uiImageWidth); 
    ciErrNum = CECL_SET_KERNEL_ARG(ckTranspose, 0, sizeof(cl_mem), (void*)&cmDevBufTemp);
    ciErrNum |= CECL_SET_KERNEL_ARG(ckTranspose, 1, sizeof(cl_mem), (void*)&cmDevBufOut);
    ciErrNum |= CECL_SET_KERNEL_ARG(ckTranspose, 2, sizeof(unsigned int), (void*)&uiImageHeight);
    ciErrNum |= CECL_SET_KERNEL_ARG(ckTranspose, 3, sizeof(unsigned int), (void*)&uiImageWidth);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    // Launch transpose kernel in 2nd direction
    ciErrNum = CECL_ND_RANGE_KERNEL(cqCommandQueue, ckTranspose, 2, NULL, szTransposeGlobalWork, szTransposeLocalWork, 0, NULL, NULL);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

   // sync host and stop timer
    clFinish(cqCommandQueue);
    dKernelTime = shrDeltaT(0);

    // Copy results back to host, block until complete
    ciErrNum = CECL_READ_BUFFER(cqCommandQueue, cmDevBufOut, CL_TRUE, 0, szBuffBytes, uiOutput, 0, NULL, NULL);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    return dKernelTime;
}

// Initialize GL
//*****************************************************************************
void InitGL(int* argc, char **argv)
{
    // init GLUT and GLUT window
    glutInit(argc, (char**)argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowPosition (glutGet(GLUT_SCREEN_WIDTH)/2 - iGraphicsWinWidth/2, 
                            glutGet(GLUT_SCREEN_HEIGHT)/2 - iGraphicsWinHeight/2);
    glutInitWindowSize(iGraphicsWinWidth, iGraphicsWinHeight);
    iGLUTWindowHandle = glutCreateWindow("OpenCL for GPU Recursive Gaussian");
#if !(defined (__APPLE__) || defined(MACOSX))
    glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS);
#endif

    // register GLUT callbacks
    glutKeyboardFunc(KeyboardGL);
    glutDisplayFunc(DisplayGL);
    glutReshapeFunc(Reshape);
    glutIdleFunc(Idle);
    glutTimerFunc(REFRESH_DELAY, timerEvent, 0);

    // create GLUT menu
    iGLUTMenuHandle = glutCreateMenu(MenuGL);
    glutAddMenuEntry("Toggle Filter On/Off <spacebar>", ' ');
    glutAddMenuEntry("Toggle Processing between GPU and CPU [p]", 'p');
    glutAddMenuEntry("Toggle between Full Screen and Windowed [f]", 'f');
    glutAddMenuEntry("Increment Sigma +1.0", '+');
    glutAddMenuEntry("Decrement Sigma -1.0", '_');
    glutAddMenuEntry("Increment Sigma +0.1", '=');
    glutAddMenuEntry("Decrement Sigma -0.1", '-');
    glutAddMenuEntry("Set Order 0 [0]", '0');
    glutAddMenuEntry("Set Order 1 [1]", '1');
    glutAddMenuEntry("Set Order 2 [2]", '2');
    glutAddMenuEntry("Quit <esc>", '\033');
    glutAttachMenu(GLUT_RIGHT_BUTTON);

    // Set clear color
    glClearColor(0.f, 0.f, 0.f, 0.f);

    // Zoom with fixed aspect ratio
    float fAspects[2] = {(float)glutGet(GLUT_WINDOW_WIDTH)/(float)uiImageWidth , (float)glutGet(GLUT_WINDOW_HEIGHT)/(float)uiImageHeight};
    fZoom = fAspects[0] > fAspects[1] ? fAspects[1] : fAspects[0];
    glPixelZoom(fZoom, fZoom);

    glewInit();

    // Disable vertical sync, if supported
    #ifdef _WIN32
        if (wglewIsSupported("WGL_EXT_swap_control")) 
        {
            iVsyncState = wglGetSwapIntervalEXT();
            wglSwapIntervalEXT(0);
        }
    #else
        #if defined (__APPLE__) || defined(MACOSX)
	        GLint VBL = 0;
	        CGLGetParameter(CGLGetCurrentContext(), kCGLCPSwapInterval, &iVsyncState); 
	        CGLSetParameter(CGLGetCurrentContext(), kCGLCPSwapInterval, &VBL); 
        #else
	        if(glxewIsSupported("GLX_SGI_swap_control"))
            {
	            glXSwapIntervalSGI(0);	 
	        }
	    #endif
    #endif
}

// De-initialize GL
//*****************************************************************************
void DeInitGL()
{
    // Restore startup Vsync state, if supported
    #ifdef _WIN32
        if (wglewIsSupported("WGL_EXT_swap_control")) 
        {
            wglSwapIntervalEXT(iVsyncState);
        }
    #else
        #if defined (__APPLE__) || defined(MACOSX)
            CGLSetParameter(CGLGetCurrentContext(), kCGLCPSwapInterval, &iVsyncState); 
        #endif
    #endif

}

// Display callback for GLUT main loop
//*****************************************************************************
void DisplayGL()
{
    glClear(GL_COLOR_BUFFER_BIT);

    // Run filter processing (if toggled on), then render 
    if (bFilter)
    {
        // process on the GPU or the Host, depending on user toggle/flag
        if (iProcFlag == 0)
        {
            dProcessingTime += GPUGaussianFilterRGBA(&oclGP);
        }
        else 
        {
            dProcessingTime += HostRecursiveGaussianRGBA(uiInput, uiTemp, uiOutput, uiImageWidth, uiImageHeight, &oclGP);
        }

        // Draw processed image
        glDrawPixels(uiImageWidth, uiImageHeight, GL_RGBA, GL_UNSIGNED_BYTE, uiOutput); 
    }
    else 
    {
        // Skip processing and draw the raw input image
        glDrawPixels(uiImageWidth, uiImageHeight, GL_RGBA, GL_UNSIGNED_BYTE, uiInput); 
    }

    //  Flip backbuffer to screen
    glutSwapBuffers();

    // Increment the frame counter, and do fps stuff if it's time
    if (iFrameCount++ > iFrameTrigger)
    {
        // Buffer for display window title
        char cTitle[256];

        // Get average fps and average computation time
        iFramesPerSec = (int)((double)iFrameCount / shrDeltaT(1));
        dProcessingTime /= (double)iFrameCount; 

#ifdef GPU_PROFILING
        if (bFilter)
        {
            if (!USE_SIMPLE_FILTER)
            {
                #ifdef _WIN32
                    sprintf_s(cTitle, 256, "%s Recursive Gaussian | W: %u  H: %u | Full Filter, Order %i, Sigma %.1f | %i fps | Proc. t = %.5f s | %.1f Mpix/s", 
                            cProcessor[iProcFlag], uiImageWidth, uiImageHeight, iOrder, fSigma, 
                            iFramesPerSec, dProcessingTime, (1.0e-6 * uiImageWidth * uiImageHeight)/dProcessingTime);  
                #else
                    sprintf(cTitle, "%s Recursive Gaussian | W: %u  H: %u | Full Filter, Order %i, Sigma %.1f | %i fps | Proc. t = %.5f s | %.1f Mpix/s", 
                            cProcessor[iProcFlag], uiImageWidth, uiImageHeight, iOrder, fSigma, 
                            iFramesPerSec, dProcessingTime, (1.0e-6 * uiImageWidth * uiImageHeight)/dProcessingTime);  
                #endif
            }
            else 
            {
                #ifdef _WIN32
                    sprintf_s(cTitle, 256, "%s Recursive Gaussian | W: %u  H: %u | Simple Filter, Sigma %.1f | %i fps | Proc. t = %.5f s | %.1f Mpix/s",  
                            cProcessor[iProcFlag], uiImageWidth, uiImageHeight, fSigma, 
                            iFramesPerSec, dProcessingTime, (1.0e-6 * uiImageWidth * uiImageHeight)/dProcessingTime);  
                #else
                        sprintf(cTitle, "%s Recursive Gaussian | W: %u  H: %u | Simple Filter, Sigma %.1f | %i fps | Proc. t = %.5f s | %.1f Mpix/s", 
                            cProcessor[iProcFlag], uiImageWidth, uiImageHeight, fSigma, 
                            iFramesPerSec, dProcessingTime, (1.0e-6 * uiImageWidth * uiImageHeight)/dProcessingTime);  
                #endif
            }
        }
        else 
        {
            #ifdef _WIN32
                sprintf_s(cTitle, 256, "Recursive Gaussian OFF | W: %u  H: %u | %i fps", 
	                        uiImageWidth, uiImageHeight, iFramesPerSec);  
            #else
                sprintf(cTitle, "Recursive Gaussian OFF | W: %u  H: %u | %i fps", 
	                        uiImageWidth, uiImageHeight, iFramesPerSec);  
            #endif
        }
#else
        if (bFilter)
        {
            if (!USE_SIMPLE_FILTER)
            {
                #ifdef _WIN32
                    sprintf_s(cTitle, 256, "%s Recursive Gaussian | W: %u  H: %u | Full Filter, Order %i, Sigma %.1f", 
                            cProcessor[iProcFlag], uiImageWidth, uiImageHeight, iOrder, fSigma);  
                #else
                    sprintf(cTitle, "%s Recursive Gaussian | W: %u  H: %u | Full Filter, Order %i, Sigma %.1f", 
                            cProcessor[iProcFlag], uiImageWidth, uiImageHeight, iOrder, fSigma);  
                #endif
            }
            else 
            {
                #ifdef _WIN32
                    sprintf_s(cTitle, 256, "%s Recursive Gaussian | W: %u  H: %u | Simple Filter, Sigma %.1f",  
                            cProcessor[iProcFlag], uiImageWidth, uiImageHeight, fSigma);  
                #else
                        sprintf(cTitle, "%s Recursive Gaussian | W: %u  H: %u | Simple Filter, Sigma %.1f", 
                            cProcessor[iProcFlag], uiImageWidth, uiImageHeight, fSigma);  
                #endif
            }
        }
        else 
        {
            #ifdef _WIN32
                sprintf_s(cTitle, 256, "Recursive Gaussian OFF | W: %u  H: %u", 
	                        uiImageWidth, uiImageHeight);  
            #else
                sprintf(cTitle, "Recursive Gaussian OFF | W: %u  H: %u", 
	                        uiImageWidth, uiImageHeight);  
            #endif
        }
#endif
        glutSetWindowTitle(cTitle);

        // Log fps and processing info to console and file 
        shrLog("%s\n", cTitle); 

        // if doing quick test, exit
        if ((bNoPrompt) && (!--iTestSets))
        {
            // Cleanup up and quit
            shrQAFinish2(false, *pArgc, (const char **)pArgv, QA_PASSED);
            Cleanup(EXIT_SUCCESS);
        }

        // reset the frame counter and processing timer val and adjust trigger
        iFrameCount = 0; 
        dProcessingTime = 0.0;
        iFrameTrigger = (iFramesPerSec > 1) ? iFramesPerSec * 2 : 1;
    }
}

// Window resize handler callback
//*****************************************************************************
void Reshape(int w, int h)
{
    // Zoom image
    glPixelZoom((float)w / uiImageWidth, (float)h / uiImageHeight);

    //float fAspects[2] = {(float)glutGet(GLUT_WINDOW_WIDTH)/(float)uiImageWidth , (float)glutGet(GLUT_WINDOW_HEIGHT)/(float)uiImageHeight};
    //fZoom = fAspects[0] > fAspects[1] ? fAspects[1] : fAspects[0];
    //glPixelZoom(fZoom, fZoom);
}


void timerEvent(int value)
{
    glutPostRedisplay();
	glutTimerFunc(REFRESH_DELAY, timerEvent,0);
}

// Keyboard event handler callback
//*****************************************************************************
void KeyboardGL(unsigned char key, int /*x*/, int /*y*/)
{
    switch(key) 
    {
        case 'P':   // P toggles Processing between CPU and GPU
        case 'p':   // p toggles Processing between CPU and GPU
            if (iProcFlag == 0)
            {
                iProcFlag = 1;
            }
            else 
            {
                iProcFlag = 0;
            }
            shrLog("\n%s Processing...\n", cProcessor[iProcFlag]);
            break;
        case 'F':   // F toggles main graphics display full screen
        case 'f':   // f toggles main graphics display full screen
            bFullScreen = !bFullScreen;
            if (bFullScreen)
            {
                iGraphicsWinPosX = glutGet(GLUT_WINDOW_X) - 8;
                iGraphicsWinPosY = glutGet(GLUT_WINDOW_Y) - 30;
                iGraphicsWinWidth  = min(glutGet(GLUT_WINDOW_WIDTH) , glutGet(GLUT_SCREEN_WIDTH) - 2*iGraphicsWinPosX ); 
                iGraphicsWinHeight = min(glutGet(GLUT_WINDOW_HEIGHT), glutGet(GLUT_SCREEN_HEIGHT)- 2*iGraphicsWinPosY ); 
                printf("(x,y)=(%d,%d), (w,h)=(%d,%d)\n", iGraphicsWinPosX, iGraphicsWinPosY, iGraphicsWinWidth, iGraphicsWinHeight);
                glutFullScreen();
            }
            else
            {
                glutPositionWindow(iGraphicsWinPosX, iGraphicsWinPosY);
                glutReshapeWindow(iGraphicsWinWidth, iGraphicsWinHeight);
            }
            shrLog("\nMain Graphics %s...\n", bFullScreen ? "FullScreen" : "Windowed");
            break;
        case ' ':   // space bar toggles filter on and off
            bFilter = !bFilter;
            shrLog("\nRecursive Gaussian Filter Toggled %s...\n", bFilter ? "ON" : "OFF");
            break;
        case '=':
            fSigma += 0.1f;
            break;
        case '-':
            fSigma -= 0.1f;
            break;
        case '+':
			fSigma += 1.0f;
			break;
		case '_':
			fSigma -= 1.0f;
			break;
        case '0':
            iOrder = 0;
            break;
        case '1':
            if (!USE_SIMPLE_FILTER)
            {
                iOrder = 1;
                fSigma = 2.0f;
            }
            break;
        case '2':
            if (!USE_SIMPLE_FILTER)
            {
                iOrder = 2;
                fSigma = 0.2f;
            }
            break;
        case '\033': // escape quits
        case '\015':// Enter quits    
        case 'Q':    // Q quits
        case 'q':    // q (or escape) quits
            // Cleanup up and quit
            bNoPrompt = shrTRUE;
            shrQAFinish2(false, *pArgc, (const char **)pArgv, QA_PASSED);
            Cleanup(EXIT_SUCCESS);
            break;
    }

    // range check order and sigma 
    iOrder = CLAMP (iOrder, 0, 2);
    fSigma = MAX (0.1f, fSigma);

    // pre-compute filter coefficients and set common kernel args
    PreProcessGaussParms (fSigma, iOrder, &oclGP);
    GPUGaussianSetCommonArgs (&oclGP);

    // Log filter params
    if (bFilter)
    {
        if (USE_SIMPLE_FILTER)
        {
            shrLog("Simple Filter, Sigma =  %.1f\n", fSigma);  
        }
        else 
        {
            shrLog("Full Filter, Order = %i, Sigma =  %.1f\n", iOrder, fSigma);  
        }
    }

    // Trigger fps update and call for refresh
    TriggerFPSUpdate();
}

// GLUT menu callback function
//*****************************************************************************
void MenuGL(int i)
{
    KeyboardGL((unsigned char) i, 0, 0);
}

// GL Idle time callback
//*****************************************************************************
void Idle(void)
{
}

// Helper to trigger reset of fps vars at transition 
//*****************************************************************************
void TriggerFPSUpdate()
{
    iFrameCount = 0; 
    iFramesPerSec = 1;
    iFrameTrigger = 2;
    shrDeltaT(1);
    shrDeltaT(0);
    dProcessingTime = 0.0;
}

// Run a test sequence without any GL 
//*****************************************************************************
void TestNoGL()
{
    // Warmup call to assure OpenCL driver is awake
    GPUGaussianFilterRGBA(&oclGP);
    clFinish(cqCommandQueue);

	// Start round-trip timer and process iCycles loops on the GPU
    const int iCycles = 150;
    dProcessingTime = 0.0;
    shrLog("\nRunning GPUGaussianFilterRGBA for %d cycles...\n\n", iCycles);
	shrDeltaT(2); 
    for (int i = 0; i < iCycles; i++)
    {
        dProcessingTime += GPUGaussianFilterRGBA(&oclGP);
    }

    // Get round-trip and average computation time
    double dRoundtripTime = shrDeltaT(2)/(double)iCycles;
    dProcessingTime /= (double)iCycles;

    // log testname, throughput, timing and config info to sample and master logs
    if (!USE_SIMPLE_FILTER)
    {
        shrLogEx(LOGBOTH | MASTER, 0, "oclRecursiveGaussian-full, Throughput = %.4f M RGBA Pixels/s, Time = %.5f s, Size = %u RGBA Pixels, NumDevsUsed = %u, Workgroup = %u\n", 
               (1.0e-6 * uiImageWidth * uiImageHeight)/dProcessingTime, dProcessingTime, (uiImageWidth * uiImageHeight), uiNumDevsUsed, szGaussLocalWork); 
    }
    else 
    {
        shrLogEx(LOGBOTH | MASTER, 0, "oclRecursiveGaussian-simple, Throughput = %.4f M RGBA Pixels/s, Time = %.5f s, Size = %u RGBA Pixels, NumDevsUsed = %u, Workgroup = %u\n", 
               (1.0e-6 * uiImageWidth * uiImageHeight)/dProcessingTime, dProcessingTime, (uiImageWidth * uiImageHeight), uiNumDevsUsed, szGaussLocalWork); 
    }
    shrLog("\nRoundTrip Time = %.5f s, Equivalent FPS = %.1f\n\n", dRoundtripTime, 1.0/dRoundtripTime);
    
    // Compute on host 
    cl_uint* uiGolden = (cl_uint*)malloc(szBuffBytes);
    HostRecursiveGaussianRGBA(uiInput, uiTemp, uiGolden, uiImageWidth, uiImageHeight, &oclGP);

    // Compare GPU and Host results:  Allow variance of 1 GV in up to 0.1% of pixels 
    shrLog("Comparing GPU Result to CPU Result...\n"); 
    shrBOOL bMatch = shrCompareuit(uiGolden, uiOutput, (uiImageWidth * uiImageHeight), 1.0f, 0.01f);
    shrLog("\nGPU Result %s CPU Result within tolerance...\n", (bMatch == shrTRUE) ? "matches" : "DOESN'T match"); 

    // Cleanup and exit
    free(uiGolden);

    shrQAFinish2(true, *pArgc, (const char **)pArgv, (bMatch == shrTRUE) ? QA_PASSED : QA_FAILED);
    Cleanup((bMatch == shrTRUE) ? EXIT_SUCCESS : EXIT_FAILURE);
}

// Function to print menu items
//*****************************************************************************
void ShowMenuItems()
{
    // Show help info in console / log
    // Show help info, start timers
    shrLog("  <Right Click Mouse Button> for Menu\n"); 
    shrLog("  or\n  Press:\n\n");
	shrLog("  <-/=> change Sigma (-/+ 0.1)\n");
	shrLog("  <_/+> change Sigma (-/+ 1.0)\n");
	shrLog("  <0,1,2> keys to set Order\n");
    shrLog("  <spacebar> to toggle Filter On/Off\n");
	shrLog("  <F> key to toggle FullScreen On/Off\n");
    shrLog("  <P> key to toggle Processing between GPU and CPU\n");
	shrLog("  <ESC> to Quit\n\n"); 
}

// Function to clean up and exit
//*****************************************************************************
void Cleanup(int iExitCode)
{
    // Cleanup allocated objects
    shrLog("\nStarting Cleanup...\n\n");
    if(cSourceCL)free(cSourceCL);
    if(cPathAndName)free(cPathAndName);
    if(uiInput)free(uiInput);
    if(uiTemp)free(uiTemp);
    if(uiOutput)free(uiOutput);
    if(ckSimpleRecursiveRGBA)clReleaseKernel(ckSimpleRecursiveRGBA);
    if(ckTranspose)clReleaseKernel(ckTranspose);
    if(ckRecursiveGaussianRGBA)clReleaseKernel(ckRecursiveGaussianRGBA);
    if(cpProgram)clReleaseProgram(cpProgram);
    if(cmDevBufIn)clReleaseMemObject(cmDevBufIn); 
    if(cmDevBufTemp)clReleaseMemObject(cmDevBufTemp); 
    if(cmDevBufOut)clReleaseMemObject(cmDevBufOut); 
    if(cqCommandQueue)clReleaseCommandQueue(cqCommandQueue);
    if(cxGPUContext)clReleaseContext(cxGPUContext);
    if(cdDevices)free(cdDevices);

    // Cleanup GL objects if used
    if (!bQATest)
    {
        DeInitGL();
    }

    // finalize logs and leave
    if (bNoPrompt || bQATest)
    {
        shrLogEx(LOGBOTH | CLOSELOG, 0, "%s Exiting...\n", cExecutableName);
    }
    else 
    {
        shrLogEx(LOGBOTH | CLOSELOG, 0, "%s Exiting...\nPress <Enter> to Quit\n", cExecutableName);
        #ifdef WIN32
            getchar();
        #endif
    }
    exit (iExitCode);
}
