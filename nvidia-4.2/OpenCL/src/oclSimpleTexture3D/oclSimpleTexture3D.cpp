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

/*
    3D texture sample

    This sample loads a 3D volume from disk and displays slices through it
    using 3D texture lookups.
*/

// Includes
//*****************************************************************************
// GLEW and GLUT includes
#include <GL/glew.h>
#if defined (__APPLE__) || defined(MACOSX)
    #include <GLUT/glut.h>
#else
    #include <GL/freeglut.h>
#endif

// Utilities, OpenCL and system includes
#include <oclUtils.h>
#include <shrQATest.h>

// Constants, defines, typedefs and global declarations
//*****************************************************************************

// Uncomment this #define to enable CL/GL Interop
//#define GL_INTEROP    

#define REFRESH_DELAY	  10 //ms

#define MAX_EPSILON_ERROR 5.0f
#define THRESHOLD         0.15f

const char *sSDKsample = "oclSimpleTexture3D";

int *pArgc = NULL;
char **pArgv = NULL;

typedef unsigned int uint;
typedef unsigned char uchar;

const char *volumeFilename = "Bucky.raw";
const size_t volumeSize[] = {32, 32, 32};
const uint width = 512, height = 512;
const size_t localWorkSize[] = {16, 16};
const size_t globalWorkSize[] = {width, height};

// OpenCL vars
cl_platform_id cpPlatform;
cl_uint uiNumDevices;
cl_device_id* cdDevices;
cl_context cxGPUContext;
cl_device_id cdDevice;
cl_command_queue cqCommandQueue;
cl_program cpProgram;
cl_kernel ckKernel;
cl_int ciErrNum;
cl_mem pbo_cl;
cl_mem d_volume;
cl_sampler volumeSamplerLinear;
cl_sampler volumeSamplerNearest;
char* cPathAndName = NULL;          // var for full paths to data, src, etc.
char* cSourceCL = NULL;             // Buffer to hold source for compilation 
const char* cExecutableName = NULL;

// Sim app config parameters
int iFrameCount = 0;                // FPS count for averaging
int iFrameTrigger = 90;             // FPS trigger for sampling
int iFramesPerSec = 0;              // frames per second
int iTestSets = 3;
float w = 0.5;                      // initial texture coordinate in z
int g_Index = 0;
shrBOOL bQATest = shrFALSE;
shrBOOL bNoPrompt = shrFALSE;
bool linearFiltering = true;
bool animate = true;

int iGLUTWindowHandle;              // handle to the GLUT window
GLuint pbo;                         // OpenGL pixel buffer object

// Forward Function declarations
//*****************************************************************************

// OpenCL functions
cl_mem initTexture3D(uchar *h_volume, const size_t volumeSize[3]);
void loadVolumeData(const char *exec_path);
void render();
cl_mem initTexture3D(uchar *h_volume, const size_t volumeSize[3]);

// OpenGL functionality
void InitGL(int* argc, char** argv);
void DisplayGL();
void Reshape(int w, int h);
void Idle(void);
void KeyboardGL(unsigned char key, int x, int y);
void timerEvent(int value);
void initGLBuffers();

// Helpers
void Cleanup(int iExitCode);
void (*pCleanup)(int) = &Cleanup;
void TestNoGL();

// Main program
//*****************************************************************************
int main(int argc, char** argv) 
{
	pArgc = &argc;
	pArgv = argv;

    shrQAStart(argc, argv);

    // start logs
	cExecutableName = argv[0];
    shrSetLogFileName ("oclSimpleTexture3D.txt");
    shrLog("%s Starting...\n\n", argv[0]); 
    shrLog(" Press <spacebar> to toggle animation ON/OFF\n");
    shrLog(" Press '+' and '-' to change displayed slice\n");
    shrLog(" Press <f> to toggle filtering ON/OFF\n\n");

    // process command line arguments
    if (argc > 1) 
    {
        bQATest   = shrCheckCmdLineFlag(argc, (const char**)argv, "qatest");
        bNoPrompt = shrCheckCmdLineFlag(argc, (const char**)argv, "noprompt");
    }

    // Initialize OpenGL context, so we can properly set the GL for CL.
    if(!bQATest)
    {
        InitGL(&argc, argv); 
    }

    //Get the NVIDIA platform
    ciErrNum = oclGetPlatformID(&cpPlatform);
    oclCheckError(ciErrNum, CL_SUCCESS);

    //Get the devices
    ciErrNum = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_CPU, 0, NULL, &uiNumDevices);
    oclCheckError(ciErrNum, CL_SUCCESS);
    cdDevices = (cl_device_id *)malloc(uiNumDevices * sizeof(cl_device_id) );
    ciErrNum = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_CPU, uiNumDevices, cdDevices, NULL);
    oclCheckError(ciErrNum, CL_SUCCESS);

    //Create the context
    cxGPUContext = clCreateContext(0, uiNumDevices, cdDevices, NULL, NULL, &ciErrNum);
    oclCheckError(ciErrNum, CL_SUCCESS);

    // get and log device info
    if(shrCheckCmdLineFlag(argc, (const char**)argv, "device")) 
    {
      int device_nr = 0;
      shrGetCmdLineArgumenti(argc, (const char**)argv, "device", &device_nr);
	  if( (unsigned int)device_nr < uiNumDevices ) {
        cdDevice = oclGetDev(cxGPUContext, device_nr);
	  } else {
		shrLog("Invalid Device %d Requested.\n", device_nr);
		shrExitEX(argc, (const char **)argv, EXIT_FAILURE);
	  }
    } 
    else 
    {
      cdDevice = oclGetMaxFlopsDev(cxGPUContext);
    }
    oclPrintDevInfo(LOGBOTH, cdDevice);

    // create a command-queue
    cqCommandQueue = CECL_CREATE_COMMAND_QUEUE(cxGPUContext, cdDevice, 0, &ciErrNum);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    // Program Setup
    size_t program_length;
    cPathAndName = shrFindFilePath("oclSimpleTexture3D_kernel.cl", argv[0]);
    oclCheckErrorEX(cPathAndName != NULL, shrTRUE, pCleanup);
    cSourceCL = oclLoadProgSource(cPathAndName, "", &program_length);
    oclCheckErrorEX(cSourceCL != NULL, shrTRUE, pCleanup);

    // create the program
    cpProgram = CECL_PROGRAM_WITH_SOURCE(cxGPUContext, 1,
					  (const char **) &cSourceCL , &program_length, &ciErrNum);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    
    // build the program
    ciErrNum = CECL_PROGRAM(cpProgram, 0, NULL, "-cl-fast-relaxed-math", NULL, NULL);
    if (ciErrNum != CL_SUCCESS)
    {
        // write out standard error, Build Log and PTX, then cleanup and exit
        shrLogEx(LOGBOTH | ERRORMSG, ciErrNum, STDERROR);
        oclLogBuildInfo(cpProgram, oclGetFirstDev(cxGPUContext));
        oclLogPtx(cpProgram, oclGetFirstDev(cxGPUContext), "oclSimpleTexture3D.ptx");
        Cleanup(EXIT_FAILURE); 
    }

    // create the kernel
    ckKernel = CECL_KERNEL(cpProgram, "render", &ciErrNum);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    // Load the data
    loadVolumeData(argv[0]);

    // Create buffers and textures, 
    // and then start main GLUT rendering loop for processing and rendering, 
	// or otherwise run No-GL Q/A test sequence
    if (!bQATest)
    {
        // OpenGL buffers
        initGLBuffers();

        // init timer 1 for fps measurement 
        shrDeltaT(1);    

        // start rendering mainloop
        glutMainLoop();
    } 
    else 
    {
	    TestNoGL();
    }
    
    Cleanup(EXIT_SUCCESS);
}

// Initialize GL
//*****************************************************************************
void InitGL(int* argc, char **argv )
{
    // init GLUT 
    glutInit(argc, (char**)argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
    glutInitWindowPosition (glutGet(GLUT_SCREEN_WIDTH)/2 - width/2, 
                            glutGet(GLUT_SCREEN_HEIGHT)/2 - height/2);
    glutInitWindowSize(width, height);
    iGLUTWindowHandle = glutCreateWindow("OpenCL 3D texture");
#if !(defined (__APPLE__) || defined(MACOSX))
    glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS);
#endif

    // register GLUT callback functions
    glutDisplayFunc(DisplayGL);
    glutKeyboardFunc(KeyboardGL);
    glutReshapeFunc(Reshape);
    glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
    glutIdleFunc(Idle);

    // init GLEW
    glewInit();
    GLboolean bGLEW = glewIsSupported("GL_VERSION_2_0 GL_ARB_pixel_buffer_object"); 
    shrCheckErrorEX(bGLEW, shrTRUE, pCleanup);
}

// render image using OpenCL    
//*****************************************************************************
void render()
{
    ciErrNum = CL_SUCCESS;

    // Transfer ownership of buffer from GL to CL
#ifdef GL_INTEROP
    // Acquire PBO for OpenCL writing
    ciErrNum |= clEnqueueAcquireGLObjects(cqCommandQueue, 1, &pbo_cl, 0,0,0);
#endif    

    // set kernel argumanets 
    ciErrNum |= CECL_SET_KERNEL_ARG(ckKernel, 5, sizeof(float), &w);

    // execute OpenCL kernel, writing results to PBO
    ciErrNum |= CECL_ND_RANGE_KERNEL(cqCommandQueue, ckKernel, 2, NULL, globalWorkSize, localWorkSize, 0, 0, 0);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

#ifdef GL_INTEROP
    // Transfer ownership of buffer back from CL to GL    
    ciErrNum |= clEnqueueReleaseGLObjects(cqCommandQueue, 1, &pbo_cl, 0, 0, 0);
#else

    // Explicit Copy 
    // map the PBO to copy data from the CL buffer via host
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);    

    // map the buffer object into client's memory
    GLubyte* ptr = (GLubyte*)glMapBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB,
                                        GL_WRITE_ONLY_ARB);
    ciErrNum |= CECL_READ_BUFFER(cqCommandQueue, pbo_cl, CL_TRUE, 0, sizeof(unsigned int) * height * width, ptr, 0, NULL, NULL);        
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    glUnmapBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB); 
#endif
}

// Display callback for GLUT main loop
//*****************************************************************************
void DisplayGL()
{
    // start timer 0 if it's update time
    double dProcessingTime = 0.0;
    if (iFrameCount >= iFrameTrigger)
    {
        shrDeltaT(0); 
    }

    // run OpenCL kernel 
    render();

    // get processing time from timer 0, if it's update time
    if (iFrameCount >= iFrameTrigger)
    {
        dProcessingTime = shrDeltaT(0); 
    }

    // display results
    glClear(GL_COLOR_BUFFER_BIT);
    glDisable(GL_DEPTH_TEST);
    glRasterPos2i(0, 0);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
    glDrawPixels(width, height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

    // flip backbuffer to screen
    glutSwapBuffers();
    glutReportErrors();

    // Increment the frame counter, and do fps and Q/A stuff if it's time
    if (iFrameCount++ > iFrameTrigger) 
    {
        // set GLUT Window Title
        char cTitle[256];
        iFramesPerSec = (int)((double)iFrameCount/shrDeltaT(1));
#ifdef GPU_PROFILING
        #ifdef _WIN32
            sprintf_s(cTitle, 256, "%s | %i fps | Proc. t = %.4f s", sSDKsample, iFramesPerSec, dProcessingTime);
        #else 
            sprintf(cTitle, "%s | %i fps | Proc. t = %.4f s", sSDKsample, iFramesPerSec, dProcessingTime);
        #endif
#else 
        #ifdef _WIN32
            sprintf_s(cTitle, 256, "%s", sSDKsample);
        #else 
            sprintf(cTitle, "%s", sSDKsample);
        #endif
#endif
       glutSetWindowTitle(cTitle); 

        // Log fps and processing info to console and file 
       shrLog(" %s\n", cTitle);  
       
        // Cleanup and leave if --noprompt mode and counter is up
        iTestSets--;
        if (bNoPrompt && (!iTestSets)) 
        {
            Cleanup(EXIT_SUCCESS);
        }

        // reset the frame counter and adjust trigger
        iFrameCount = 0; 
        iFrameTrigger = (iFramesPerSec > 1) ? iFramesPerSec * 2 : 1;
    }
}

//*****************************************************************************
void Idle()
{
    if (animate) {
        w += 0.01f;
		if( w > 1.0f )
			w = 0.0f;
    }
}

// Window resize handler callback
//*****************************************************************************
void Reshape(int x, int y)
{
    glViewport(0, 0, x, y);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0); 
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
    switch(key) {
        case '=':
        case '+':
            w += 0.01f;
            break;
        case '-':
        case '_':
            w -= 0.01f;
            break;
        case 'F':
        case 'f':
            linearFiltering = !linearFiltering;
            ciErrNum = CECL_SET_KERNEL_ARG(ckKernel, 1, sizeof(cl_sampler), linearFiltering ? &volumeSamplerLinear : &volumeSamplerNearest);
            shrLog("\nLinear Filtering Toggled %s...\n", linearFiltering ? "ON" : "OFF");
            oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
            break;
        case ' ':
            animate = !animate;
            break;
        case '\033': // escape quits
        case '\015':// Enter quits    
        case 'Q':    // Q quits
        case 'q':    // q (or escape) quits
            // Cleanup up and quit
            bNoPrompt = shrTRUE;
            Cleanup(EXIT_SUCCESS);
            break;
    }
}

//*****************************************************************************
void initGLBuffers()
{
    // create pixel buffer object
    glGenBuffersARB(1, &pbo);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
    glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, width*height*sizeof(GLubyte)*4, 0, GL_STREAM_DRAW_ARB);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

#ifdef GL_INTEROP
    // create OpenCL buffer from GL PBO
    pbo_cl = clCreateFromGLBuffer(cxGPUContext,CL_MEM_WRITE_ONLY, pbo, &ciErrNum);
#else            
    pbo_cl = CECL_BUFFER(cxGPUContext, CL_MEM_WRITE_ONLY,  width*height*sizeof(GLubyte)*4, NULL, &ciErrNum);
#endif

    ciErrNum |= CECL_SET_KERNEL_ARG(ckKernel, 2, sizeof(cl_mem), (void *) &pbo_cl);
    ciErrNum |= CECL_SET_KERNEL_ARG(ckKernel, 3, sizeof(unsigned int), &width);
    ciErrNum |= CECL_SET_KERNEL_ARG(ckKernel, 4, sizeof(unsigned int), &height);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
}

//*****************************************************************************
cl_mem initTexture3D(uchar *h_volume, const size_t volumeSize[3])
{
    cl_int ciErrNum;

    // create 3D array and copy data to device
	cl_image_format volume_format;
    volume_format.image_channel_order = CL_RGBA;
    volume_format.image_channel_data_type = CL_UNORM_INT8;
    uchar* h_tempVolume =(uchar*)malloc(volumeSize[0] * volumeSize[1] * volumeSize[2] * 4);
    for(unsigned int i = 0; i<(volumeSize[0] * volumeSize[1] * volumeSize[2]); i++)
    {
        h_tempVolume[4 * i] = h_volume[i];
    }
    cl_mem d_volume = clCreateImage3D(cxGPUContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, &volume_format, 
                                        volumeSize[0], volumeSize[1], volumeSize[2],
                                        (volumeSize[0] * 4), (volumeSize[0] * volumeSize[1] * 4),
                                        h_tempVolume, &ciErrNum);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    free (h_tempVolume);
    return d_volume;
}

//*****************************************************************************
void loadVolumeData(const char *exec_path)
{
    // load volume data
    if(cPathAndName)free(cPathAndName);
    cPathAndName = shrFindFilePath(volumeFilename, exec_path);
    oclCheckErrorEX(cPathAndName != NULL, shrTRUE, pCleanup);
    size_t size = volumeSize[0]*volumeSize[1]*volumeSize[2];
    uchar* h_volume = shrLoadRawFile(cPathAndName, size);
    oclCheckErrorEX(h_volume != NULL, true, pCleanup);
    shrLog(" Raw file data loaded...\n\n");

    // setup 3D image
    d_volume = initTexture3D(h_volume, volumeSize);
    ciErrNum = CECL_SET_KERNEL_ARG(ckKernel, 0, sizeof(cl_mem), &d_volume);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    // Create samplers for linear and nearest interpolation
    volumeSamplerLinear = clCreateSampler(cxGPUContext, true, CL_ADDRESS_REPEAT, CL_FILTER_LINEAR, &ciErrNum);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    volumeSamplerNearest = clCreateSampler(cxGPUContext, true, CL_ADDRESS_REPEAT, CL_FILTER_NEAREST, &ciErrNum);
    ciErrNum |= CECL_SET_KERNEL_ARG(ckKernel, 1, sizeof(cl_sampler), &volumeSamplerLinear);        
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    free(h_volume);
}

// Run a test sequence without any GL 
//*****************************************************************************
void TestNoGL()
{
    // execute OpenCL kernel without GL interaction

    pbo_cl = CECL_BUFFER(cxGPUContext, CL_MEM_WRITE_ONLY, width * height * sizeof(GLubyte) * 4, NULL, &ciErrNum);
    ciErrNum |= CECL_SET_KERNEL_ARG(ckKernel, 2, sizeof(cl_mem), (void *)&pbo_cl);
    ciErrNum |= CECL_SET_KERNEL_ARG(ckKernel, 3, sizeof(unsigned int), &width);
    ciErrNum |= CECL_SET_KERNEL_ARG(ckKernel, 4, sizeof(unsigned int), &height);   
    ciErrNum |= CECL_SET_KERNEL_ARG(ckKernel, 5, sizeof(float), &w);

    // warm up
    int iCycles = 20;
    for (int i = 0; i < iCycles; i++)
    {
        ciErrNum |= CECL_ND_RANGE_KERNEL(cqCommandQueue, ckKernel, 2, NULL, globalWorkSize, localWorkSize, 0,0,0 );
        oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);	
    }
    clFinish(cqCommandQueue);
    
	// Start timer 0 and process n loops on the GPU 
	shrDeltaT(0); 
    for (int i = 0; i < iCycles; i++)
    {
        ciErrNum |= CECL_ND_RANGE_KERNEL(cqCommandQueue, ckKernel, 2, NULL, globalWorkSize, localWorkSize, 0,0,0 );
        oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);	
    }
    clFinish(cqCommandQueue);
    
    // Get elapsed time and throughput, then log to sample and master logs
    double dAvgTime = shrDeltaT(0)/(double)iCycles;
    shrLogEx(LOGBOTH | MASTER, 0, "oclSimpleTexture3D, Throughput = %.4f, Time = %.5f, Size = %u, NumDevsUsed = %u, Workgroup = %u\n", 
           (1.0e-6 * width * height)/dAvgTime, dAvgTime, (width * height), 1, (localWorkSize[0] * localWorkSize[1])); 

    // Cleanup and exit
    Cleanup(EXIT_SUCCESS);
}

// Helper to clean up
//*****************************************************************************
void Cleanup(int iExitCode)
{
    // Cleanup allocated objects
    shrLog( "\nStarting Cleanup...\n\n");
    if(cPathAndName)free(cPathAndName);
    if(cSourceCL)free(cSourceCL);
	if(ckKernel)clReleaseKernel(ckKernel); 
    if(cpProgram)clReleaseProgram(cpProgram);
    if(cqCommandQueue)clReleaseCommandQueue(cqCommandQueue);
    if(cxGPUContext)clReleaseContext(cxGPUContext);
    if(volumeSamplerLinear)clReleaseSampler(volumeSamplerLinear);
    if(volumeSamplerNearest)clReleaseSampler(volumeSamplerNearest);
    if(pbo_cl)clReleaseMemObject(pbo_cl);
    if(!bQATest)glDeleteBuffersARB(1, &pbo);
    shrQAFinish2(bQATest, *pArgc, (const char **)pArgv, (iExitCode == 0) ? QA_PASSED : QA_FAILED); 

    // finalize logs and leave
    if (bQATest || bNoPrompt)
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
    exit(iExitCode);
}
