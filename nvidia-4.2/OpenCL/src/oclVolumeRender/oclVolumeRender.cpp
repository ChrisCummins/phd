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
    Volume rendering sample

    This sample loads a 3D volume from disk and displays it using
    ray marching and 3D textures.
*/

// OpenGL Graphics Includes
#include <GL/glew.h>
#if defined(__APPLE__) || defined(__MACOSX)
    #include <OpenGL/OpenGL.h>
    #include <GLUT/glut.h>
#else
    #include <GL/freeglut.h>
    #ifdef UNIX
       #include <GL/glx.h>
    #endif
#endif

// Includes
#include <iostream>
#include <string>

// Utilities, OpenCL and system includes
#include <oclUtils.h>
#include <shrQATest.h>

#if defined (__APPLE__) || defined(MACOSX)
   #define GL_SHARING_EXTENSION "cl_APPLE_gl_sharing"
#else
   #define GL_SHARING_EXTENSION "cl_khr_gl_sharing"
#endif

// Constants, defines, typedefs and global declarations
//*****************************************************************************
#define REFRESH_DELAY	  10 //ms

int *pArgc = NULL;
char **pArgv = NULL;

typedef unsigned int uint;
typedef unsigned char uchar;

const char *volumeFilename = "Bucky.raw";
size_t volumeSize[3] = {32, 32, 32};

uint width = 512, height = 512;
size_t gridSize[2] = {width, height};

#define LOCAL_SIZE_X 16
#define LOCAL_SIZE_Y 16

float viewRotation[3];
float viewTranslation[3] = {0.0, 0.0, -4.0f};
float invViewMatrix[12];

float density = 0.05f;
float brightness = 1.0f;
float transferOffset = 0.0f;
float transferScale = 1.0f;
bool linearFiltering = true;

GLuint pbo = 0;                 // OpenGL pixel buffer object
int iGLUTWindowHandle;          // handle to the GLUT window

// OpenCL vars
cl_platform_id cpPlatform;
cl_uint uiNumDevices;
cl_device_id* cdDevices;
cl_uint uiDeviceUsed;
cl_uint uiDevCount;
cl_context cxGPUContext;
cl_device_id device;
cl_command_queue cqCommandQueue;
cl_program cpProgram;
cl_kernel ckKernel;
cl_int ciErrNum;
cl_mem pbo_cl;
cl_mem d_volumeArray;
cl_mem d_transferFuncArray;
cl_mem d_invViewMatrix;
char* cPathAndName = NULL;          // var for full paths to data, src, etc.
char* cSourceCL;                    // Buffer to hold source for compilation 
const char* cExecutableName = NULL;
cl_bool g_bImageSupport;
cl_sampler volumeSamplerLinear;
cl_sampler volumeSamplerNearest;
cl_sampler transferFuncSampler;
cl_bool g_glInterop = false;

int iFrameCount = 0;                // FPS count for averaging
int iFrameTrigger = 20;             // FPS trigger for sampling
int iFramesPerSec = 0;              // frames per second
int iTestSets = 3;
int g_Index = 0;
shrBOOL bNoPrompt = shrFALSE;		// false = normal GL loop, true = Finite period of GL loop (a few seconds)
shrBOOL bQATest = shrFALSE;			// false = normal GL loop, true = run No-GL test sequence  
bool g_bFBODisplay = false;
int ox, oy;                         // mouse location vars
int buttonState = 0;                

// Forward Function declarations
//*****************************************************************************
// OpenCL Functions
void initPixelBuffer();
void render();
void createCLContext(int argc, const char** argv);
void initCLVolume(uchar *h_volume);

// OpenGL functionality
void InitGL(int* argc, char** argv);
void DisplayGL();
void Reshape(int w, int h);
void Idle(void);
void KeyboardGL(unsigned char key, int x, int y);
void timerEvent(int value);
void motion(int x, int y);
void mouse(int button, int state, int x, int y);

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
    shrSetLogFileName ("oclVolumeRender.txt");
    shrLog("%s Starting...\n\n", argv[0]); 
    shrLog("Press <f> to toggle filtering ON/OFF\n"
		                 "       '-' and '+' to change density (0.01 increments)\n"
                         "       ']' and '[' to change brightness\n"
                         "       ';' and ''' to modify transfer function offset\n"
                         "       '.' and ',' to modify transfer function scale\n\n");

    // get command line arg for quick test, if provided
    // process command line arguments
    if (argc > 1) 
    {
        bQATest = shrCheckCmdLineFlag(argc, (const char**)argv, "qatest");
        bNoPrompt = shrCheckCmdLineFlag(argc, (const char**)argv, "noprompt");
    }

    // First initialize OpenGL context, so we can properly setup the OpenGL / OpenCL interop.
    if(!bQATest) 
    {
        InitGL(&argc, argv); 
    }

    // Create OpenCL context, get device info, select device, select options for image/texture and CL-GL interop
    createCLContext(argc, (const char**)argv);

    // Print device info
    clGetDeviceInfo(cdDevices[uiDeviceUsed], CL_DEVICE_IMAGE_SUPPORT, sizeof(g_bImageSupport), &g_bImageSupport, NULL);
    shrLog("%s...\n\n", g_bImageSupport ? "Using Image (Texture)" : "No Image (Texuture) Support");      
    shrLog("Detailed Device info:\n\n");
    oclPrintDevInfo(LOGBOTH, cdDevices[uiDeviceUsed]);

    // create a command-queue
    cqCommandQueue = clCreateCommandQueue(cxGPUContext, cdDevices[uiDeviceUsed], 0, &ciErrNum);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    // Program Setup
    size_t program_length;
    cPathAndName = shrFindFilePath("volumeRender.cl", argv[0]);
    oclCheckErrorEX(cPathAndName != NULL, shrTRUE, pCleanup);
    cSourceCL = oclLoadProgSource(cPathAndName, "", &program_length);
    oclCheckErrorEX(cSourceCL != NULL, shrTRUE, pCleanup);

    // create the program
    cpProgram = clCreateProgramWithSource(cxGPUContext, 1,
					  (const char **)&cSourceCL, &program_length, &ciErrNum);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    
    // build the program
    std::string buildOpts = "-cl-fast-relaxed-math";
    buildOpts += g_bImageSupport ? " -DIMAGE_SUPPORT" : "";
    ciErrNum = clBuildProgram(cpProgram, 0, NULL, buildOpts.c_str(), NULL, NULL);
    if (ciErrNum != CL_SUCCESS)
    {
        // write out standard error, Build Log and PTX, then cleanup and return error
        shrLogEx(LOGBOTH | ERRORMSG, ciErrNum, STDERROR);
        oclLogBuildInfo(cpProgram, oclGetFirstDev(cxGPUContext));
        oclLogPtx(cpProgram, oclGetFirstDev(cxGPUContext), "oclVolumeRender.ptx");
        Cleanup(EXIT_FAILURE); 
    }

    // create the kernel
    ckKernel = clCreateKernel(cpProgram, "d_render", &ciErrNum);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    // parse arguments
    char *filename;
    if (shrGetCmdLineArgumentstr(argc, (const char**)argv, "file", &filename)) {
        volumeFilename = filename;
    }
    int n;
    if (shrGetCmdLineArgumenti(argc, (const char**)argv, "size", &n)) {
        volumeSize[0] = volumeSize[1] = volumeSize[2] = n;
    }
    if (shrGetCmdLineArgumenti(argc, (const char**)argv, "xsize", &n)) {
        volumeSize[0] = n;
    }
    if (shrGetCmdLineArgumenti(argc, (const char**)argv, "ysize", &n)) {
        volumeSize[1] = n;
    }
    if (shrGetCmdLineArgumenti(argc, (const char**)argv, "zsize", &n)) {
         volumeSize[2] = n;
    }

    // load volume data
    free(cPathAndName);
    cPathAndName = shrFindFilePath(volumeFilename, argv[0]);
    oclCheckErrorEX(cPathAndName != NULL, shrTRUE, pCleanup);
    size_t size = volumeSize[0] * volumeSize[1] * volumeSize[2];
    uchar* h_volume = shrLoadRawFile(cPathAndName, size);
    oclCheckErrorEX(h_volume != NULL, true, pCleanup);
    shrLog(" Raw file data loaded...\n\n");

    // Init OpenCL
    initCLVolume(h_volume);
    free (h_volume);

    // init timer 1 for fps measurement 
    shrDeltaT(1);  
    
    // Create buffers and textures, 
    // and then start main GLUT rendering loop for processing and rendering, 
	// or otherwise run No-GL Q/A test sequence
    if(!bQATest) 
    {
        initPixelBuffer();
        glutMainLoop();
    } 
    else 
    {
        TestNoGL();
    }

    // Normally unused return path
    Cleanup(EXIT_SUCCESS);
}

// render image using OpenCL
//*****************************************************************************
void render()
{
    ciErrNum = CL_SUCCESS;

    // Transfer ownership of buffer from GL to CL

	if( g_glInterop ) {
		// Acquire PBO for OpenCL writing
		glFlush();
		ciErrNum |= clEnqueueAcquireGLObjects(cqCommandQueue, 1, &pbo_cl, 0, 0, 0);
	}

	ciErrNum |= clEnqueueWriteBuffer(cqCommandQueue,d_invViewMatrix,CL_FALSE, 0,12*sizeof(float), invViewMatrix, 0, 0, 0);	

    // execute OpenCL kernel, writing results to PBO
    size_t localSize[] = {LOCAL_SIZE_X,LOCAL_SIZE_Y};

    ciErrNum |= clSetKernelArg(ckKernel, 3, sizeof(float), &density);
    ciErrNum |= clSetKernelArg(ckKernel, 4, sizeof(float), &brightness);
    ciErrNum |= clSetKernelArg(ckKernel, 5, sizeof(float), &transferOffset);
    ciErrNum |= clSetKernelArg(ckKernel, 6, sizeof(float), &transferScale);
    ciErrNum |= clEnqueueNDRangeKernel(cqCommandQueue, ckKernel, 2, NULL, gridSize, localSize, 0, 0, 0);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

	if( g_glInterop ) {
		// Transfer ownership of buffer back from CL to GL    
		ciErrNum |= clEnqueueReleaseGLObjects(cqCommandQueue, 1, &pbo_cl, 0, 0, 0);
		oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
		clFinish( cqCommandQueue );
	} else {
		// Explicit Copy 
		// map the PBO to copy data from the CL buffer via host
		glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);    

		// map the buffer object into client's memory
		GLubyte* ptr = (GLubyte*)glMapBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB,
			GL_WRITE_ONLY_ARB);
		clEnqueueReadBuffer(cqCommandQueue, pbo_cl, CL_TRUE, 0, sizeof(unsigned int) * height * width, ptr, 0, NULL, NULL);        
		oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
		glUnmapBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB); 
	}
}

// Display callback for GLUT main loop
//*****************************************************************************
void DisplayGL()
{
    // use OpenGL to build view matrix
    GLfloat modelView[16];
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
    glRotatef(-viewRotation[0], 1.0, 0.0, 0.0);
    glRotatef(-viewRotation[1], 0.0, 1.0, 0.0);
    glTranslatef(-viewTranslation[0], -viewTranslation[1], -viewTranslation[2]);
    glGetFloatv(GL_MODELVIEW_MATRIX, modelView);
    glPopMatrix();

    invViewMatrix[0] = modelView[0]; invViewMatrix[1] = modelView[4]; invViewMatrix[2] = modelView[8]; invViewMatrix[3] = modelView[12];
    invViewMatrix[4] = modelView[1]; invViewMatrix[5] = modelView[5]; invViewMatrix[6] = modelView[9]; invViewMatrix[7] = modelView[13];
    invViewMatrix[8] = modelView[2]; invViewMatrix[9] = modelView[6]; invViewMatrix[10] = modelView[10]; invViewMatrix[11] = modelView[14];

    // start timer 0 if it's update time
    double dProcessingTime = 0.0;
    if (iFrameCount >= iFrameTrigger)
    {
        shrDeltaT(0); 
    }

     // process 
    render();

    // get processing time from timer 0, if it's update time
    if (iFrameCount >= iFrameTrigger)
    {
        dProcessingTime = shrDeltaT(0); 
    }

    // draw image from PBO
    glClear(GL_COLOR_BUFFER_BIT);
    glDisable(GL_DEPTH_TEST);
    glRasterPos2i(0, 0);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
    glDrawPixels(width, height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

    // flip backbuffer to screen
    glutSwapBuffers();

    // Increment the frame counter, and do fps and Q/A stuff if it's time
    if (iFrameCount++ > iFrameTrigger) 
    {
        // set GLUT Window Title
        char cFPS[256];
        iFramesPerSec = (int)((double)iFrameCount/shrDeltaT(1));
#ifdef GPU_PROFILING
        #ifdef _WIN32
            sprintf_s(cFPS, 256, "Volume Render %ux%u | %i fps | Proc.t = %.3f s | %.3f MP/s", 
                width, height, iFramesPerSec, dProcessingTime, (1.0e-6 * width * height)/dProcessingTime);
        #else 
            sprintf(cFPS, "Volume Render %ux%u |  %i fps | Proc. t = %.3f s | %.3f MP/s", 
                width, height, iFramesPerSec, dProcessingTime, (1.0e-6 * width * height)/dProcessingTime);
        #endif
#else
        #ifdef _WIN32
            sprintf_s(cFPS, 256, "Volume Render | W: %u  H: %u", width, height);
        #else 
            sprintf(cFPS, "Volume Render | W: %u  H: %u", width, height);
        #endif
#endif
        glutSetWindowTitle(cFPS);

        // Log fps and processing info to console and file 
        shrLog(" %s\n", cFPS); 

        // if doing quick test, exit
        if ((bNoPrompt) && (!--iTestSets))
        {
            // Cleanup up and quit
            Cleanup(EXIT_SUCCESS);
        }

        // reset framecount, trigger and timer
        iFrameCount = 0; 
        iFrameTrigger = (iFramesPerSec > 1) ? iFramesPerSec * 2 : 1;
    }
}

// GL Idle time callback
//*****************************************************************************
void Idle()
{
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
        case '-':
            density -= 0.01f;
            break;
        case '+':
            density += 0.01f;
            break;

        case ']':
            brightness += 0.1f;
            break;
        case '[':
            brightness -= 0.1f;
            break;

        case ';':
            transferOffset += 0.01f;
            break;
        case '\'':
            transferOffset -= 0.01f;
            break;

        case '.':
            transferScale += 0.01f;
            break;
        case ',':
            transferScale -= 0.01f;
            break;
        case '\033': // escape quits
        case '\015': // Enter quits    
        case 'Q':    // Q quits
        case 'q':    // q (or escape) quits
            // Cleanup up and quit
            bNoPrompt = shrTRUE;
            Cleanup(EXIT_SUCCESS);
            break;
        case 'F':
        case 'f':
                    linearFiltering = !linearFiltering;
                    ciErrNum = clSetKernelArg(ckKernel, 10, sizeof(cl_sampler), linearFiltering ? &volumeSamplerLinear : &volumeSamplerNearest);
                    shrLog("\nLinear Filtering Toggled %s...\n", linearFiltering ? "ON" : "OFF");
                    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
                    break;
        default:
            break;
    }
    shrLog("density = %.2f, brightness = %.2f, transferOffset = %.2f, transferScale = %.2f\n", density, brightness, transferOffset, transferScale);
}

// Mouse event handlers
//*****************************************************************************
void mouse(int button, int state, int x, int y)
{
    if (state == GLUT_DOWN)
        buttonState |= 1<<button;
    else if (state == GLUT_UP)
        buttonState = 0;

    ox = x; 
    oy = y;
}

void motion(int x, int y)
{
    float dx, dy;
    dx = (float)(x - ox);
    dy = (float)(y - oy);

    if (buttonState == 3) {
        // left+middle = zoom
        viewTranslation[2] += dy / 100.0f;
    } 
    else if (buttonState & 2) {
        // middle = translate
        viewTranslation[0] += dx / 100.0f;
        viewTranslation[1] -= dy / 100.0f;
    }
    else if (buttonState & 1) {
        // left = rotate
        viewRotation[0] += dy / 5.0f;
        viewRotation[1] += dx / 5.0f;
    }

    ox = x; 
    oy = y;
}

// Window resize handler callback
//*****************************************************************************
void Reshape(int x, int y)
{
    width = x; height = y;
    initPixelBuffer();

    glViewport(0, 0, x, y);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0); 
}

// Intitialize OpenCL
//*****************************************************************************
void createCLContext(int argc, const char** argv) {
    //Get the NVIDIA platform
    ciErrNum = oclGetPlatformID(&cpPlatform);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    // Get the number of GPU devices available to the platform
    ciErrNum = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_CPU, 0, NULL, &uiDevCount);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    // Create the device list
    cdDevices = new cl_device_id [uiDevCount];
    ciErrNum = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_CPU, uiDevCount, cdDevices, NULL);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    // Get device requested on command line, if any
    uiDeviceUsed = 0;
    unsigned int uiEndDev = uiDevCount - 1;
    if(shrGetCmdLineArgumentu(argc, argv, "device", &uiDeviceUsed ))
    {
      uiDeviceUsed = CLAMP(uiDeviceUsed, 0, uiEndDev);
      uiEndDev = uiDeviceUsed; 
    } 

	// Check if the requested device (or any of the devices if none requested) supports context sharing with OpenGL
    if(g_glInterop && !bQATest)
    {
        bool bSharingSupported = false;
        for(unsigned int i = uiDeviceUsed; (!bSharingSupported && (i <= uiEndDev)); ++i) 
        {
            size_t extensionSize;
            ciErrNum = clGetDeviceInfo(cdDevices[i], CL_DEVICE_EXTENSIONS, 0, NULL, &extensionSize );
            oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
            if(extensionSize > 0) 
            {
                char* extensions = (char*)malloc(extensionSize);
                ciErrNum = clGetDeviceInfo(cdDevices[i], CL_DEVICE_EXTENSIONS, extensionSize, extensions, &extensionSize);
                oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
                std::string stdDevString(extensions);
                free(extensions);

                size_t szOldPos = 0;
                size_t szSpacePos = stdDevString.find(' ', szOldPos); // extensions string is space delimited
                while (szSpacePos != stdDevString.npos)
                {
                    if( strcmp(GL_SHARING_EXTENSION, stdDevString.substr(szOldPos, szSpacePos - szOldPos).c_str()) == 0 ) 
                    {
                        // Device supports context sharing with OpenGL
                        uiDeviceUsed = i;
                        bSharingSupported = true;
                        break;
                    }
                    do 
                    {
                        szOldPos = szSpacePos + 1;
                        szSpacePos = stdDevString.find(' ', szOldPos);
                    } 
                    while (szSpacePos == szOldPos);
                }
            }
        }
       
        // Log CL-GL interop support and quit if not available (sample needs it)
        shrLog("%s...\n", bSharingSupported ? "Using CL-GL Interop" : "No device found that supports CL/GL context sharing");  
        oclCheckErrorEX(bSharingSupported, true, pCleanup);

        // Define OS-specific context properties and create the OpenCL context
        #if defined (__APPLE__)
            CGLContextObj kCGLContext = CGLGetCurrentContext();
            CGLShareGroupObj kCGLShareGroup = CGLGetShareGroup(kCGLContext);
            cl_context_properties props[] = 
            {
                CL_CONTEXT_PROPERTY_USE_CGL_SHAREGROUP_APPLE, (cl_context_properties)kCGLShareGroup, 
                0 
            };
            cxGPUContext = clCreateContext(props, 0,0, NULL, NULL, &ciErrNum);
        #else
            #ifdef UNIX
                cl_context_properties props[] = 
                {
                    CL_GL_CONTEXT_KHR, (cl_context_properties)glXGetCurrentContext(), 
                    CL_GLX_DISPLAY_KHR, (cl_context_properties)glXGetCurrentDisplay(), 
                    CL_CONTEXT_PLATFORM, (cl_context_properties)cpPlatform, 
                    0
                };
                cxGPUContext = clCreateContext(props, 1, &cdDevices[uiDeviceUsed], NULL, NULL, &ciErrNum);
            #else // Win32
                cl_context_properties props[] = 
                {
                    CL_GL_CONTEXT_KHR, (cl_context_properties)wglGetCurrentContext(), 
                    CL_WGL_HDC_KHR, (cl_context_properties)wglGetCurrentDC(), 
                    CL_CONTEXT_PLATFORM, (cl_context_properties)cpPlatform, 
                    0
                };
                cxGPUContext = clCreateContext(props, 1, &cdDevices[uiDeviceUsed], NULL, NULL, &ciErrNum);
            #endif
        #endif
    }
    else 
    {
		// No GL interop
        cl_context_properties props[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)cpPlatform, 0};
        cxGPUContext = clCreateContext(props, 1, &cdDevices[uiDeviceUsed], NULL, NULL, &ciErrNum);

		g_glInterop = false;
    }
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
}

void initCLVolume(uchar *h_volume)
{
    ciErrNum = CL_SUCCESS;

	if (g_bImageSupport) 
    {
		// create 3D array and copy data to device
		cl_image_format volume_format;
        volume_format.image_channel_order = CL_RGBA;
        volume_format.image_channel_data_type = CL_UNORM_INT8;
        uchar* h_tempVolume = (uchar*)malloc(volumeSize[0] * volumeSize[1] * volumeSize[2] * 4 );
        for(int i = 0; i <(int)(volumeSize[0] * volumeSize[1] * volumeSize[2]); i++)
        {
            h_tempVolume[4 * i] = h_volume[i];
        }
        d_volumeArray = clCreateImage3D(cxGPUContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, &volume_format, 
                                        volumeSize[0], volumeSize[1], volumeSize[2],
                                        (volumeSize[0] * 4), (volumeSize[0] * volumeSize[1] * 4),
                                        h_tempVolume, &ciErrNum);
		oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
        free (h_tempVolume);

		// create transfer function texture
		float transferFunc[] = {
			 0.0, 0.0, 0.0, 0.0, 
			 1.0, 0.0, 0.0, 1.0, 
			 1.0, 0.5, 0.0, 1.0, 
			 1.0, 1.0, 0.0, 1.0, 
			 0.0, 1.0, 0.0, 1.0, 
			 0.0, 1.0, 1.0, 1.0, 
			 0.0, 0.0, 1.0, 1.0, 
			 1.0, 0.0, 1.0, 1.0, 
			 0.0, 0.0, 0.0, 0.0, 
		};

		cl_image_format transferFunc_format;
		transferFunc_format.image_channel_order = CL_RGBA;
		transferFunc_format.image_channel_data_type = CL_FLOAT;
		d_transferFuncArray = clCreateImage2D(cxGPUContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, &transferFunc_format,
											  9, 1, sizeof(float) * 9 * 4,
											  transferFunc, &ciErrNum);                                          
		oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

        // Create samplers for transfer function, linear interpolation and nearest interpolation 
        transferFuncSampler = clCreateSampler(cxGPUContext, true, CL_ADDRESS_CLAMP_TO_EDGE, CL_FILTER_LINEAR, &ciErrNum);
        oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
        volumeSamplerLinear = clCreateSampler(cxGPUContext, true, CL_ADDRESS_REPEAT, CL_FILTER_LINEAR, &ciErrNum);
        oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
        volumeSamplerNearest = clCreateSampler(cxGPUContext, true, CL_ADDRESS_REPEAT, CL_FILTER_NEAREST, &ciErrNum);
        oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

        // set image and sampler args
        ciErrNum = clSetKernelArg(ckKernel, 8, sizeof(cl_mem), (void *) &d_volumeArray);
		ciErrNum |= clSetKernelArg(ckKernel, 9, sizeof(cl_mem), (void *) &d_transferFuncArray);
        ciErrNum |= clSetKernelArg(ckKernel, 10, sizeof(cl_sampler), linearFiltering ? &volumeSamplerLinear : &volumeSamplerNearest);
        ciErrNum |= clSetKernelArg(ckKernel, 11, sizeof(cl_sampler), &transferFuncSampler);
		oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
	}

    // init invViewMatrix
    d_invViewMatrix = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY, 12 * sizeof(float), 0, &ciErrNum);
    ciErrNum |= clSetKernelArg(ckKernel, 7, sizeof(cl_mem), (void *) &d_invViewMatrix);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
}

// Initialize GL
//*****************************************************************************
void InitGL(int* argc, char **argv)
{
    // initialize GLUT 
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
    glutInitWindowPosition (glutGet(GLUT_SCREEN_WIDTH)/2 - width/2, 
                            glutGet(GLUT_SCREEN_HEIGHT)/2 - height/2);
    glutInitWindowSize(width, height);
    iGLUTWindowHandle = glutCreateWindow("OpenCL volume rendering");
#if !(defined (__APPLE__) || defined(MACOSX))
    glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS);
#endif

    // register glut callbacks
    glutDisplayFunc(DisplayGL);
    glutKeyboardFunc(KeyboardGL);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);
    glutReshapeFunc(Reshape);
    glutIdleFunc(Idle);
	glutTimerFunc(REFRESH_DELAY, timerEvent,0);

	// initialize necessary OpenGL extensions
    glewInit();
    GLboolean bGLEW = glewIsSupported("GL_VERSION_2_0 GL_ARB_pixel_buffer_object"); 
    oclCheckErrorEX(bGLEW, shrTRUE, pCleanup);

	g_glInterop = true;
}

// Initialize GL
//*****************************************************************************
void initPixelBuffer()
{
     ciErrNum = CL_SUCCESS;

    if (pbo) {
        // delete old buffer
        clReleaseMemObject(pbo_cl);
        glDeleteBuffersARB(1, &pbo);
    }

    // create pixel buffer object for display
    glGenBuffersARB(1, &pbo);
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
	glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, width * height * sizeof(GLubyte) * 4, 0, GL_STREAM_DRAW_ARB);
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

	if( g_glInterop ) {
		// create OpenCL buffer from GL PBO
		pbo_cl = clCreateFromGLBuffer(cxGPUContext,CL_MEM_WRITE_ONLY, pbo, &ciErrNum);
		oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
	} else {
		pbo_cl = clCreateBuffer(cxGPUContext, CL_MEM_WRITE_ONLY, width * height * sizeof(GLubyte) * 4, NULL, &ciErrNum);
	}

    // calculate new grid size
	gridSize[0] = shrRoundUp(LOCAL_SIZE_X,width);
	gridSize[1] = shrRoundUp(LOCAL_SIZE_Y,height);

    ciErrNum |= clSetKernelArg(ckKernel, 0, sizeof(cl_mem), (void *) &pbo_cl);
    ciErrNum |= clSetKernelArg(ckKernel, 1, sizeof(unsigned int), &width);
    ciErrNum |= clSetKernelArg(ckKernel, 2, sizeof(unsigned int), &height);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
}

// Run a test sequence without any GL 
//*****************************************************************************
void TestNoGL()
{
    // execute OpenCL kernel without GL interaction
    float modelView[16] = 
        {
            1.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 1.0f, 0.0f,
            0.0f, 0.0f, 4.0f, 1.0f
        };
    
    invViewMatrix[0] = modelView[0]; invViewMatrix[1] = modelView[4]; invViewMatrix[2] = modelView[8]; invViewMatrix[3] = modelView[12];
    invViewMatrix[4] = modelView[1]; invViewMatrix[5] = modelView[5]; invViewMatrix[6] = modelView[9]; invViewMatrix[7] = modelView[13];
    invViewMatrix[8] = modelView[2]; invViewMatrix[9] = modelView[6]; invViewMatrix[10] = modelView[10]; invViewMatrix[11] = modelView[14];
    
    pbo_cl = clCreateBuffer(cxGPUContext, CL_MEM_WRITE_ONLY,  width*height*sizeof(GLubyte)*4, NULL, &ciErrNum);
    ciErrNum |= clEnqueueWriteBuffer(cqCommandQueue,d_invViewMatrix,CL_FALSE, 0,12*sizeof(float), invViewMatrix, 0, 0, 0);

    gridSize[0] = width;
    gridSize[1] = height;

    ciErrNum |= clSetKernelArg(ckKernel, 0, sizeof(cl_mem), (void *) &pbo_cl);
    ciErrNum |= clSetKernelArg(ckKernel, 1, sizeof(unsigned int), &width);
    ciErrNum |= clSetKernelArg(ckKernel, 2, sizeof(unsigned int), &height);
    ciErrNum |= clSetKernelArg(ckKernel, 3, sizeof(float), &density);
    ciErrNum |= clSetKernelArg(ckKernel, 4, sizeof(float), &brightness);
    ciErrNum |= clSetKernelArg(ckKernel, 5, sizeof(float), &transferOffset);
    ciErrNum |= clSetKernelArg(ckKernel, 6, sizeof(float), &transferScale);
    
    // Warmup
    int iCycles = 20;
    size_t localSize[] = {LOCAL_SIZE_X,LOCAL_SIZE_Y};
    for (int i = 0; i < iCycles ; i++)
    {
        ciErrNum |= clEnqueueNDRangeKernel(cqCommandQueue, ckKernel, 2, NULL, gridSize, localSize, 0, 0, 0);
        oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    }
    clFinish(cqCommandQueue);
    
    // Start timer 0 and process n loops on the GPU 
    shrDeltaT(0); 
    for (int i = 0; i < iCycles ; i++)
    {
        ciErrNum |= clEnqueueNDRangeKernel(cqCommandQueue, ckKernel, 2, NULL, gridSize, localSize, 0, 0, 0);
        oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    }
    clFinish(cqCommandQueue);
    
    // Get elapsed time and throughput, then log to sample and master logs
    double dAvgTime = shrDeltaT(0)/(double)iCycles;
    shrLogEx(LOGBOTH | MASTER, 0, "oclVolumeRender, Throughput = %.4f MTexels/s, Time = %.5f s, Size = %u Texels, NumDevsUsed = %u, Workgroup = %u\n", 
           (1.0e-6 * width * height)/dAvgTime, dAvgTime, (width * height), 1, localSize[0] * localSize[1]); 

    // Cleanup and exit
    Cleanup(EXIT_SUCCESS);
}

// Function to clean up and exit
//*****************************************************************************
void Cleanup(int iExitCode)
{
    // cleanup allocated objects
    shrLog("\nStarting Cleanup...\n\n");
    if(cPathAndName)free(cPathAndName);
    if(cSourceCL)free(cSourceCL);
	if(ckKernel)clReleaseKernel(ckKernel);  
    if(cpProgram)clReleaseProgram(cpProgram);
    if(volumeSamplerLinear)clReleaseSampler(volumeSamplerLinear);
    if(volumeSamplerNearest)clReleaseSampler(volumeSamplerNearest);
    if(transferFuncSampler)clReleaseSampler(transferFuncSampler);
    if(d_volumeArray)clReleaseMemObject(d_volumeArray);
    if(d_transferFuncArray)clReleaseMemObject(d_transferFuncArray);
    if(pbo_cl)clReleaseMemObject(pbo_cl);    
    if(d_invViewMatrix)clReleaseMemObject(d_invViewMatrix);    
    if(cqCommandQueue)clReleaseCommandQueue(cqCommandQueue);
    if(cxGPUContext)clReleaseContext(cxGPUContext);
	if(!bQATest) 
    {
        glDeleteBuffersARB(1, &pbo);
    }
    shrQAFinish2(bQATest, *pArgc, (const char **)pArgv, (iExitCode == 0) ? QA_PASSED : QA_FAILED); 

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
