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
    This example demonstrates how to use the OpenCL/OpenGL interoperability to
    dynamically modify a vertex buffer using a OpenCL kernel.

    The steps are:
    1. Create an empty vertex buffer object (VBO)
    2. Create an OpenCL memory object from the vertex buffer object
    3. Acquire the VBO for writing from OpenCL
    4. Run OpenCL kernel to modify the vertex positions
    5. Release the VBO for returning ownership to OpenGL
    6. Render the results using OpenGL

    Host code
*/

#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

// OpenGL Graphics Includes
#include <GL/glew.h>
#if defined (__APPLE__) || defined(MACOSX)
    #include <OpenGL/OpenGL.h>
    #include <GLUT/glut.h>
#else
    #include <GL/freeglut.h>
    #ifdef UNIX
       #include <GL/glx.h>
    #endif
#endif

// Includes
#include <memory>
#include <iostream>
#include <cassert>

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

// Rendering window vars
const unsigned int window_width = 512;
const unsigned int window_height = 512;
const unsigned int mesh_width = 256;
const unsigned int mesh_height = 256;

// OpenCL vars
cl_platform_id cpPlatform;
cl_context cxGPUContext;
cl_device_id* cdDevices;
cl_uint uiDevCount;
cl_command_queue cqCommandQueue;
cl_kernel ckKernel;
cl_mem vbo_cl;
cl_program cpProgram;
cl_int ciErrNum;
char* cPathAndName = NULL;          // var for full paths to data, src, etc.
char* cSourceCL = NULL;             // Buffer to hold source for compilation 
size_t szGlobalWorkSize[] = {mesh_width, mesh_height};
const char* cExecutableName = NULL;

// vbo variables
GLuint vbo;
int iGLUTWindowHandle = 0;          // handle to the GLUT window

// mouse controls
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0;
float translate_z = -3.0;

// Sim and Auto-Verification parameters 
float anim = 0.0;
int iFrameCount = 0;                // FPS count for averaging
int iFrameTrigger = 90;             // FPS trigger for sampling
int iFramesPerSec = 0;              // frames per second
int iTestSets = 3;
int g_Index = 0;
shrBOOL bQATest = shrFALSE;
shrBOOL bNoPrompt = shrFALSE;  

int *pArgc = NULL;
char **pArgv = NULL;

// Forward Function declarations
//*****************************************************************************
// OpenCL functionality
void runKernel();
void saveResultOpenCL(int argc, const char** argv, const GLuint& vbo);

// GL functionality
void InitGL(int* argc, char** argv);
void createVBO(GLuint* vbo);
void DisplayGL();
void KeyboardGL(unsigned char key, int x, int y);
void mouse(int button, int state, int x, int y);
void motion(int x, int y);
void timerEvent(int value);

// Helpers
void TestNoGL();
void Cleanup(int iExitCode);
void (*pCleanup)(int) = &Cleanup;

// Main program
//*****************************************************************************
int main(int argc, char** argv)
{
    pArgc = &argc;
    pArgv = argv;

    // start logs 
    shrQAStart(argc, argv);
    cExecutableName = argv[0];
    shrSetLogFileName ("oclSimpleGL.txt");
    shrLog("%s Starting...\n\n", argv[0]); 

    // check command line args
    if (argc > 1) 
    {
        bQATest   = shrCheckCmdLineFlag(argc, (const char**)argv, "qatest");
        bNoPrompt = shrCheckCmdLineFlag(argc, (const char**)argv, "noprompt");
    }

    // Initialize OpenGL items (if not No-GL QA test)
    shrLog("%sInitGL...\n\n", bQATest ? "Skipping " : "Calling "); 
    if(!bQATest)
    {
        InitGL(&argc, argv);
    }

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
    unsigned int uiDeviceUsed = 0;
    unsigned int uiEndDev = uiDevCount - 1;
    if(shrGetCmdLineArgumentu(argc, (const char**)argv, "device", &uiDeviceUsed ))
    {
      uiDeviceUsed = CLAMP(uiDeviceUsed, 0, uiEndDev);
      uiEndDev = uiDeviceUsed; 
    } 

    // Check if the requested device (or any of the devices if none requested) supports context sharing with OpenGL
    if(!bQATest)
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
       
        shrLog("%s...\n\n", bSharingSupported ? "Using CL-GL Interop" : "No device found that supports CL/GL context sharing");  
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
        cl_context_properties props[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)cpPlatform, 0};
        cxGPUContext = clCreateContext(props, 1, &cdDevices[uiDeviceUsed], NULL, NULL, &ciErrNum);
    }
    shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    // Log device used (reconciled for requested requested and/or CL-GL interop capable devices, as applies)
    shrLog("Device # %u, ", uiDeviceUsed);
    oclPrintDevName(LOGBOTH, cdDevices[uiDeviceUsed]);
    shrLog("\n");

    // create a command-queue
    cqCommandQueue = CECL_CREATE_COMMAND_QUEUE(cxGPUContext, cdDevices[uiDeviceUsed], 0, &ciErrNum);
    shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    // Program Setup
    size_t program_length;
    cPathAndName = shrFindFilePath("simpleGL.cl", argv[0]);
    shrCheckErrorEX(cPathAndName != NULL, shrTRUE, pCleanup);
    cSourceCL = oclLoadProgSource(cPathAndName, "", &program_length);
    shrCheckErrorEX(cSourceCL != NULL, shrTRUE, pCleanup);

    // create the program
    cpProgram = CECL_PROGRAM_WITH_SOURCE(cxGPUContext, 1,
					  (const char **) &cSourceCL, &program_length, &ciErrNum);
    shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    // build the program
    ciErrNum = CECL_PROGRAM(cpProgram, 0, NULL, "-cl-fast-relaxed-math", NULL, NULL);
    if (ciErrNum != CL_SUCCESS)
    {
        // write out standard error, Build Log and PTX, then cleanup and exit
        shrLogEx(LOGBOTH | ERRORMSG, ciErrNum, STDERROR);
        oclLogBuildInfo(cpProgram, oclGetFirstDev(cxGPUContext));
        oclLogPtx(cpProgram, oclGetFirstDev(cxGPUContext), "oclSimpleGL.ptx");
        Cleanup(EXIT_FAILURE); 
    }

    // create the kernel
    ckKernel = CECL_KERNEL(cpProgram, "sine_wave", &ciErrNum);
    shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    // create VBO (if using standard GL or CL-GL interop), otherwise create Cl buffer
    createVBO(&vbo);

    // set the args values 
    ciErrNum  = CECL_SET_KERNEL_ARG(ckKernel, 0, sizeof(cl_mem), (void *) &vbo_cl);
    ciErrNum |= CECL_SET_KERNEL_ARG(ckKernel, 1, sizeof(unsigned int), &mesh_width);
    ciErrNum |= CECL_SET_KERNEL_ARG(ckKernel, 2, sizeof(unsigned int), &mesh_height);
    shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    // If specified, compute and save off data for regression tests
    if(shrCheckCmdLineFlag(argc, (const char**) argv, "regression")) 
    {
        // run OpenCL kernel once to generate vertex positions, then save results
        runKernel();
        saveResultOpenCL(argc, (const char**)argv, vbo);
    }

    // init timer 1 for fps measurement 
    shrDeltaT(1);  

    // Start main GLUT rendering loop for processing and rendering, 
	// or otherwise run No-GL Q/A test sequence
    shrLog("\n%s...\n", bQATest ? "No-GL test sequence" : "Standard GL Loop"); 
    if(!bQATest) 
    {
        glutMainLoop();
    }
    else
    {
        TestNoGL();
    }

    // Normally unused return path
    Cleanup(EXIT_SUCCESS);
}

// Initialize GL
//*****************************************************************************
void InitGL(int* argc, char** argv)
{
    // initialize GLUT 
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowPosition (glutGet(GLUT_SCREEN_WIDTH)/2 - window_width/2, 
                            glutGet(GLUT_SCREEN_HEIGHT)/2 - window_height/2);
    glutInitWindowSize(window_width, window_height);
    iGLUTWindowHandle = glutCreateWindow("OpenCL/GL Interop (VBO)");
#if !(defined (__APPLE__) || defined(MACOSX))
    glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS);
#endif

    // register GLUT callback functions
    glutDisplayFunc(DisplayGL);
    glutKeyboardFunc(KeyboardGL);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);
	glutTimerFunc(REFRESH_DELAY, timerEvent,0);

	// initialize necessary OpenGL extensions
    glewInit();
    GLboolean bGLEW = glewIsSupported("GL_VERSION_2_0 GL_ARB_pixel_buffer_object"); 
    shrCheckErrorEX(bGLEW, shrTRUE, pCleanup);

    // default initialization
    glClearColor(0.0, 0.0, 0.0, 1.0);
    glDisable(GL_DEPTH_TEST);

    // viewport
    glViewport(0, 0, window_width, window_height);

    // projection
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (GLfloat)window_width / (GLfloat) window_height, 0.1, 10.0);

    // set view matrix
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(0.0, 0.0, translate_z);
    glRotatef(rotate_x, 1.0, 0.0, 0.0);
    glRotatef(rotate_y, 0.0, 1.0, 0.0);

    return;
}

// Run the OpenCL part of the computation
//*****************************************************************************
void runKernel()
{
    ciErrNum = CL_SUCCESS;
 
#ifdef GL_INTEROP   
    // map OpenGL buffer object for writing from OpenCL
    glFinish();
    ciErrNum = clEnqueueAcquireGLObjects(cqCommandQueue, 1, &vbo_cl, 0,0,0);
    shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
#endif

    // Set arg 3 and execute the kernel
    ciErrNum = CECL_SET_KERNEL_ARG(ckKernel, 3, sizeof(float), &anim);
    ciErrNum |= CECL_ND_RANGE_KERNEL(cqCommandQueue, ckKernel, 2, NULL, szGlobalWorkSize, NULL, 0,0,0 );
    shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

#ifdef GL_INTEROP
    // unmap buffer object
    ciErrNum = clEnqueueReleaseGLObjects(cqCommandQueue, 1, &vbo_cl, 0,0,0);
    shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    clFinish(cqCommandQueue);
#else

    // Explicit Copy 
    // map the PBO to copy data from the CL buffer via host
    glBindBufferARB(GL_ARRAY_BUFFER, vbo);    

    // map the buffer object into client's memory
    void* ptr = glMapBufferARB(GL_ARRAY_BUFFER, GL_WRITE_ONLY_ARB);

    ciErrNum = CECL_READ_BUFFER(cqCommandQueue, vbo_cl, CL_TRUE, 0, sizeof(float) * 4 * mesh_height * mesh_width, ptr, 0, NULL, NULL);
    shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    glUnmapBufferARB(GL_ARRAY_BUFFER); 
#endif
}

// Create VBO
//*****************************************************************************
void createVBO(GLuint* vbo)
{
    // create VBO
    unsigned int size = mesh_width * mesh_height * 4 * sizeof(float);
    if(!bQATest)
    {
        // create buffer object
        glGenBuffers(1, vbo);
        glBindBuffer(GL_ARRAY_BUFFER, *vbo);

        // initialize buffer object
        glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);

        #ifdef GL_INTEROP
            // create OpenCL buffer from GL VBO
            vbo_cl = clCreateFromGLBuffer(cxGPUContext, CL_MEM_WRITE_ONLY, *vbo, NULL);
        #else
            // create standard OpenCL mem buffer
            vbo_cl = CECL_BUFFER(cxGPUContext, CL_MEM_WRITE_ONLY, size, NULL, &ciErrNum);
        #endif
        shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    }
    else 
    {
        // create standard OpenCL mem buffer
        vbo_cl = CECL_BUFFER(cxGPUContext, CL_MEM_WRITE_ONLY, size, NULL, &ciErrNum);
        shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    }
}

// Display callback
//*****************************************************************************
void DisplayGL()
{
    // increment the geometry computation parameter (or set to reference for Q/A check)
    if (iFrameCount < iFrameTrigger)
    {
        anim += 0.01f;
    }

    // start timer 0 if it's update time
    double dProcessingTime = 0.0;
    if (iFrameCount >= iFrameTrigger)
    {
        shrDeltaT(0); 
    }

    // run OpenCL kernel to generate vertex positions
    runKernel();

    // get processing time from timer 0, if it's update time
    if (iFrameCount >= iFrameTrigger)
    {
        dProcessingTime = shrDeltaT(0); 
    }

    // clear graphics then render from the vbo
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexPointer(4, GL_FLOAT, 0, 0);
    glEnableClientState(GL_VERTEX_ARRAY);
    glColor3f(1.0, 0.0, 0.0);
    glDrawArrays(GL_POINTS, 0, mesh_width * mesh_height);
    glDisableClientState(GL_VERTEX_ARRAY);

    // flip backbuffer to screen
    glutSwapBuffers();

    // Increment the frame counter, and do fps if it's time
    if (iFrameCount++ > iFrameTrigger) 
    {
        // set GLUT Window Title
        char cTitle[256];
        iFramesPerSec = (int)((double)iFrameCount/shrDeltaT(1));
#ifdef GPU_PROFILING
        #ifdef _WIN32
            sprintf_s(cTitle, 256, "OpenCL Simple GL (VBO) | %u x %u | %i fps | Proc. t = %.5f s", 
                      mesh_width, mesh_height, iFramesPerSec, dProcessingTime);
        #else 
            sprintf(cTitle, "OpenCL Simple GL (VBO) | %u x %u | %i fps | Proc. t = %.5f s", 
                    mesh_width, mesh_height, iFramesPerSec, dProcessingTime);
        #endif
#else
        #ifdef _WIN32
            sprintf_s(cTitle, 256, "OpenCL Simple GL (VBO) | W: %u  H: %u", mesh_width, mesh_height );
        #else 
            sprintf(cTitle, "OpenCL Simple GL (VBO) | W: %u  H: %u", mesh_width, mesh_height);
        #endif
#endif
        glutSetWindowTitle(cTitle);

        // Log fps and processing info to console and file 
        shrLog(" %s\n", cTitle); 

        // Cleanup up and quit if requested and counter is up
        iTestSets--;
        if (bNoPrompt && (!iTestSets)) 
        {
            Cleanup(EXIT_SUCCESS);
        }

        // reset framecount, trigger and timer
        iFrameCount = 0; 
        iFrameTrigger = (iFramesPerSec > 1) ? iFramesPerSec * 2 : 1;
    }
}

void timerEvent(int value)
{
    glutPostRedisplay();
	glutTimerFunc(REFRESH_DELAY, timerEvent,0);
}

// Keyboard events handler
//*****************************************************************************
void KeyboardGL(unsigned char key, int x, int y)
{
    switch(key) 
    {
        case '\033': // escape quits
        case '\015': // Enter quits    
        case 'Q':    // Q quits
        case 'q':    // q (or escape) quits
            // Cleanup up and quit
            bNoPrompt = shrTRUE;
            Cleanup(EXIT_SUCCESS);
            break;
    }
}

// Mouse event handlers
//*****************************************************************************
void mouse(int button, int state, int x, int y)
{
    if (state == GLUT_DOWN) {
        mouse_buttons |= 1<<button;
    } else if (state == GLUT_UP) {
        mouse_buttons = 0;
    }

    mouse_old_x = x;
    mouse_old_y = y;
}

void motion(int x, int y)
{
    float dx, dy;
    dx = (float)(x - mouse_old_x);
    dy = (float)(y - mouse_old_y);

    if (mouse_buttons & 1) {
        rotate_x += dy * 0.2f;
        rotate_y += dx * 0.2f;
    } else if (mouse_buttons & 4) {
        translate_z += dy * 0.01f;
    }

    mouse_old_x = x;
    mouse_old_y = y;

    // set view matrix
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(0.0, 0.0, translate_z);
    glRotatef(rotate_x, 1.0, 0.0, 0.0);
    glRotatef(rotate_y, 0.0, 1.0, 0.0);
}

// If specified, write data to file for external regression testing
//*****************************************************************************
void saveResultOpenCL(int argc, const char** argv, const GLuint& vbo)
{
    // map buffer object
    glBindBuffer(GL_ARRAY_BUFFER_ARB, vbo);
    float* data = (float*)glMapBuffer(GL_ARRAY_BUFFER, GL_READ_ONLY);

    // save data for regression testing result
    shrWriteFilef("./data/regression.dat",
                      data, mesh_width * mesh_height * 3, 0.0);

    // unmap GL buffer object
    if(!glUnmapBuffer(GL_ARRAY_BUFFER))
    {
        shrLog("Unmap buffer failed !\n");
    }
}

// Run a test sequence without any GL 
//*****************************************************************************
void TestNoGL()
{
    // Set arg 3 and Warmup call to assure OpenCL driver is awake
    ciErrNum = CECL_SET_KERNEL_ARG(ckKernel, 3, sizeof(float), &anim);
    ciErrNum |= CECL_ND_RANGE_KERNEL(cqCommandQueue, ckKernel, 2, NULL, szGlobalWorkSize, NULL, 0, 0, 0 );
    shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    clFinish(cqCommandQueue);

    // Start timer 0 and process n loops on the GPU 
    const int iCycles = 250;
    shrLog("Running CECL_ND_RANGE_KERNEL for %d cycles...\n\n", iCycles); 
    shrDeltaT(0); 
    for (int i = 0; i < iCycles; i++)
    {
        ciErrNum |= CECL_ND_RANGE_KERNEL(cqCommandQueue, ckKernel, 2, NULL, szGlobalWorkSize, NULL, 0, 0, 0 );
    }
    shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    clFinish(cqCommandQueue);

    // Get elapsed time and throughput, then log to sample and master logs
    double dAvgTime = shrDeltaT(0)/(double)iCycles;
    shrLogEx(LOGBOTH | MASTER, 0, "oclSimpleGL, Throughput = %.4f MPixels/s, Time = %.5f s, Size = %u Pixels, NumDevsUsed = %u, Workgroup = %u\n", 
           (1.0e-6 * mesh_width * mesh_height)/dAvgTime, dAvgTime, (mesh_width * mesh_height), 1, 0); 

    // Cleanup and exit
    Cleanup(EXIT_SUCCESS);
}

// Function to clean up and exit
//*****************************************************************************
void Cleanup(int iExitCode)
{
    // Cleanup allocated objects
    shrLog("\nStarting Cleanup...\n\n");
    if(ckKernel)       clReleaseKernel(ckKernel); 
    if(cpProgram)      clReleaseProgram(cpProgram);
    if(cqCommandQueue) clReleaseCommandQueue(cqCommandQueue);
    if(vbo)
    {
        glBindBuffer(1, vbo);
        glDeleteBuffers(1, &vbo);
        vbo = 0;
    }
    if(vbo_cl)clReleaseMemObject(vbo_cl);
    if(cxGPUContext)clReleaseContext(cxGPUContext);
    if(cPathAndName)free(cPathAndName);
    if(cSourceCL)free(cSourceCL);
    if(cdDevices)delete(cdDevices);

    // finalize logs and leave
    shrQAFinish2(bQATest, *pArgc, (const char **)pArgv, (iExitCode == 0) ? QA_PASSED : QA_FAILED ); 
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
    exit (iExitCode);
}
