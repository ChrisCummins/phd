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

// *********************************************************************
// Demo application for postprocessing of OpenGL renderings with OpenCL
// Based on the CUDA postprocessGL sample
// *********************************************************************



// Glew, GLUT, GL includes
#include <GL/glew.h>
#if defined (_WIN32)
    #include <GL/wglew.h>
#endif

#ifdef UNIX
    #if defined(__APPLE__) || defined(MACOSX)
       #include <OpenGL/OpenGL.h>
       #include <GLUT/glut.h>
    #else
       #include <GL/freeglut.h>
       #include <GL/glx.h>
    #endif
#else
    #include <GL/freeglut.h>
#endif

// Includes
#include <memory>
#include <iostream>
#include <cassert>

// Utilities, OpenCL and system includes
#include <oclUtils.h>

// Shared Libraries for QA Test
#include <shrQATest.h>

#if defined (__APPLE__) || defined(MACOSX)
   #define GL_SHARING_EXTENSION "cl_APPLE_gl_sharing"
#else
   #define GL_SHARING_EXTENSION "cl_khr_gl_sharing"
#endif

#define REFRESH_DELAY	  10 //ms

int *pArgc = NULL;
char **pArgv = NULL;

// constants / global variables
//*****************************************************************************

// GL
int iGLUTWindowHandle;                      // handle to the GLUT window
int iGLUTMenuHandle;                        // handle to the GLUT menu
int iGraphicsWinWidth = 512;                // GL Window width
int iGraphicsWinHeight = 512;               // GL Window height
cl_int image_width = iGraphicsWinWidth;     // teapot image width
cl_int image_height = iGraphicsWinHeight;   // teapot image height
GLuint tex_screen;                          // (offscreen) render target
float rotate[3];                            // var for teapot view rotation 

// pbo variables
GLuint pbo_source;
GLuint pbo_dest;
unsigned int size_tex_data;
unsigned int num_texels;
unsigned int num_values;

// CL objects
cl_context cxGPUContext;
cl_command_queue cqCommandQueue;
cl_device_id device;
cl_uint uiNumDevsUsed = 1;          // Number of devices used in this sample 
cl_program cpProgram;
cl_kernel ckKernel;
size_t szGlobalWorkSize[2];
size_t szLocalWorkSize[2];
cl_mem cl_pbos[2] = {0,0};
cl_int ciErrNum;
const char* clSourcefile = "postprocessGL.cl";

// Timer and fps vars
int iFrameCount = 0;                // FPS count for averaging
int iFrameTrigger = 90;             // FPS trigger for sampling
int iFramesPerSec = 0;              // frames per second
int iTestSets = 3;                  // # of loop set retriggers before auto exit when bNoPrompt = shrTrue

// app configuration parms
const char* cProcessor [] = {"OpenCL GPU", "Host C++ CPU"};
int iProcFlag = 0;                  // 0 = GPU, 1 = CPU
shrBOOL bNoPrompt = shrFALSE;		// false = normal GL loop, true = Finite period of GL loop (a few seconds)
shrBOOL bQATest = shrFALSE;			// false = normal GL loop, true = run No-GL test sequence
bool bPostprocess = shrTRUE;        // true = run blur filter processing on GPU or host, false = just do display of old data
bool bAnimate = true;               // true = continue incrementing rotation of view with GL, false = stop rotation    
int blur_radius = 4;                // radius of 2D convolution performed in post processing step
bool bGLinterop = shrFALSE;
const char* cExecutableName = NULL;

// Forward Function declarations
//*****************************************************************************
// OpenCL functionality
int initCL(int argc, const char** argv);
void renderScene();
void displayImage();
void processImage();
void postprocessHost(unsigned int* g_data, unsigned int* g_odata, int imgw, int imgh, int tilew, int radius, float threshold, float highlight);

// GL functionality
bool InitGL(int* argc, char** argv);
void createPBO(GLuint* pbo);
void deletePBO(GLuint* pbo);
void createTexture(GLuint* tex_name, unsigned int size_x, unsigned int size_y);
void deleteTexture(GLuint* tex);
void dumpImage();
void DisplayGL();
void idle();
void KeyboardGL(unsigned char key, int x, int y);
void timerEvent(int value);
void Reshape(int w, int h);
void mainMenu(int i);

// Helpers
void Cleanup(int iExitCode);
void (*pCleanup)(int) = &Cleanup;
void TestNoGL();
void TriggerFPSUpdate();

// Main Program
//*****************************************************************************
int main(int argc, char** argv) 
{
    pArgc = &argc;
    pArgv = argv;

    shrQAStart(argc, argv);

    // start logs 
    cExecutableName = argv[0];
    shrSetLogFileName ("oclPostProcessGL.txt");
    shrLog("%s Starting...\n\n", argv[0]); 

    // process command line arguments
    if (argc > 1) 
    {
        bQATest   = shrCheckCmdLineFlag(argc, (const char**)argv, "qatest");
        bNoPrompt = shrCheckCmdLineFlag(argc, (const char**)argv, "noprompt");
    }
    shrLog(" Image Width = %d, Image Height = %d, Blur Radius = %d\n\n", image_width, image_height, blur_radius);

    // init GL
    if(!bQATest) 
    {
        InitGL(&argc, argv);

        // create pbo
        createPBO(&pbo_source);
        createPBO(&pbo_dest);
        
        // create texture for blitting onto the screen
        createTexture(&tex_screen, image_width, image_height);        

        bGLinterop = shrTRUE;
    }

    // init CL
    if( initCL(argc, (const char**)argv) != 0 ) 
    {
        return -1;
    }

    // init fps timer
    shrDeltaT (1);

    // Create buffers and textures, 
    // and then start main GLUT rendering loop for processing and rendering, 
	// or otherwise run No-GL Q/A test sequence
    shrLog("\n%s...\n", bQATest ? "No-GL test sequence" : "Standard GL Loop"); 

    printf("\n"
        "\tControls\n"
		"\t(right click mouse button for Menu)\n"
		"\t[   ] : Toggle Post-Processing (blur filter) ON/OFF\n"
		"\t[ p ] : Toggle Processing (between GPU or CPU)\n"
		"\t[ a ] : Toggle OpenGL Animation (rotation) ON/OFF\n"
		"\t[+/=] : Increase Blur Radius\n"
		"\t[-/_] : Decrease Blur Radius\n"
		"\t[Esc] - Quit\n\n"
        );

    if(!bQATest) 
    {
        glutMainLoop();
    } 
    else 
    {
        TestNoGL();
    }
    
    Cleanup(EXIT_SUCCESS);
}

//*****************************************************************************
void dumpImage() 
{
    unsigned char* h_dump = (unsigned char*) malloc(sizeof(unsigned int) * image_height * image_width);
    
    clEnqueueReadBuffer(cqCommandQueue, cl_pbos[1], CL_TRUE, 0, sizeof(unsigned int) * image_height * image_width, 
                        h_dump, 0, NULL, NULL);
    
    shrSavePPM4ub( "dump.ppm", h_dump, image_width, image_height);
    free(h_dump);
}

//*****************************************************************************
void displayImage()
{
    // render a screen sized quad
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_LIGHTING);
    glEnable(GL_TEXTURE_2D);
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0);

    glMatrixMode( GL_MODELVIEW);
    glLoadIdentity();

    glViewport(0, 0, iGraphicsWinWidth, iGraphicsWinHeight);

    glBegin(GL_QUADS);

    glTexCoord2f(0.0, 0.0);
    glVertex3f(-1.0, -1.0, 0.5);

    glTexCoord2f(1.0, 0.0);
    glVertex3f(1.0, -1.0, 0.5);

    glTexCoord2f(1.0, 1.0);
    glVertex3f(1.0, 1.0, 0.5);

    glTexCoord2f(0.0, 1.0);
    glVertex3f(-1.0, 1.0, 0.5);

    glEnd();

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();

    glDisable(GL_TEXTURE_2D);
    glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
}

// Display callback
//*****************************************************************************
void DisplayGL()
{
    // Render the 3D teapot with GL
    renderScene();

    // start timer 0 if it's update time
    double dProcessingTime = 0.0;
    if (iFrameCount >= iFrameTrigger)
    {
        shrDeltaT(0); 
    }

    // process 
    processImage();

    // get processing time from timer 0, if it's update time
    if (iFrameCount >= iFrameTrigger)
    {
        dProcessingTime = shrDeltaT(0); 
    }

    // flip backbuffer to screen
    displayImage();
    glutSwapBuffers();
    glutPostRedisplay();

    // Increment the frame counter, and do fps and Q/A stuff if it's time
    if (iFrameCount++ > iFrameTrigger) 
    {
        // set GLUT Window Title
        char cFPS[256];
        iFramesPerSec = (int)((double)iFrameCount/shrDeltaT(1));
#ifdef GPU_PROFILING
        if( bPostprocess ) 
        {
            #ifdef _WIN32
            sprintf_s(cFPS, 256, "%s Postprocess ON  %ix%i | %i fps | Proc.t = %.3f s",  
                                      cProcessor[iProcFlag], iGraphicsWinWidth, iGraphicsWinHeight, 
                                      iFramesPerSec, dProcessingTime);
            #else 
            sprintf(cFPS, "%s Postprocess ON  %ix%i | %i fps | Proc.t = %.3f s",  
                               cProcessor[iProcFlag], iGraphicsWinWidth, iGraphicsWinHeight, 
                               iFramesPerSec, dProcessingTime);
            #endif
        } 
        else 
        {
            #ifdef _WIN32
            sprintf_s(cFPS, 256, "Postprocess OFF  %ix%i | %i fps",  iGraphicsWinWidth, iGraphicsWinHeight, iFramesPerSec);
            #else 
            sprintf(cFPS, "Postprocess OFF  %ix%i | %i fps",  iGraphicsWinWidth, iGraphicsWinHeight, iFramesPerSec);
            #endif
        }
#else 
        if(bPostprocess) 
        {
            #ifdef _WIN32
            sprintf_s(cFPS, 256, "%s Postprocess ON  %ix%i",  cProcessor[iProcFlag], iGraphicsWinWidth, iGraphicsWinHeight);
            #else 
            sprintf(cFPS, "%s Postprocess ON  %ix%i",  cProcessor[iProcFlag], iGraphicsWinWidth, iGraphicsWinHeight);
            #endif
        } 
        else 
        {
            #ifdef _WIN32
            sprintf_s(cFPS, 256, "%s Postprocess OFF  %ix%i",  cProcessor[iProcFlag], iGraphicsWinWidth, iGraphicsWinHeight);
            #else 
            sprintf(cFPS, "%s Postprocess OFF  %ix%i",  cProcessor[iProcFlag], iGraphicsWinWidth, iGraphicsWinHeight);
            #endif
        }
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

//*****************************************************************************
void timerEvent(int value)
{
    if (bAnimate) {
        rotate[0] += 0.2f; if( rotate[0] > 360.0f ) rotate[0] -= 360.0f;
        rotate[1] += 0.6f; if( rotate[1] > 360.0f ) rotate[1] -= 360.0f;
        rotate[2] += 1.0f; if( rotate[2] > 360.0f ) rotate[2] -= 360.0f;    
    }
    glutPostRedisplay();
    glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
}

// Keyboard events handler
//*****************************************************************************
void KeyboardGL(unsigned char key, int x, int y)
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
        case ' ':   // space bar toggles processing on and off
            bPostprocess = !bPostprocess;
            shrLog("\nPostprocessing (Blur Filter) Toggled %s...\n", bPostprocess ? "ON" : "OFF");
            break;
        case 'A':   // 'A' toggles animation (spinning of teacup) on/off  
        case 'a':   // 'a' toggles animation (spinning of teacup) on/off 
            bAnimate = !bAnimate;
            shrLog("\nGL Animation (Rotation) Toggled %s...\n", bAnimate ? "ON" : "OFF");
            break;
        case '=':
        case '+':
            if (blur_radius < 16) blur_radius++;
            shrLog("\nBlur radius = %d\n", blur_radius);
            break;
        case '-':
        case '_':
            if (blur_radius > 1) blur_radius--;
            shrLog("\nBlur radius = %d\n", blur_radius);
            break;
        case '\033': // escape quits
        case '\015': // Enter quits    
        case 'Q':    // Q quits
        case 'q':    // q (or escape) quits
            // Cleanup then quit (without prompting)
            bNoPrompt = shrTRUE;
            Cleanup(EXIT_SUCCESS);
            break;
    }

    // Trigger fps update and call for refresh
    TriggerFPSUpdate();
    glutPostRedisplay();
}

// Window resize handler callback
//*****************************************************************************
void Reshape(int w, int h)
{		
    w = MAX(w,1);
    h = MAX(h,1);

    iGraphicsWinWidth = w;
    iGraphicsWinHeight = h;

    glBindTexture(GL_TEXTURE_2D, tex_screen);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

    image_width = w;
    image_height = h;

    num_texels = image_width * image_height;
    num_values = num_texels * 4;
    size_tex_data = sizeof(GLubyte) * num_values;

    if( cl_pbos[0] != 0 ) {
      // update sizes of pixel buffer objects
      glBindBuffer(GL_ARRAY_BUFFER, pbo_source);
      glBufferData(GL_ARRAY_BUFFER, size_tex_data, NULL, GL_DYNAMIC_DRAW);

      glBindBuffer(GL_ARRAY_BUFFER, pbo_dest);
      glBufferData(GL_ARRAY_BUFFER, size_tex_data, NULL, GL_DYNAMIC_DRAW);

      glBindBuffer(GL_ARRAY_BUFFER,0);
    

	  // release current mem objects
	  clReleaseMemObject(cl_pbos[0]);
	  clReleaseMemObject(cl_pbos[1]);

	  // create new objects for the current sizes
	  if( bGLinterop ) {
		  cl_pbos[0] = clCreateFromGLBuffer(cxGPUContext, CL_MEM_READ_ONLY, pbo_source, &ciErrNum);
		  cl_pbos[1] = clCreateFromGLBuffer(cxGPUContext, CL_MEM_WRITE_ONLY, pbo_dest, &ciErrNum);
	  } else {
		  cl_pbos[0] = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY, 4 * image_width * image_height, NULL, &ciErrNum);
		  cl_pbos[1] = clCreateBuffer(cxGPUContext, CL_MEM_WRITE_ONLY, 4 * image_width * image_height, NULL, &ciErrNum);
	  }

	  // update kernel arguments
	  clSetKernelArg(ckKernel, 0, sizeof(cl_mem), (void *) &(cl_pbos[0]));
	  clSetKernelArg(ckKernel, 1, sizeof(cl_mem), (void *) &(cl_pbos[1]));
      clSetKernelArg(ckKernel, 2, sizeof(cl_int), &image_width);
      clSetKernelArg(ckKernel, 3, sizeof(cl_int), &image_height);	
    }

    glutPostRedisplay();
}


//*****************************************************************************
void mainMenu(int i)
{
    KeyboardGL((unsigned char) i, 0, 0);
}

//*****************************************************************************
void deleteTexture(GLuint* tex)
{
    glDeleteTextures(1, tex);

    *tex = 0;
}

//*****************************************************************************
void createTexture( GLuint* tex_name, unsigned int size_x, unsigned int size_y)
{
    // create a texture
    glGenTextures(1, tex_name);
    glBindTexture(GL_TEXTURE_2D, *tex_name);

    // set basic parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    // buffer data
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, size_x, size_y, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
}

//*****************************************************************************
void pboRegister()
{    
    // Transfer ownership of buffer from GL to CL
    if( bGLinterop ) {
		glFlush();
        clEnqueueAcquireGLObjects(cqCommandQueue,2, cl_pbos, 0, NULL, NULL);
	} else {
        // Explicit Copy 
        // map the PBO to copy data to the CL buffer via host
        glBindBufferARB(GL_PIXEL_PACK_BUFFER_ARB, pbo_source);    

        GLubyte* ptr = (GLubyte*)glMapBufferARB(GL_PIXEL_PACK_BUFFER_ARB,
                                            GL_READ_ONLY_ARB);

        clEnqueueWriteBuffer(cqCommandQueue, cl_pbos[0], CL_TRUE, 0, 
                            sizeof(unsigned int) * image_height * image_width, ptr, 0, NULL, NULL);
        glUnmapBufferARB(GL_PIXEL_PACK_BUFFER_ARB);
        glBindBufferARB(GL_PIXEL_PACK_BUFFER_ARB, 0);
	}
}

//*****************************************************************************
void pboUnregister()
{
        // Transfer ownership of buffer back from CL to GL
    if( bGLinterop ) {
        clEnqueueReleaseGLObjects(cqCommandQueue,2, cl_pbos, 0, NULL, NULL);
		clFinish(cqCommandQueue);
	} else {
        // Explicit Copy 
        // map the PBO to copy data from the CL buffer via host
        glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo_dest);    

        // map the buffer object into client's memory
        GLubyte* ptr = (GLubyte*)glMapBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB,
                                            GL_WRITE_ONLY_ARB);
        clEnqueueReadBuffer(cqCommandQueue, cl_pbos[1], CL_TRUE, 0, 
                            sizeof(unsigned int) * image_height * image_width, ptr, 0, NULL, NULL);        
        glUnmapBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB); 
        glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
	}
}

// Initialize GL
//*****************************************************************************
bool InitGL(int* argc, char **argv )
{
    // init GLUT and GLUT window
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_ALPHA | GLUT_DOUBLE | GLUT_DEPTH);
    glutInitWindowPosition (glutGet(GLUT_SCREEN_WIDTH)/2 - iGraphicsWinWidth/2, 
                            glutGet(GLUT_SCREEN_HEIGHT)/2 - iGraphicsWinHeight/2);
    glutInitWindowSize(iGraphicsWinWidth, iGraphicsWinHeight);
    iGLUTWindowHandle = glutCreateWindow("OpenCL/OpenGL post-processing");
#if !(defined (__APPLE__) || defined(MACOSX))
    glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS);
#endif

    // register GLUT callbacks
    glutDisplayFunc(DisplayGL);
    glutKeyboardFunc(KeyboardGL);
    glutReshapeFunc(Reshape);
    glutTimerFunc(REFRESH_DELAY, timerEvent, 0);

    // create GLUT menu
    iGLUTMenuHandle = glutCreateMenu(mainMenu);
    glutAddMenuEntry("Toggle Post-processing (Blur filter) ON/OFF <spacebar>", ' ');
    glutAddMenuEntry("Toggle Processor between GPU and CPU [p]", 'p');
    glutAddMenuEntry("Toggle GL animation (rotation) ON/OFF [a]", 'a');
    glutAddMenuEntry("Increment blur radius [+ or =]", '=');
    glutAddMenuEntry("Decrement blur radius [- or _]", '-');
    glutAddMenuEntry("Quit <esc>", '\033');
    glutAttachMenu(GLUT_RIGHT_BUTTON);

    // init GLEW
    glewInit();
    GLboolean bGLEW = glewIsSupported("GL_VERSION_2_0 GL_ARB_pixel_buffer_object"); 
    oclCheckErrorEX(bGLEW, shrTRUE, pCleanup);

    // default initialization
    glClearColor(0.5, 0.5, 0.5, 1.0);
    glDisable(GL_DEPTH_TEST);

    // viewport
    glViewport(0, 0, iGraphicsWinWidth, iGraphicsWinHeight);

    // projection
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (GLfloat)iGraphicsWinWidth / (GLfloat) iGraphicsWinHeight, 0.1, 10.0);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glEnable(GL_LIGHT0);
    float red[] = { 1.0, 0.1, 0.1, 1.0 };
    float white[] = { 1.0, 1.0, 1.0, 1.0 };
    glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, red);
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, white);
    glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 60.0);

    return true;
}

// Create PBO
//*****************************************************************************
void createPBO(GLuint* pbo)
{
    // set up data parameter
    num_texels = image_width * image_height;
    num_values = num_texels * 4;
    size_tex_data = sizeof(GLubyte) * num_values;

    // create buffer object
    glGenBuffers(1, pbo);
    glBindBuffer(GL_ARRAY_BUFFER, *pbo);

    // buffer data
    glBufferData(GL_ARRAY_BUFFER, size_tex_data, NULL, GL_DYNAMIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

// Delete PBO
//*****************************************************************************
void deletePBO(GLuint* pbo)
{
    glBindBuffer(GL_ARRAY_BUFFER, *pbo);
    glDeleteBuffers(1, pbo);

    *pbo = 0;
}

// render a simple 3D scene
//*****************************************************************************
void renderScene()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (GLfloat)iGraphicsWinWidth / (GLfloat) iGraphicsWinHeight, 0.1, 10.0);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(0.0, 0.0, -3.0);
    glRotatef(rotate[0], 1.0, 0.0, 0.0);
    glRotatef(rotate[1], 0.0, 1.0, 0.0);
    glRotatef(rotate[2], 0.0, 0.0, 1.0);

    glViewport(0, 0, iGraphicsWinWidth, iGraphicsWinHeight);

    glEnable(GL_LIGHTING);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);

    glutSolidTeapot(1.0);
}

// Init OpenCL
//*****************************************************************************
int initCL(int argc, const char** argv)
{
    cl_platform_id cpPlatform;
    cl_uint uiDevCount;
    cl_device_id *cdDevices;

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
    if(shrGetCmdLineArgumentu(argc, argv, "device", &uiDeviceUsed))
    {
      uiDeviceUsed = CLAMP(uiDeviceUsed, 0, uiEndDev);
      uiEndDev = uiDeviceUsed; 
    } 

    // Check if the requested device (or any of the devices if none requested) supports context sharing with OpenGL   
    if(bGLinterop && !bQATest)
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
        #if defined (__APPLE__) || defined (MACOSX)
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

		bGLinterop = shrFALSE;
    }

    shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    // Log device used 
    shrLog("Device # %u, ", uiDeviceUsed);
    oclPrintDevName(LOGBOTH, cdDevices[uiDeviceUsed]);
    shrLog("\n");

    // create a command-queue
    cqCommandQueue = clCreateCommandQueue(cxGPUContext, cdDevices[uiDeviceUsed], 0, &ciErrNum);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    // Memory Setup
	if( bGLinterop ) {
        cl_pbos[0] = clCreateFromGLBuffer(cxGPUContext, CL_MEM_READ_ONLY, pbo_source, &ciErrNum);
        cl_pbos[1] = clCreateFromGLBuffer(cxGPUContext, CL_MEM_WRITE_ONLY, pbo_dest, &ciErrNum);
	} else {
        cl_pbos[0] = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY, 4 * image_width * image_height, NULL, &ciErrNum);
        cl_pbos[1] = clCreateBuffer(cxGPUContext, CL_MEM_WRITE_ONLY, 4 * image_width * image_height, NULL, &ciErrNum);
	}
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    // Program Setup
    size_t program_length;
    const char* source_path = shrFindFilePath(clSourcefile, argv[0]);
    char *source = oclLoadProgSource(source_path, "", &program_length);
    oclCheckErrorEX(source != NULL, shrTRUE, pCleanup);

    // create the program
    cpProgram = clCreateProgramWithSource(cxGPUContext, 1,(const char **) &source, &program_length, &ciErrNum);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    free(source);

    // build the program
    ciErrNum = clBuildProgram(cpProgram, 0, NULL, "-cl-fast-relaxed-math", NULL, NULL);
    if (ciErrNum != CL_SUCCESS)
    {
        // write out standard error, Build Log and PTX, then cleanup and exit
        shrLogEx(LOGBOTH | ERRORMSG, ciErrNum, STDERROR);
        oclLogBuildInfo(cpProgram, oclGetFirstDev(cxGPUContext));
        oclLogPtx(cpProgram, oclGetFirstDev(cxGPUContext), "oclPostProcessGL.ptx");
        Cleanup(EXIT_FAILURE); 
    }

    // create the kernel
    ckKernel = clCreateKernel(cpProgram, "postprocess", &ciErrNum);

    // set the args values
    ciErrNum |= clSetKernelArg(ckKernel, 0, sizeof(cl_mem), (void *) &(cl_pbos[0]));
    ciErrNum |= clSetKernelArg(ckKernel, 1, sizeof(cl_mem), (void *) &(cl_pbos[1]));
    ciErrNum |= clSetKernelArg(ckKernel, 2, sizeof(image_width), &image_width);
    ciErrNum |= clSetKernelArg(ckKernel, 3, sizeof(image_width), &image_height);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    
    return 0;
}

// Kernel function
//*****************************************************************************
int executeKernel(cl_int radius)
{

    // set global and local work item dimensions
    szLocalWorkSize[0] = 16;
    szLocalWorkSize[1] = 16;
    szGlobalWorkSize[0] = shrRoundUp((int)szLocalWorkSize[0], image_width);
    szGlobalWorkSize[1] = shrRoundUp((int)szLocalWorkSize[1], image_height);

    // set the args values
    cl_int tilew =  (cl_int)szLocalWorkSize[0]+(2*radius);
    ciErrNum = clSetKernelArg(ckKernel, 4, sizeof(tilew), &tilew);
    ciErrNum |= clSetKernelArg(ckKernel, 5, sizeof(radius), &radius);    
    cl_float threshold = 0.8f;
    ciErrNum |= clSetKernelArg(ckKernel, 6, sizeof(threshold), &threshold);        
    cl_float highlight = 4.0f;
    ciErrNum |= clSetKernelArg(ckKernel, 7, sizeof(highlight), &highlight);            
    
    // Local memory
    ciErrNum |= clSetKernelArg(ckKernel, 8, (szLocalWorkSize[0]+(2*16))*(szLocalWorkSize[1]+(2*16))*sizeof(int), NULL);

    // launch computation kernel
#ifdef GPU_PROFILING
    int nIter = 30;
    for( int i=-1; i< nIter; ++i) {
        if( i ==0 )
            shrDeltaT(0);
#endif        
    ciErrNum |= clEnqueueNDRangeKernel(cqCommandQueue, ckKernel, 2, NULL,
                                      szGlobalWorkSize, szLocalWorkSize, 
                                     0, NULL, NULL);
#ifdef GPU_PROFILING
    }
    clFinish(cqCommandQueue);
    double dSeconds = shrDeltaT(0)/(double)nIter;
    double dNumTexels = (double)image_width * (double)image_height;
    double mtexps = 1.0e-6 * dNumTexels/dSeconds;
    shrLogEx(LOGBOTH | MASTER, 0, "oclPostprocessGL, Throughput = %.4f MTexels/s, Time = %.5f s, Size = %.0f Texels, NumDevsUsed = %u, Workgroup = %u\n", 
            mtexps, dSeconds, dNumTexels, uiNumDevsUsed, szLocalWorkSize[0] * szLocalWorkSize[1]);

#endif

    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    return 0;
}

// copy image and process using OpenCL
//*****************************************************************************
void processImage()
{
    // activate destination buffer
    glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, pbo_source);

    //// read data into pbo. note: use BGRA format for optimal performance
    glReadPixels(0, 0, image_width, image_height, GL_BGRA, GL_UNSIGNED_BYTE, NULL); 

    if (bPostprocess)
    {
        if (iProcFlag == 0) 
        {
            pboRegister();
            executeKernel(blur_radius);
            pboUnregister();
        } 
        else 
        {
            // map the PBOs
            glBindBufferARB(GL_PIXEL_PACK_BUFFER_ARB, pbo_source);    
            glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo_dest);    
            
            unsigned int* source_ptr = (unsigned int*)glMapBufferARB(GL_PIXEL_PACK_BUFFER_ARB,
                                                                     GL_READ_ONLY_ARB);
            
            unsigned int* dest_ptr = (unsigned int*)glMapBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB,
                                                                   GL_WRITE_ONLY_ARB);
            
            // Postprocessing on the CPU
            postprocessHost(source_ptr, dest_ptr, image_width, image_height, 0, blur_radius, 0.8f, 4.0f);
            
            // umap the PBOs
            glUnmapBufferARB(GL_PIXEL_PACK_BUFFER_ARB);
            glBindBufferARB(GL_PIXEL_PACK_BUFFER_ARB, 0);
            glUnmapBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB); 
            glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
        }

        // download texture from PBO
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo_dest);
        glBindTexture(GL_TEXTURE_2D, tex_screen);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 
                        image_width, image_height, 
                        GL_BGRA, GL_UNSIGNED_BYTE, NULL);

    } 
    else 
    {
        // download texture from PBO
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo_source);
        glBindTexture(GL_TEXTURE_2D, tex_screen);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 
                        image_width, image_height, 
                        GL_BGRA, GL_UNSIGNED_BYTE, NULL);
        
    }
}

// Helper to trigger reset of fps vars at transition 
//*****************************************************************************
void TriggerFPSUpdate()
{
    iFrameCount = 0; 
    shrDeltaT(1);
    iFramesPerSec = 1;
    iFrameTrigger = 2;
}

// Run a test sequence without any GL 
//*****************************************************************************
void TestNoGL()
{
    // execute OpenCL kernel without GL interaction
    cl_pbos[0] = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY, 4 * image_width * image_height, NULL, &ciErrNum);
    cl_pbos[1] = clCreateBuffer(cxGPUContext, CL_MEM_WRITE_ONLY, 4 * image_width * image_height, NULL, &ciErrNum);

    // set the args values
    ciErrNum |= clSetKernelArg(ckKernel, 0, sizeof(cl_mem), (void *) &(cl_pbos[0]));
    ciErrNum |= clSetKernelArg(ckKernel, 1, sizeof(cl_mem), (void *) &(cl_pbos[1]));
    ciErrNum |= clSetKernelArg(ckKernel, 2, sizeof(image_width), &image_width);
    ciErrNum |= clSetKernelArg(ckKernel, 3, sizeof(image_width), &image_height);
    
    // Start timer 0 and process n loops on the GPU 
    executeKernel(blur_radius);

    // Cleanup and exit
    Cleanup(EXIT_SUCCESS);
}

// Function to clean up and exit
//*****************************************************************************
void Cleanup(int iExitCode)
{
    // Cleanup allocated objects
    shrLog("\nStarting Cleanup...\n\n");
    if(pbo_source)deletePBO(&pbo_source);
    if(pbo_dest)deletePBO(&pbo_dest);
    if(tex_screen)deleteTexture(&tex_screen);
	if(ckKernel)clReleaseKernel(ckKernel); 
    if(cpProgram)clReleaseProgram(cpProgram);
    if(cl_pbos[0])clReleaseMemObject(cl_pbos[0]);
    if(cl_pbos[1])clReleaseMemObject(cl_pbos[1]);    
    if(cqCommandQueue)clReleaseCommandQueue(cqCommandQueue);
    if(cxGPUContext)clReleaseContext(cxGPUContext);

    shrLogEx(LOGBOTH | CLOSELOG, 0, "%s Exiting...\n", cExecutableName);

    // finalize logs and leave
    shrQAFinish2(bQATest, *pArgc, (const char **)pArgv, (iExitCode == 0) ? QA_PASSED : QA_FAILED);
    exit (iExitCode);
}
