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

// OpenGL Graphics Includes
#include <GL/glew.h>
#ifdef UNIX
    #include <GL/glxew.h>
#endif
#if defined (_WIN32)
    #include <GL/wglew.h>
#endif

#if defined (__APPLE__) || defined(MACOSX)
    #include <OpenGL/OpenGL.h>
    #include <GLUT/glut.h>
#else
    #include <GL/freeglut.h>
#endif

// Includes
#include <paramgl.h>
#include <algorithm>

// Project includes
#include "oclBodySystemOpencl.h"
#include "oclBodySystemCpu.h"
#include "oclRenderParticles.h"

// OpenCL and Shared QA Test Includes 
#include <oclUtils.h>
#include <shrQATest.h>

#ifndef min
#define min(a,b) (a < b ? a : b)
#endif

int *pArgc = NULL;
char **pArgv = NULL;

// view, GLUT and display params
int ox = 0, oy = 0;
int buttonState          = 0;
float camera_trans[]     = {0, -2, -100};
float camera_rot[]       = {0, 0, 0};
float camera_trans_lag[] = {0, -2, -100};
float camera_rot_lag[]   = {0, 0, 0};
const float inertia      = 0.1f;
ParamListGL *paramlist;      // parameter list
ParticleRenderer::DisplayMode displayMode = ParticleRenderer::PARTICLE_SPRITES_COLOR;
bool displayEnabled = true;
bool bPause = false;
bool bUsePBO = false;
bool bFullScreen = false;
bool bShowSliders = true;
int iGLUTWindowHandle;              // handle to the GLUT window
int iGraphicsWinPosX = 0;           // GLUT Window X location
int iGraphicsWinPosY = 0;           // GLUT Window Y location
int iGraphicsWinWidth = 1024;       // GLUT Window width
int iGraphicsWinHeight = 768;       // GL Window height
GLint iVsyncState;                  // state var to cache startup Vsync setting
int flopsPerInteraction = 20;

// Struct defintion for Nbody demo physical parameters
struct NBodyParams
{       
    float m_timestep;
    float m_clusterScale;
    float m_velocityScale;
    float m_softening;
    float m_damping;
    float m_pointSize;
    float m_x, m_y, m_z;

    void print()
    { 
        shrLog("{ %f, %f, %f, %f, %f, %f, %f, %f, %f },\n", 
                   m_timestep, m_clusterScale, m_velocityScale, 
                   m_softening, m_damping, m_pointSize, m_x, m_y, m_z); 
    }
};

// Array of structs of physical parameters to flip among
NBodyParams demoParams[] = 
{
    { 0.016f, 1.54f, 8.0f, 0.1f, 1.0f, 1.0f, 0, -2, -100},
    { 0.016f, 0.68f, 20.0f, 0.1f, 1.0f, 0.8f, 0, -2, -30},
    { 0.0006f, 0.16f, 1000.0f, 1.0f, 1.0f, 0.07f, 0, 0, -1.5f},
    { 0.0006f, 0.16f, 1000.0f, 1.0f, 1.0f, 0.07f, 0, 0, -1.5f},
    { 0.0019f, 0.32f, 276.0f, 1.0f, 1.0f, 0.07f, 0, 0, -5.0f},
    { 0.0016f, 0.32f, 272.0f, 0.145f, 1.0f, 0.08f, 0, 0, -5.0f},
    { 0.016f, 6.04f, 0.0f, 1.0f, 1.0f, 0.76f, 0, 0, -50.0f},
};

// Basic simulation parameters
int numBodies = 7680;               // default # of bodies in sim (can be overridden by command line switch --n=<N>)
bool bDouble = false;               //false: sp float, true: dp 
int numDemos = sizeof(demoParams) / sizeof(NBodyParams);
int activeDemo = 0;
NBodyParams activeParams = demoParams[activeDemo];
BodySystem *nbody         = 0;
BodySystemOpenCL *nbodyGPU = 0;
float* hPos = 0;
float* hVel = 0;
float* hColor = 0;
ParticleRenderer *renderer = 0;

// OpenCL vars
cl_platform_id cpPlatform;          // OpenCL Platform
cl_context cxContext;               // OpenCL Context
cl_command_queue cqCommandQueue;    // OpenCL Command Queue
cl_device_id *cdDevices = NULL;     // OpenCL device list
cl_uint uiNumDevices = 0;           // Number of OpenCL devices available
cl_uint uiNumDevsUsed = 1;          // Number of OpenCL devices used in this sample 
cl_uint uiTargetDevice = 0;	        // OpenCL Device to compute on
const char* cExecutablePath;

// Timers
#define DEMOTIME 0
#define FUNCTIME 1
#define FPSTIME 2

// fps, quick test and qatest vars
int iFrameCount = 0;                // FPS count for averaging
int iFrameTrigger = 90;             // FPS trigger for sampling
int iFramesPerSec = 60;             // frames per second
double dElapsedTime = 0.0;          // timing var to hold elapsed time in each phase of tour mode
double demoTime = 5.0;              // length of each demo phase in sec
shrBOOL bTour = shrTRUE;            // true = cycles between modes, false = stays on selected 1 mode (manually switchable)
shrBOOL bNoPrompt = shrFALSE;       // false = normal GL loop, true = Finite period of GL loop (a few seconds)
shrBOOL bQATest = shrFALSE;         // false = normal GL loop, true = run No-GL test sequence (checks against host and also does a perf test)
int iTestSets = 3;

// Forward Function declarations
//*****************************************************************************
// OpenGL (GLUT) functionality
void InitGL(int* argc, char **argv);
void DisplayGL();
void ReshapeGL(int w, int h);
void IdleGL(void);
void KeyboardGL(unsigned char key, int x, int y);
void MouseGL(int button, int state, int x, int y);
void MotionGL(int x, int y);
void SpecialGL (int key, int x, int y);

// Simulation
void ResetSim(BodySystem *system, int numBodies, NBodyConfig config, bool useGL);
void InitNbody(cl_device_id dev, cl_context ctx, cl_command_queue cmdq,
               int numBodies, int p, int q, bool bUsePBO, bool bDouble);
void SelectDemo(int index);
bool CompareResults(int numBodies);
void RunProfiling(int iterations, unsigned int uiWorkgroup);
void ComputePerfStats(double &dGigaInteractionsPerSecond, double &dGigaFlops, 
                      double dSeconds, int iterations);

// helpers
void Cleanup(int iExitCode);
void (*pCleanup)(int) = &Cleanup;
void TriggerFPSUpdate();

// Main program
//*****************************************************************************
int main(int argc, char** argv) 
{
	// Locals used with command line args
    int p = 256;            // workgroup X dimension
    int q = 1;              // workgroup Y dimension

	pArgc = &argc;
	pArgv = argv;

    shrQAStart(argc, argv);

    // latch the executable path for other funcs to use
    cExecutablePath = argv[0];

    // start logs and show command line help
	shrSetLogFileName ("oclNbody.txt");
    shrLog("%s Starting...\n\n", cExecutablePath);
    shrLog("Command line switches:\n");
	shrLog("  --qatest\t\tCheck correctness of GPU execution and measure performance)\n");
	shrLog("  --noprompt\t\tQuit simulation automatically after a brief period\n");
    shrLog("  --n=<numbodies>\tSpecify # of bodies to simulate (default = %d)\n", numBodies);
	shrLog("  --double\t\tUse double precision floating point values for simulation\n");
	shrLog("  --p=<workgroup X dim>\tSpecify X dimension of workgroup (default = %d)\n", p);
	shrLog("  --q=<workgroup Y dim>\tSpecify Y dimension of workgroup (default = %d)\n\n", q);

	// Get command line arguments if there are any and set vars accordingly
    if (argc > 0)
    {
        shrGetCmdLineArgumenti(argc, (const char**)argv, "p", &p);
        shrGetCmdLineArgumenti(argc, (const char**)argv, "q", &q);
        shrGetCmdLineArgumenti(argc, (const char**)argv, "n", &numBodies);
	    bDouble = (shrTRUE == shrCheckCmdLineFlag(argc, (const char**)argv, "double"));
        bNoPrompt = shrCheckCmdLineFlag(argc, (const char**)argv, "noprompt");
        bQATest = shrCheckCmdLineFlag(argc, (const char**)argv, "qatest");
    }

    //Get the NVIDIA platform
    cl_int ciErrNum = oclGetPlatformID(&cpPlatform);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    shrLog("clGetPlatformID...\n\n"); 
	
	if (bDouble)
	{
		shrLog("Double precision execution...\n\n");
	}
	else
	{
		shrLog("Single precision execution...\n\n");
	}

	flopsPerInteraction = bDouble ? 30 : 20; 
    
	//Get all the devices
    shrLog("Get the Device info and select Device...\n");
    ciErrNum = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_CPU, 0, NULL, &uiNumDevices);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    cdDevices = (cl_device_id *)malloc(uiNumDevices * sizeof(cl_device_id) );
    ciErrNum = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_CPU, uiNumDevices, cdDevices, NULL);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    // Set target device and Query number of compute units on uiTargetDevice
    shrLog("  # of Devices Available = %u\n", uiNumDevices); 
    if(shrGetCmdLineArgumentu(argc, (const char**)argv, "device", &uiTargetDevice)== shrTRUE) 
    {
        uiTargetDevice = CLAMP(uiTargetDevice, 0, (uiNumDevices - 1));
    }
    shrLog("  Using Device %u, ", uiTargetDevice); 
    oclPrintDevName(LOGBOTH, cdDevices[uiTargetDevice]);  
    cl_uint uiNumComputeUnits;        
    clGetDeviceInfo(cdDevices[uiTargetDevice], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(uiNumComputeUnits), &uiNumComputeUnits, NULL);
    shrLog("  # of Compute Units = %u\n", uiNumComputeUnits); 

    //Create the context
    shrLog("clCreateContext...\n"); 
    cxContext = clCreateContext(0, uiNumDevsUsed, &cdDevices[uiTargetDevice], NULL, NULL, &ciErrNum);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    // Create a command-queue 
    shrLog("CECL_CREATE_COMMAND_QUEUE...\n\n"); 
    cqCommandQueue = CECL_CREATE_COMMAND_QUEUE(cxContext, cdDevices[uiTargetDevice], CL_QUEUE_PROFILING_ENABLE, &ciErrNum);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    // Log and config for number of bodies
    shrLog("Number of Bodies = %d\n", numBodies); 
    switch (numBodies)
    {
        case 1024:
            activeParams.m_clusterScale = 1.52f;
            activeParams.m_velocityScale = 2.f;
            break;
        case 2048:
            activeParams.m_clusterScale = 1.56f;
            activeParams.m_velocityScale = 2.64f;
            break;
        case 4096:
            activeParams.m_clusterScale = 1.68f;
            activeParams.m_velocityScale = 2.98f;
            break;
        case 7680:
        case 8192:
            activeParams.m_clusterScale = 1.98f;
            activeParams.m_velocityScale = 2.9f;
            break;
        default:
        case 15360:
        case 16384:
            activeParams.m_clusterScale = 1.54f;
            activeParams.m_velocityScale = 8.f;
            break;
        case 30720:
        case 32768:
            activeParams.m_clusterScale = 1.44f;
            activeParams.m_velocityScale = 11.f;
            break;
    }

    if ((q * p) > 256)
    {
        p = 256 / q;
        shrLog("Setting p=%d to maintain %d threads per block\n", p, 256);
    }

    if ((q == 1) && (numBodies < p))
    {
        p = numBodies;
        shrLog("Setting p=%d because # of bodies < p\n", p);
    }
    shrLog("Workgroup Dims = (%d x %d)\n\n", p, q); 

    // Initialize OpenGL items if using GL 
    if (bQATest == shrFALSE)
    {
	    shrLog("Calling InitGL...\n"); 
	    InitGL(&argc, argv);
    }
    else 
    {
	    shrLog("Skipping InitGL...\n"); 
    }
	
    // CL/GL interop disabled
    bUsePBO = (false && (bQATest == shrFALSE));
    InitNbody(cdDevices[uiTargetDevice], cxContext, cqCommandQueue, numBodies, p, q, bUsePBO, bDouble);
    ResetSim(nbody, numBodies, NBODY_CONFIG_SHELL, bUsePBO);

    // init timers
    shrDeltaT(DEMOTIME); // timer 0 is for timing demo periods
    shrDeltaT(FUNCTIME); // timer 1 is for logging function delta t's
    shrDeltaT(FPSTIME);  // timer 2 is for fps measurement   

    // Standard simulation
    if (bQATest == shrFALSE)
    {
        shrLog("Running standard oclNbody simulation...\n\n"); 
        glutDisplayFunc(DisplayGL);
        glutReshapeFunc(ReshapeGL);
        glutMouseFunc(MouseGL);
        glutMotionFunc(MotionGL);
        glutKeyboardFunc(KeyboardGL);
        glutSpecialFunc(SpecialGL);
        glutIdleFunc(IdleGL);
        glutMainLoop();
    }


    // Compare to host, profile and write out file for regression analysis
    if (bQATest == shrTRUE) {
	    bool bTestResults = false;
        shrLog("Running oclNbody Results Comparison...\n\n"); 
        bTestResults = CompareResults(numBodies);

        shrLog("Profiling oclNbody...\n\n"); 
        RunProfiling(100, (unsigned int)(p * q));  // 100 iterations

		shrQAFinish(argc, (const char **)argv, bTestResults ? QA_PASSED : QA_FAILED);
    } else {
        // Cleanup/exit 
	    bNoPrompt = shrTRUE;
        shrQAFinish2(false, *pArgc, (const char **)pArgv, QA_PASSED);
    }
    Cleanup(EXIT_SUCCESS);
}

// Setup function for GLUT parameters and loop
//*****************************************************************************
void InitGL(int* argc, char **argv)
{  
    // init GLUT 
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE);
    glutInitWindowPosition (glutGet(GLUT_SCREEN_WIDTH)/2 - iGraphicsWinWidth/2, 
                            glutGet(GLUT_SCREEN_HEIGHT)/2 - iGraphicsWinHeight/2);
    glutInitWindowSize(iGraphicsWinWidth, iGraphicsWinHeight);
    iGLUTWindowHandle = glutCreateWindow("OpenCL for GPU Nbody Demo");
#if !(defined (__APPLE__) || defined(MACOSX))
    glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS);
#endif

    // init GLEW
    glewInit();
    GLboolean bGlew = glewIsSupported("GL_VERSION_2_0 "
                         "GL_VERSION_1_5 "
			             "GL_ARB_multitexture "
                         "GL_ARB_vertex_buffer_object"); 
    oclCheckErrorEX(bGlew, shrTRUE, pCleanup);
    glEnable(GL_DEPTH_TEST);
    glClearColor(0.0, 0.0, 0.0, 1.0);
    renderer = new ParticleRenderer;

    // check GL errors
    GLenum error;
    while ((error = glGetError()) != GL_NO_ERROR) 
    {
        shrLog("InitGL: error - %s\n", (char *)gluErrorString(error));
    }

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

    // create a new parameter list
    paramlist = new ParamListGL("sliders");
    paramlist->bar_col_outer[0] = 0.8f;
    paramlist->bar_col_outer[1] = 0.8f;
    paramlist->bar_col_outer[2] = 0.0f;
    paramlist->bar_col_inner[0] = 0.8f;
    paramlist->bar_col_inner[1] = 0.8f;
    paramlist->bar_col_inner[2] = 0.0f;
    
    // add parameters to the list

    // Point Size
    paramlist->AddParam(new Param<float>("Point Size", activeParams.m_pointSize, 
                    0.0f, 10.0f, 0.01f, &activeParams.m_pointSize));

    // Velocity Damping
    paramlist->AddParam(new Param<float>("Velocity Damping", activeParams.m_damping, 
                    0.5f, 1.0f, .0001f, &(activeParams.m_damping)));

    // Softening Factor
    paramlist->AddParam(new Param<float>("Softening Factor", activeParams.m_softening,
                    0.001f, 1.0f, .0001f, &(activeParams.m_softening)));

    // Time step size
    paramlist->AddParam(new Param<float>("Time Step", activeParams.m_timestep, 
                    0.0f, 1.0f, .0001f, &(activeParams.m_timestep)));

    // Cluster scale (only affects starting configuration
    paramlist->AddParam(new Param<float>("Cluster Scale", activeParams.m_clusterScale, 
                    0.0f, 10.0f, 0.01f, &(activeParams.m_clusterScale)));

    
    // Velocity scale (only affects starting configuration)
    paramlist->AddParam(new Param<float>("Velocity Scale", activeParams.m_velocityScale, 
                    0.0f, 1000.0f, 0.1f, &activeParams.m_velocityScale));
}

// Primary GLUT callback loop function
//*****************************************************************************
void DisplayGL()
{
    // update the simulation, unless paused
    double dProcessingTime = 0.0;
    if (!bPause)
    {
        // start timer FUNCTIME if it's update time
        if (iFrameCount >= iFrameTrigger)
        {
            shrDeltaT(FUNCTIME); 
        }

        // Run the simlation computations
        nbody->update(activeParams.m_timestep); 
        nbody->getArray(BodySystem::BODYSYSTEM_POSITION);

        // Make graphics work with or without CL/GL interop 
        if (bUsePBO) 
        {
            renderer->setPBO((unsigned int)nbody->getCurrentReadBuffer(), nbody->getNumBodies());
        } 
        else 
        { 
            renderer->setPositions((float*)nbody->getCurrentReadBuffer(), nbody->getNumBodies());
        }

        // get processing time from timer FUNCTIME, if it's update time
        if (iFrameCount >= iFrameTrigger)
        {
            dProcessingTime = shrDeltaT(FUNCTIME); 
        }
    }

    // Redraw main graphics display, if enabled
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);  
    if (displayEnabled)
    {
        // view transform
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        for (int c = 0; c < 3; ++c)
        {
            camera_trans_lag[c] += (camera_trans[c] - camera_trans_lag[c]) * inertia;
            camera_rot_lag[c] += (camera_rot[c] - camera_rot_lag[c]) * inertia;
        }
        glTranslatef(camera_trans_lag[0], camera_trans_lag[1], camera_trans_lag[2]);
        glRotatef(camera_rot_lag[0], 1.0, 0.0, 0.0);
        glRotatef(camera_rot_lag[1], 0.0, 1.0, 0.0);
        renderer->setSpriteSize(activeParams.m_pointSize);
        renderer->display(displayMode);
    }

    // Display user interface if enabled
    if (bShowSliders)
    {
        glBlendFunc(GL_ONE_MINUS_DST_COLOR, GL_ZERO); // invert color
        glEnable(GL_BLEND);
	    paramlist->Render(0, 0);
        glDisable(GL_BLEND);
    }

    // Flip backbuffer to screen 
    glutSwapBuffers();

    //  If frame count has triggerd, increment the frame counter, and do fps stuff 
    if (iFrameCount++ > iFrameTrigger)
    {
        // If tour mode is enabled & interval has timed out, switch to next tour/demo mode
        dElapsedTime += shrDeltaT(DEMOTIME); 
        if (bTour && (dElapsedTime > demoTime))
        {
            dElapsedTime = 0.0;
            activeDemo = (activeDemo + 1) % numDemos;
            SelectDemo(activeDemo);
        }

        // get the perf and fps stats
        iFramesPerSec = (int)((double)iFrameCount/ shrDeltaT(FPSTIME));
        double dGigaInteractionsPerSecond = 0.0;
        double dGigaFlops = 0.0;
        ComputePerfStats(dGigaInteractionsPerSecond, dGigaFlops, dProcessingTime, 1);

        // If not paused, set the display window title, reset trigger and log info
        char cTitle[256];
        if(!bPause) 
        {
        #ifdef GPU_PROFILING
            #ifdef _WIN32
                sprintf_s(cTitle, 
                        "OpenCL for GPU Nbody Demo (%d bodies): %i fps | %0.4f BIPS | %0.4f GFLOP/s", 
		                numBodies, iFramesPerSec, dGigaInteractionsPerSecond, dGigaFlops);  
            #else 
                sprintf(cTitle, 
                        "OpenCL for GPU Nbody Demo (%d bodies): %i fps | %0.4f BIPS | %0.4f GFLOP/s", 
		                numBodies, iFramesPerSec, dGigaInteractionsPerSecond, dGigaFlops);  
            #endif
        #else
            #ifdef _WIN32
                sprintf_s(cTitle, 
                        "OpenCL for GPU Nbody Demo (%d bodies)", 
		                numBodies, iFramesPerSec, dGigaInteractionsPerSecond);  
            #else 
                sprintf(cTitle, 
                        "OpenCL for GPU Nbody Demo (%d bodies)", 
		                numBodies, iFramesPerSec, dGigaInteractionsPerSecond);  
            #endif
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

            // reset the frame counter and adjust trigger
            iFrameCount = 0; 
            iFrameTrigger = (iFramesPerSec > 1) ? iFramesPerSec * 2 : 1;
        }
    }

    glutReportErrors();
}

// GLUT key event handler
//*****************************************************************************
void KeyboardGL(unsigned char key, int /*x*/, int /*y*/)
{
    switch (key) 
    {
        case ' ': // space toggle computation flag on/off
            bPause = !bPause;
            shrLog("\nSim %s...\n\n", bPause ? "Paused" : "Running");
            break;
        case '`':   // Tilda toggles slider display
            bShowSliders = !bShowSliders;
            shrLog("\nSlider Display %s...\n\n", bShowSliders ? "ON" : "OFF");
            break;
        case 'p':   // 'p' falls through to 'P' 
        case 'P':   // p switched between points and blobs 
            displayMode = (ParticleRenderer::DisplayMode)((displayMode + 1) % ParticleRenderer::PARTICLE_NUM_MODES);
            break;
        case 'c':   // 'c' falls through to 'C'
        case 'C':   // c switches between cycle demo mode and fixed demo mode
            bTour = bTour ? shrFALSE : shrTRUE;
            shrLog("\nTour Mode %s...\n\n", bTour ? "ON" : "OFF");
            break;
        case '[':
            activeDemo = (activeDemo == 0) ? numDemos - 1 : (activeDemo - 1) % numDemos;
            SelectDemo(activeDemo);
            break;
        case ']':
            activeDemo = (activeDemo + 1) % numDemos;
            SelectDemo(activeDemo);
            break;
        case 'd':   // 'd' falls through to 'D'
        case 'D':   // d toggled main graphics display on/off
            displayEnabled = !displayEnabled;
            shrLog("\nMain Graphics Display %s...\n\n", displayEnabled ? "ON" : "OFF");
            break;
        case 'f':   // 'f' falls through to 'F'
        case 'F':   // f toggles main graphics display full screen
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
            shrLog("\nMain Graphics %s...\n\n", bFullScreen ? "FullScreen" : "Windowed");
            break;
        case 'o':   // 'o' falls through to 'O'
        case 'O':   // 'O' prints Nbody sim physical parameters
            activeParams.print();
            break;
        case 'T':   // Toggles from (T)our mode to standard mode and back
        case 't':   // Toggles from (t)our mode to standard mode and back
            bTour = bTour ? shrFALSE : shrTRUE;
            shrLog("\nTour Mode %s...\n", bTour ? "ON" : "OFF");
            break;
        case '1':
            ResetSim(nbody, numBodies, NBODY_CONFIG_SHELL, true);
            break;
        case '2':
            ResetSim(nbody, numBodies, NBODY_CONFIG_RANDOM, true);
            break;
        case '3':
            ResetSim(nbody, numBodies, NBODY_CONFIG_EXPAND, true);
            break;
        case '\033': // escape quits
        case '\015': // Enter quits    
        case 'Q':    // Q quits
        case 'q':    // q (or escape) quits
            // Cleanup and quit
            bNoPrompt = shrTRUE;
            shrQAFinish2(false, *pArgc, (const char **)pArgv, QA_PASSED);
            Cleanup(EXIT_SUCCESS);
            break;
    }

    // Trigger fps update and call for refresh
    TriggerFPSUpdate();
    glutPostRedisplay();
}

//*****************************************************************************
void RunProfiling(int iterations, unsigned int uiWorkgroup)
{
    // once without timing to prime the GPU
    nbody->update(activeParams.m_timestep);
    nbody->synchronizeThreads();

	// Start timer 0 and process n loops on the GPU
    shrDeltaT(FUNCTIME);
    for (int i = 0; i < iterations; ++i)
    {
        nbody->update(activeParams.m_timestep);
    }
    nbody->synchronizeThreads();

    // Get elapsed time and throughput, then log to sample and master logs
    double dSeconds = shrDeltaT(FUNCTIME);
    double dGigaInteractionsPerSecond = 0.0;
    double dGigaFlops = 0.0;
    ComputePerfStats(dGigaInteractionsPerSecond, dGigaFlops, dSeconds, iterations);
    shrLogEx(LOGBOTH | MASTER, 0, "oclNBody-%s, Throughput = %.4f GFLOP/s, Time = %.5f s, Size = %u bodies, NumDevsUsed = %u, Workgroup = %u\n", 
        (bDouble ? "DP" : "SP"), dGigaFlops, dSeconds/(double)iterations, numBodies, uiNumDevsUsed, uiWorkgroup); 
}

// Handler for GLUT window resize event
//*****************************************************************************
void ReshapeGL(int w, int h)
{
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (float) w / (float) h, 0.1, 1000.0);
    glMatrixMode(GL_MODELVIEW);
    glViewport(0, 0, w, h);
}

// Handler for GLU Mouse events
//*****************************************************************************
void MouseGL(int button, int state, int x, int y)
{
    if (bShowSliders) 
    {
	    // call list mouse function
        if (paramlist->Mouse(x, y, button, state))
        {
            nbody->setSoftening(activeParams.m_softening);
            nbody->setDamping(activeParams.m_damping);
        }
    }
    
    int mods;

    if (state == GLUT_DOWN)
    {
        buttonState |= 1<<button;
    }
    else if (state == GLUT_UP)
    {
        buttonState = 0;
    }

    mods = glutGetModifiers();
    if (mods & GLUT_ACTIVE_SHIFT) 
    {
        buttonState = 2;
    } 
    else if (mods & GLUT_ACTIVE_CTRL) 
    {
        buttonState = 3;
    }

    ox = x; 
    oy = y;

    glutPostRedisplay();
}

//*****************************************************************************
void MotionGL(int x, int y)
{
    if (bShowSliders) 
    {
        // call parameter list motion function
        if (paramlist->Motion(x, y))
	    {
            nbody->setSoftening(activeParams.m_softening);
            nbody->setDamping(activeParams.m_damping);
            glutPostRedisplay();
	        return;
        }
    }

    float dx = (float)(x - ox);
    float dy = (float)(y - oy);

    if (buttonState == 3) 
    {
        // left+middle = zoom
        camera_trans[2] += (dy * 0.01f) * 0.5f * fabs(camera_trans[2]);
    } 
    else if (buttonState & 2) 
    {
        // middle = translate
        camera_trans[0] += (dx * 0.01f);
        camera_trans[1] -= (dy * 0.01f);
    }
    else if (buttonState & 1) 
    {
        // left = rotate
        camera_rot[0] += (dy * 0.20f);
        camera_rot[1] += (dx * 0.20f);
    }
    
    ox = x; 
    oy = y;
    glutPostRedisplay();
}

//*****************************************************************************
void SpecialGL(int key, int x, int y)
{
    paramlist->Special(key, x, y);
    glutPostRedisplay();
}

//*****************************************************************************
void IdleGL(void)
{
    glutPostRedisplay();
}

// Helper to trigger reset of fps vars at transition 
//*****************************************************************************
void TriggerFPSUpdate()
{
    iFrameCount = 0; 
    shrDeltaT(FPSTIME);
    iFramesPerSec = 1;
    iFrameTrigger = 2;
}

//*****************************************************************************
void ResetSim(BodySystem *system, int numBodies, NBodyConfig config, bool useGL)
{
    shrLog("\nReset Nbody System...\n\n");

    // initalize the memory
    randomizeBodies(config, hPos, hVel, hColor, activeParams.m_clusterScale, 
		            activeParams.m_velocityScale, numBodies);

    system->setArray(BodySystem::BODYSYSTEM_POSITION, hPos);
    system->setArray(BodySystem::BODYSYSTEM_VELOCITY, hVel);
    if (useGL)
    {
        renderer->setColors(hColor, nbody->getNumBodies());
        renderer->setSpriteSize(activeParams.m_pointSize);
    }
}

//*****************************************************************************
void InitNbody(cl_device_id dev, cl_context ctx, cl_command_queue cmdq,
               int numBodies, int p, int q, bool bUsePBO, bool bDouble)
{
    // New nbody system for Device/GPU computations
    nbodyGPU = new BodySystemOpenCL(numBodies, dev, ctx, cmdq, p, q, bUsePBO, bDouble);
    nbody = nbodyGPU;

    // allocate host memory
    hPos = new float[numBodies*4];
    hVel = new float[numBodies*4];
    hColor = new float[numBodies*4];

    // Set sim parameters
    nbody->setSoftening(activeParams.m_softening);
    nbody->setDamping(activeParams.m_damping);
}

//*****************************************************************************
void SelectDemo(int index)
{
    oclCheckErrorEX((index < numDemos), shrTRUE, pCleanup);

    activeParams = demoParams[index];
    camera_trans[0] = camera_trans_lag[0] = activeParams.m_x;
    camera_trans[1] = camera_trans_lag[1] = activeParams.m_y;
    camera_trans[2] = camera_trans_lag[2] = activeParams.m_z;
    ResetSim(nbody, numBodies, NBODY_CONFIG_SHELL, true);

    //Rest the demo timer
    shrDeltaT(DEMOTIME);
}

//*****************************************************************************
bool CompareResults(int numBodies)
{
    // Run computation on the device/GPU
    shrLog("  Computing on the Device / GPU...\n");
    nbodyGPU->update(0.001f);
    nbodyGPU->synchronizeThreads();

    // Write out device/GPU data file for regression analysis
    shrLog("  Writing out Device/GPU data file for analysis...\n");
    float* fGPUData = nbodyGPU->getArray(BodySystem::BODYSYSTEM_POSITION);
    shrWriteFilef( "oclNbody_Regression.dat", fGPUData, numBodies, 0.0, false);

    // Run computation on the host CPU
    shrLog("  Computing on the Host / CPU...\n\n");
    BodySystemCPU* nbodyCPU = new BodySystemCPU(numBodies);
    nbodyCPU->setArray(BodySystem::BODYSYSTEM_POSITION, hPos);
    nbodyCPU->setArray(BodySystem::BODYSYSTEM_VELOCITY, hVel);
    nbodyCPU->update(0.001f);

    // Check if result matches 
    shrBOOL bMatch = shrComparefe(fGPUData, 
                        nbodyGPU->getArray(BodySystem::BODYSYSTEM_POSITION), 
						numBodies, .001f);
    shrLog("Results %s\n\n", (shrTRUE == bMatch) ? "Match" : "do not match!");

    // Cleanup local allocation
    if(nbodyCPU)delete nbodyCPU; 

    return (shrTRUE == bMatch);
}

//*****************************************************************************
void ComputePerfStats(double &dGigaInteractionsPerSecond, double &dGigaFlops, double dSeconds, int iterations)
{
	//int flopsPerInteraction = bDouble ? 30 : 20; 
    dGigaInteractionsPerSecond = 1.0e-9 * (double)numBodies * (double)numBodies * (double)iterations / dSeconds;
    dGigaFlops = dGigaInteractionsPerSecond * (float)flopsPerInteraction;	
}

// Helper to clean up
//*****************************************************************************
void Cleanup(int iExitCode)
{
//    shrLog("\nStarting Cleanup...\n\n");

    // Cleanup allocated objects
    if(nbodyGPU)delete nbodyGPU;
    if(cqCommandQueue)clReleaseCommandQueue(cqCommandQueue);
    if(cxContext)clReleaseContext(cxContext);
    if(hPos)delete [] hPos;
    if(hVel)delete [] hVel;
    if(hColor)delete [] hColor;
    if(renderer)delete renderer;

    // finalize logs and leave
    if (bNoPrompt || bQATest)
    {
//        shrLogEx(LOGBOTH | CLOSELOG, 0, "%s Exiting...\n", cExecutablePath);
    }
    else 
    {
        shrLogEx(LOGBOTH | CLOSELOG, 0, "%s Exiting...\nPress <Enter> to Quit\n", cExecutablePath);
        #ifdef WIN32
            getchar();
        #endif
    }
    exit (iExitCode);
}
