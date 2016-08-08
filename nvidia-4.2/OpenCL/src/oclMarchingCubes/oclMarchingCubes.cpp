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
    Marching cubes

    This sample extracts a geometric isosurface from a volume dataset using
    the marching cubes algorithm. It uses the scan (prefix sum) function 
    from the SDK sample oclScan to perform stream compaction. Similar techniques can
    be used for other problems that require a variable-sized output per
    thread.

    For more information on marching cubes see:
    http://local.wasp.uwa.edu.au/~pbourke/geometry/polygonise/
    http://en.wikipedia.org/wiki/Marching_cubes

    Volume data courtesy:
    http://www9.informatik.uni-erlangen.de/External/vollib/


    The algorithm consists of several stages:

    1. Execute "classifyVoxel" kernel
    This evaluates the volume at the corners of each voxel and computes the
    number of vertices each voxel will generate.
    It is executed using one thread per voxel.
    It writes two arrays - voxelOccupied and voxelVertices to global memory.
    voxelOccupied is a flag indicating if the voxel is non-empty.

    2. Scan "voxelOccupied" array 
    Read back the total number of occupied voxels from GPU to CPU.
    This is the sum of the last value of the exclusive scan and the last
    input value.

    3. Execute "compactVoxels" kernel
    This compacts the voxelOccupied array to get rid of empty voxels.
    This allows us to run the complex "generateTriangles" kernel on only
    the occupied voxels.

    4. Scan voxelVertices array
    This gives the start address for the vertex data for each voxel.
    We read back the total number of vertices generated from GPU to CPU.

    Note that by using a custom scan function we could combine the above two
    scan operations above into a single operation.

    5. Execute "generateTriangles" kernel
    This runs only on the occupied voxels.
    It looks up the field values again and generates the triangle data,
    using the results of the scan to write the output to the correct addresses.
    The marching cubes look-up tables are stored in 1D textures.

    6. Render geometry
    Using number of vertices from readback.
*/
// OpenGL Graphics includes
#include <GL/glew.h>
#if defined(__APPLE__) || defined(__MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif

#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

// includes
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <memory>
#include <iostream>
#include <cassert>

#include "defines.h"
#include "tables.h"

// standard utility and system includes
#include <oclUtils.h>
#include <shrQATest.h>

// CL/GL includes and defines
#include <CL/cl_gl.h>    

#ifdef UNIX
	#if defined(__APPLE__) || defined(MACOSX)
	    #include <OpenCL/opencl.h>
	    #include <OpenGL/OpenGL.h>
	    #include <GLUT/glut.h>
	    #include <OpenCL/cl_gl_ext.h>
	#else
	    #include <GL/freeglut.h>
	    #include <GL/glx.h>
	#endif
#endif


#if defined (__APPLE__) || defined(MACOSX)
   #define GL_SHARING_EXTENSION "cl_APPLE_gl_sharing"
#else
   #define GL_SHARING_EXTENSION "cl_khr_gl_sharing"
#endif

#define REFRESH_DELAY	  10 //ms

#include "oclScan_common.h"

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
cl_kernel classifyVoxelKernel;
cl_kernel compactVoxelsKernel;
cl_kernel generateTriangles2Kernel;
cl_int ciErrNum;
char* cPathAndName = NULL;          // var for full paths to data, src, etc.
char* cSourceCL;                    // Buffer to hold source for compilation 
cl_bool g_glInterop = false;

int *pArgc = NULL;
char **pArgv = NULL;

class dim3 {
public:
    size_t x;
    size_t y;
    size_t z;

    dim3(size_t _x=1, size_t _y=1, size_t _z=1) {x = _x; y = _y; z = _z;}
};


// constants
const unsigned int window_width = 512;
const unsigned int window_height = 512;

const char *volumeFilename = "Bucky.raw";

cl_uint gridSizeLog2[4] = {5, 5, 5,0};
cl_uint gridSizeShift[4];
cl_uint gridSize[4];
cl_uint gridSizeMask[4];

cl_float voxelSize[4];
uint numVoxels    = 0;
uint maxVerts     = 0;
uint activeVoxels = 0;
uint totalVerts   = 0;

float isoValue		= 0.2f;
float dIsoValue		= 0.005f;

// device data
GLuint posVbo, normalVbo;

GLint  gl_Shader;

cl_mem d_pos = 0;
cl_mem d_normal = 0;

cl_mem d_volume = 0;
cl_mem d_voxelVerts = 0;
cl_mem d_voxelVertsScan = 0;
cl_mem d_voxelOccupied = 0;
cl_mem d_voxelOccupiedScan = 0;
cl_mem d_compVoxelArray;

// tables
cl_mem d_numVertsTable = 0;
cl_mem d_triTable = 0;

// mouse controls
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
cl_float rotate[4] = {0.0, 0.0, 0.0, 0.0};
cl_float translate[4] = {0.0, 0.0, -3.0, 0.0};

// toggles
bool wireframe = false;
bool animate = true;
bool lighting = true;
bool render = true;
bool compute = true;

void Cleanup(int iExitCode);
void (*pCleanup)(int) = &Cleanup;

double totalTime = 0;

// Auto-Verification Code
const int frameCheckNumber = 4;
unsigned int fpsLimit = 100;        // FPS limit for sampling
int g_Index = 0;
unsigned int frameCount = 0;
unsigned int g_TotalErrors = 0;
bool g_bNoprompt = false;
bool bQATest = false;	
const char* cpExecutableName;

// forward declarations
void runTest(int argc, char** argv);
void initMC(int argc, char** argv);
void computeIsosurface();

bool initGL(int argc, char **argv);
void createVBO(GLuint* vbo, unsigned int size, cl_mem &vbo_cl);
void deleteVBO(GLuint* vbo, cl_mem vbo_cl );

void display();
void keyboard(unsigned char key, int x, int y);
void mouse(int button, int state, int x, int y);
void motion(int x, int y);
void timerEvent(int value);

void idle();
void reshape(int w, int h);
void TestNoGL();

template <class T>
void dumpBuffer(cl_mem d_buffer, T *h_buffer, int nelements);


void mainMenu(int i);

void allocateTextures(	cl_mem *d_triTable, cl_mem* d_numVertsTable )
{
    cl_image_format imageFormat;
    imageFormat.image_channel_order = CL_R;
    imageFormat.image_channel_data_type = CL_UNSIGNED_INT8;
    

    *d_triTable = clCreateImage2D(cxGPUContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  &imageFormat,
                                  16,256,0, (void*) triTable, &ciErrNum );
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    

    *d_numVertsTable = clCreateImage2D(cxGPUContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       &imageFormat,
                                       256,1,0, (void*) numVertsTable, &ciErrNum );
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    
}

void
openclScan(cl_mem d_voxelOccupiedScan, cl_mem d_voxelOccupied, int numVoxels) {
    scanExclusiveLarge(
                       cqCommandQueue,
                       d_voxelOccupiedScan,
                       d_voxelOccupied,
                       1,
                       numVoxels);
    
    
}

void
launch_classifyVoxel( dim3 grid, dim3 threads, cl_mem voxelVerts, cl_mem voxelOccupied, cl_mem volume,
					  cl_uint gridSize[4], cl_uint gridSizeShift[4], cl_uint gridSizeMask[4], uint numVoxels,
					  cl_float voxelSize[4], float isoValue)
{
    ciErrNum = CECL_SET_KERNEL_ARG(classifyVoxelKernel, 0, sizeof(cl_mem), &voxelVerts);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    ciErrNum = CECL_SET_KERNEL_ARG(classifyVoxelKernel, 1, sizeof(cl_mem), &voxelOccupied);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    ciErrNum = CECL_SET_KERNEL_ARG(classifyVoxelKernel, 2, sizeof(cl_mem), &volume);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    ciErrNum = CECL_SET_KERNEL_ARG(classifyVoxelKernel, 3, 4 * sizeof(cl_uint), gridSize);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    ciErrNum = CECL_SET_KERNEL_ARG(classifyVoxelKernel, 4, 4 * sizeof(cl_uint), gridSizeShift);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    ciErrNum = CECL_SET_KERNEL_ARG(classifyVoxelKernel, 5, 4 * sizeof(cl_uint), gridSizeMask);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    ciErrNum = CECL_SET_KERNEL_ARG(classifyVoxelKernel, 6, sizeof(uint), &numVoxels);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    ciErrNum = CECL_SET_KERNEL_ARG(classifyVoxelKernel, 7, 4 * sizeof(cl_float), voxelSize);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    ciErrNum = CECL_SET_KERNEL_ARG(classifyVoxelKernel, 8, sizeof(float), &isoValue);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    ciErrNum = CECL_SET_KERNEL_ARG(classifyVoxelKernel, 9, sizeof(cl_mem), &d_numVertsTable);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    grid.x *= threads.x;
    ciErrNum = CECL_ND_RANGE_KERNEL(cqCommandQueue, classifyVoxelKernel, 1, NULL, (size_t*) &grid, (size_t*) &threads, 0, 0, 0);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
}

void
launch_compactVoxels(dim3 grid, dim3 threads, cl_mem compVoxelArray, cl_mem voxelOccupied, cl_mem voxelOccupiedScan, uint numVoxels)
{
    ciErrNum = CECL_SET_KERNEL_ARG(compactVoxelsKernel, 0, sizeof(cl_mem), &compVoxelArray);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    ciErrNum = CECL_SET_KERNEL_ARG(compactVoxelsKernel, 1, sizeof(cl_mem), &voxelOccupied);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    ciErrNum = CECL_SET_KERNEL_ARG(compactVoxelsKernel, 2, sizeof(cl_mem), &voxelOccupiedScan);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    ciErrNum = CECL_SET_KERNEL_ARG(compactVoxelsKernel, 3, sizeof(cl_uint), &numVoxels);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    
    grid.x *= threads.x;
    ciErrNum = CECL_ND_RANGE_KERNEL(cqCommandQueue, compactVoxelsKernel, 1, NULL, (size_t*) &grid, (size_t*) &threads, 0, 0, 0);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
}

void
launch_generateTriangles2(dim3 grid, dim3 threads,
                          cl_mem pos, cl_mem norm, cl_mem compactedVoxelArray, cl_mem numVertsScanned, cl_mem volume,
                          cl_uint gridSize[4], cl_uint gridSizeShift[4], cl_uint gridSizeMask[4],
                          cl_float voxelSize[4], float isoValue, uint activeVoxels, uint maxVerts)
{
    ciErrNum = CECL_SET_KERNEL_ARG(generateTriangles2Kernel, 0, sizeof(cl_mem), &pos);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    ciErrNum = CECL_SET_KERNEL_ARG(generateTriangles2Kernel, 1, sizeof(cl_mem), &norm);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    ciErrNum = CECL_SET_KERNEL_ARG(generateTriangles2Kernel, 2, sizeof(cl_mem), &compactedVoxelArray);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    ciErrNum = CECL_SET_KERNEL_ARG(generateTriangles2Kernel, 3, sizeof(cl_mem), &numVertsScanned);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    ciErrNum = CECL_SET_KERNEL_ARG(generateTriangles2Kernel, 4, sizeof(cl_mem), &volume);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS,  pCleanup); 
    ciErrNum = CECL_SET_KERNEL_ARG(generateTriangles2Kernel, 5, 4 * sizeof(cl_uint), gridSize);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    ciErrNum = CECL_SET_KERNEL_ARG(generateTriangles2Kernel, 6, 4 * sizeof(cl_uint), gridSizeShift);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    ciErrNum = CECL_SET_KERNEL_ARG(generateTriangles2Kernel, 7, 4 * sizeof(cl_uint), gridSizeMask);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    ciErrNum = CECL_SET_KERNEL_ARG(generateTriangles2Kernel, 8, 4 * sizeof(cl_float), voxelSize);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    ciErrNum = CECL_SET_KERNEL_ARG(generateTriangles2Kernel, 9, sizeof(float), &isoValue);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    ciErrNum = CECL_SET_KERNEL_ARG(generateTriangles2Kernel, 10, sizeof(uint), &activeVoxels);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    ciErrNum = CECL_SET_KERNEL_ARG(generateTriangles2Kernel, 11, sizeof(uint), &maxVerts);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    ciErrNum = CECL_SET_KERNEL_ARG(generateTriangles2Kernel, 12, sizeof(cl_mem), &d_numVertsTable);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    ciErrNum = CECL_SET_KERNEL_ARG(generateTriangles2Kernel, 13, sizeof(cl_mem), &d_triTable);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    grid.x *= threads.x;
    ciErrNum = CECL_ND_RANGE_KERNEL(cqCommandQueue, generateTriangles2Kernel, 1, NULL, (size_t*) &grid, (size_t*) &threads, 0, 0, 0);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

}

void animation()
{
    if (animate) {
        isoValue += dIsoValue;
        if (isoValue < 0.1f ) {
            isoValue = 0.1f;
            dIsoValue *= -1.0f;
        } else if ( isoValue > 0.9f ) {
            isoValue = 0.9f;
            dIsoValue *= -1.0f;
        }
    }
}

void timerEvent(int value)
{
    animation();
    glutPostRedisplay();
    glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
}

void computeFPS()
{
    frameCount++;
 
    if (frameCount == fpsLimit) {
        char fps[256];
        float ifps = (float)frameCount / (float)(totalTime);
        sprintf(fps, "CUDA Marching Cubes: %3.1f fps", ifps);  
        
        glutSetWindowTitle(fps);
        
        frameCount = 0;
        totalTime = 0.0;

        if( g_bNoprompt) Cleanup(EXIT_SUCCESS);        
    }
}


////////////////////////////////////////////////////////////////////////////////
// Load raw data from disk
////////////////////////////////////////////////////////////////////////////////
uchar *loadRawFile(char *filename, int size)
{
	FILE *fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "Error opening file '%s'\n", filename);
        return 0;
    }

	uchar *data = (uchar *) malloc(size);
	size_t read = fread(data, 1, size, fp);
	fclose(fp);

    printf("Read '%s', %d bytes\n", filename, (int)read);

    return data;
}


void initCL(int argc, char** argv) {
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
    if(shrGetCmdLineArgumentu(argc, (const char**)argv, "device", &uiDeviceUsed ))
    {
      uiDeviceUsed = CLAMP(uiDeviceUsed, 0, uiEndDev);
      uiEndDev = uiDeviceUsed; 
    } 

	// Check if the requested device (or any of the devices if none requested) supports context sharing with OpenGL
    if(g_glInterop)
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
		// No GL interop
        cl_context_properties props[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)cpPlatform, 0};
        cxGPUContext = clCreateContext(props, 1, &cdDevices[uiDeviceUsed], NULL, NULL, &ciErrNum);

		g_glInterop = false;
    }

    oclPrintDevInfo(LOGBOTH, cdDevices[uiDeviceUsed]);

    // create a command-queue
    cqCommandQueue = CECL_CREATE_COMMAND_QUEUE(cxGPUContext, cdDevices[uiDeviceUsed], 0, &ciErrNum);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    // Program Setup
    size_t program_length;
    cPathAndName = shrFindFilePath("marchingCubes_kernel.cl", argv[0]);
    oclCheckErrorEX(cPathAndName != NULL, shrTRUE, pCleanup);
    cSourceCL = oclLoadProgSource(cPathAndName, "", &program_length);
    oclCheckErrorEX(cSourceCL != NULL, shrTRUE, pCleanup);

    // create the program
    cpProgram = CECL_PROGRAM_WITH_SOURCE(cxGPUContext, 1,
					  (const char **)&cSourceCL, &program_length, &ciErrNum);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    
    // build the program
    std::string buildOpts = "-cl-mad-enable";
    ciErrNum = CECL_PROGRAM(cpProgram, 0, NULL, buildOpts.c_str(), NULL, NULL);
    if (ciErrNum != CL_SUCCESS)
    {
        // write out standard error, Build Log and PTX, then cleanup and return error
        shrLogEx(LOGBOTH | ERRORMSG, ciErrNum, STDERROR);
        oclLogBuildInfo(cpProgram, oclGetFirstDev(cxGPUContext));
        oclLogPtx(cpProgram, oclGetFirstDev(cxGPUContext), "oclMarchinCubes.ptx");
        Cleanup(EXIT_FAILURE); 
    }

    // create the kernel
    classifyVoxelKernel = CECL_KERNEL(cpProgram, "classifyVoxel", &ciErrNum);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    compactVoxelsKernel = CECL_KERNEL(cpProgram, "compactVoxels", &ciErrNum);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    generateTriangles2Kernel = CECL_KERNEL(cpProgram, "generateTriangles2", &ciErrNum);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    // Setup Scan
    initScan(cxGPUContext, cqCommandQueue, (const char**)argv);
}


////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main(int argc, char** argv)
{
    pArgc = &argc;
    pArgv = argv;

    shrQAStart(argc, argv);

    cpExecutableName = argv[0];
    shrSetLogFileName ("oclMarchingCubes.txt");
    shrLog("%s Starting...\n\n", argv[0]); 

    if (shrCheckCmdLineFlag(argc, (const char **)argv, "noprompt") ) 
    {
        g_bNoprompt = true;
    }

    if (shrCheckCmdLineFlag(argc, (const char **)argv, "qatest") ) {    
        bQATest = true;
	animate = false;
    }

    runTest(argc, argv);

    Cleanup(EXIT_SUCCESS);
}

////////////////////////////////////////////////////////////////////////////////
// initialize marching cubes
////////////////////////////////////////////////////////////////////////////////
void
initMC(int argc, char** argv)
{
    // parse command line arguments
    int n;
    if (shrGetCmdLineArgumenti( argc, (const char**) argv, "grid", &n)) {
        gridSizeLog2[0] = gridSizeLog2[1] = gridSizeLog2[2] = n;
    }
    if (shrGetCmdLineArgumenti( argc, (const char**) argv, "gridx", &n)) {
        gridSizeLog2[0] = n;
    }
    if (shrGetCmdLineArgumenti( argc, (const char**) argv, "gridy", &n)) {
        gridSizeLog2[1] = n;
    }
    if (shrGetCmdLineArgumenti( argc, (const char**) argv, "gridz", &n)) {
        gridSizeLog2[2] = n;
    }

    char *filename;
    if (shrGetCmdLineArgumentstr( argc, (const char**) argv, "file", &filename)) {
        volumeFilename = filename;
    }

    gridSize[0] = 1<<gridSizeLog2[0];
    gridSize[1] = 1<<gridSizeLog2[1];
    gridSize[2] = 1<<gridSizeLog2[2];


    gridSizeMask[0] = gridSize[0]-1;
    gridSizeMask[1] = gridSize[1]-1;
    gridSizeMask[2] = gridSize[2]-1;

    gridSizeShift[0] = 0;
    gridSizeShift[1] = gridSizeLog2[0];
    gridSizeShift[2] = gridSizeLog2[0]+gridSizeLog2[1];

    numVoxels = gridSize[0]*gridSize[1]*gridSize[2];

    
    voxelSize[0] = 2.0f / gridSize[0];
    voxelSize[1] = 2.0f / gridSize[1];
    voxelSize[2] = 2.0f / gridSize[2];

    maxVerts = gridSize[0]*gridSize[1]*100;

    shrLog("grid: %d x %d x %d = %d voxels\n", gridSize[0], gridSize[1], gridSize[2], numVoxels);
    shrLog("max verts = %d\n", maxVerts);

    // load volume data
    char* path = shrFindFilePath(volumeFilename, argv[0]);
    if (path == 0) {
        shrLog("Error finding file '%s'\n", volumeFilename);
        exit(EXIT_FAILURE);
    }

    int size = gridSize[0]*gridSize[1]*gridSize[2]*sizeof(uchar);
    uchar *volume = loadRawFile(path, size);
    cl_image_format volumeFormat;
    volumeFormat.image_channel_order = CL_R;
    volumeFormat.image_channel_data_type = CL_UNORM_INT8;

    d_volume = clCreateImage3D(cxGPUContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, &volumeFormat, 
                                    gridSize[0], gridSize[1], gridSize[2],
                                    gridSize[0], gridSize[0] * gridSize[1],
                                    volume, &ciErrNum);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    free(volume);

    // create VBOs
    if( !bQATest) {
        createVBO(&posVbo, maxVerts*sizeof(float)*4, d_pos);
        createVBO(&normalVbo, maxVerts*sizeof(float)*4, d_normal);
    }
    
    // allocate textures
	allocateTextures(&d_triTable, &d_numVertsTable );

    // allocate device memory
    unsigned int memSize = sizeof(uint) * numVoxels;
    d_voxelVerts = CECL_BUFFER(cxGPUContext, CL_MEM_READ_WRITE, memSize, 0, &ciErrNum);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    d_voxelVertsScan = CECL_BUFFER(cxGPUContext, CL_MEM_READ_WRITE, memSize, 0, &ciErrNum);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    d_voxelOccupied = CECL_BUFFER(cxGPUContext, CL_MEM_READ_WRITE, memSize, 0, &ciErrNum);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    d_voxelOccupiedScan = CECL_BUFFER(cxGPUContext, CL_MEM_READ_WRITE, memSize, 0, &ciErrNum);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    d_compVoxelArray = CECL_BUFFER(cxGPUContext, CL_MEM_READ_WRITE, memSize, 0, &ciErrNum);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
}

void Cleanup(int iExitCode)
{
    deleteVBO(&posVbo, d_pos);
    deleteVBO(&normalVbo, d_normal);

    if( d_triTable ) clReleaseMemObject(d_triTable);
    if( d_numVertsTable ) clReleaseMemObject(d_numVertsTable);

    if( d_voxelVerts) clReleaseMemObject(d_voxelVerts);
    if( d_voxelVertsScan) clReleaseMemObject(d_voxelVertsScan);
    if( d_voxelOccupied) clReleaseMemObject(d_voxelOccupied);
    if( d_voxelOccupiedScan) clReleaseMemObject(d_voxelOccupiedScan);
    if( d_compVoxelArray) clReleaseMemObject(d_compVoxelArray);

    if( d_volume) clReleaseMemObject(d_volume);

    closeScan();
    
    if(compactVoxelsKernel)clReleaseKernel(compactVoxelsKernel);  
    if(compactVoxelsKernel)clReleaseKernel(generateTriangles2Kernel);  
    if(compactVoxelsKernel)clReleaseKernel(classifyVoxelKernel);  
    if(cpProgram)clReleaseProgram(cpProgram);

    if(cqCommandQueue)clReleaseCommandQueue(cqCommandQueue);
    if(cxGPUContext)clReleaseContext(cxGPUContext);
    
    // finalize logs and leave
    shrQAFinish2(bQATest, *pArgc, (const char **)pArgv, (iExitCode == 0) ? QA_PASSED : QA_FAILED);

    if ((g_bNoprompt)||(bQATest))
    {
        shrLogEx(LOGBOTH | CLOSELOG, 0, "%s Exiting...\n", cpExecutableName);
    }
    else 
    {
        shrLogEx(LOGBOTH | CLOSELOG, 0, "%s Exiting...\nPress <Enter> to Quit\n", cpExecutableName);
        #ifdef WIN32
            getchar();
        #endif
    }
    exit (iExitCode);
}

void initMenus()
{
    glutCreateMenu(mainMenu);
    glutAddMenuEntry("Toggle animation [ ]", ' ');
    glutAddMenuEntry("Increment isovalue [+]", '+');
    glutAddMenuEntry("Decrement isovalue [-]", '-');
    glutAddMenuEntry("Toggle computation [c]", 'c');
    glutAddMenuEntry("Toggle rendering [r]", 'r');
    glutAddMenuEntry("Toggle lighting [l]", 'l');
    glutAddMenuEntry("Toggle wireframe [w]", 'w');
    glutAddMenuEntry("Quit (esc)", '\033');
    glutAttachMenu(GLUT_RIGHT_BUTTON);
}

void
runTest(int argc, char** argv)
{
    // First initialize OpenGL context, so we can properly set the GL for CUDA.
    // This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.
    if( !bQATest ) {
        initGL(argc, argv);
    }
    
    initCL(argc, argv);

    if( !bQATest ) {
        // register callbacks
        glutDisplayFunc(display);
        glutKeyboardFunc(keyboard);
        glutMouseFunc(mouse);
        glutMotionFunc(motion);
        glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
        glutIdleFunc(idle);
        glutReshapeFunc(reshape);
        initMenus();
    }

    // Initialize CUDA buffers for Marching Cubes 
    initMC(argc, argv);

    // start rendering mainloop
    if( !bQATest ) {
        glutMainLoop();
    } else {
        TestNoGL();
    }
}

#define DEBUG_BUFFERS 0

////////////////////////////////////////////////////////////////////////////////
//! Run the Cuda part of the computation
////////////////////////////////////////////////////////////////////////////////
void
computeIsosurface()
{
    int threads = 128;
    dim3 grid(numVoxels / threads, 1, 1);
    // get around maximum grid size of 65535 in each dimension
    if (grid.x > 65535) {
        grid.y = grid.x / 32768;
        grid.x = 32768;
    }

    // calculate number of vertices need per voxel
    launch_classifyVoxel(grid, threads, 
						d_voxelVerts, d_voxelOccupied, d_volume, 
						gridSize, gridSizeShift, gridSizeMask, 
                         numVoxels, voxelSize, isoValue);

    // scan voxel occupied array
    openclScan(d_voxelOccupiedScan, d_voxelOccupied, numVoxels);

    // read back values to calculate total number of non-empty voxels
    // since we are using an exclusive scan, the total is the last value of
    // the scan result plus the last value in the input array
    {
        uint lastElement, lastScanElement;

        CECL_READ_BUFFER(cqCommandQueue, d_voxelOccupied,CL_TRUE, (numVoxels-1) * sizeof(uint), sizeof(uint), &lastElement, 0, 0, 0);
        CECL_READ_BUFFER(cqCommandQueue, d_voxelOccupiedScan,CL_TRUE, (numVoxels-1) * sizeof(uint), sizeof(uint), &lastScanElement, 0, 0, 0);

        activeVoxels = lastElement + lastScanElement;
    }

    if (activeVoxels==0) {
        // return if there are no full voxels
        totalVerts = 0;
        return;
    }

    //printf("activeVoxels = %d\n", activeVoxels);

    // compact voxel index array
    launch_compactVoxels(grid, threads, d_compVoxelArray, d_voxelOccupied, d_voxelOccupiedScan, numVoxels);


    // scan voxel vertex count array
    openclScan(d_voxelVertsScan, d_voxelVerts, numVoxels);

    // readback total number of vertices
    {
        uint lastElement, lastScanElement;
        CECL_READ_BUFFER(cqCommandQueue, d_voxelVerts,CL_TRUE, (numVoxels-1) * sizeof(uint), sizeof(uint), &lastElement, 0, 0, 0);
        CECL_READ_BUFFER(cqCommandQueue, d_voxelVertsScan,CL_TRUE, (numVoxels-1) * sizeof(uint), sizeof(uint), &lastScanElement, 0, 0, 0);

        totalVerts = lastElement + lastScanElement;
    }

    //printf("totalVerts = %d\n", totalVerts);


    cl_mem interopBuffers[] = {d_pos, d_normal};
    
    // generate triangles, writing to vertex buffers
	if( g_glInterop ) {
		// Acquire PBO for OpenCL writing
		glFlush();
		ciErrNum = clEnqueueAcquireGLObjects(cqCommandQueue, 2, interopBuffers, 0, 0, 0);
        oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    }
    
    dim3 grid2((int) ceil(activeVoxels / (float) NTHREADS), 1, 1);

    while(grid2.x > 65535) {
        grid2.x/=2;
        grid2.y*=2;
    }
    launch_generateTriangles2(grid2, NTHREADS, d_pos, d_normal, 
                                            d_compVoxelArray, 
                                            d_voxelVertsScan, d_volume, 
                                            gridSize, gridSizeShift, gridSizeMask, 
                                            voxelSize, isoValue, activeVoxels, 
                              maxVerts);

	if( g_glInterop ) {
		// Transfer ownership of buffer back from CL to GL    
		ciErrNum = clEnqueueReleaseGLObjects(cqCommandQueue, 2, interopBuffers, 0, 0, 0);
		oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
		clFinish( cqCommandQueue );
	} 

}

// shader for displaying floating-point texture
static const char *shader_code = 
"!!ARBfp1.0\n"
"TEX result.color, fragment.texcoord, texture[0], 2D; \n"
"END";

GLuint compileASMShader(GLenum program_type, const char *code)
{
    GLuint program_id;
    glGenProgramsARB(1, &program_id);
    glBindProgramARB(program_type, program_id);
    glProgramStringARB(program_type, GL_PROGRAM_FORMAT_ASCII_ARB, (GLsizei) strlen(code), (GLubyte *) code);

    GLint error_pos;
    glGetIntegerv(GL_PROGRAM_ERROR_POSITION_ARB, &error_pos);
    if ((int)error_pos != -1) {
        const GLubyte *error_string;
        error_string = glGetString(GL_PROGRAM_ERROR_STRING_ARB);
        shrLog("Program error at position: %d\n%s\n", (int)error_pos, error_string);
        return 0;
    }
    return program_id;
}

////////////////////////////////////////////////////////////////////////////////
//! Initialize OpenGL
////////////////////////////////////////////////////////////////////////////////
bool
initGL(int argc, char **argv)
{
    // Create GL context
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
    glutInitWindowSize(window_width, window_height);
    glutCreateWindow("CUDA Marching Cubes");
#if !(defined (__APPLE__) || defined(MACOSX))
    glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS);
#endif

    // initialize necessary OpenGL extensions
    glewInit();
    if (! glewIsSupported("GL_VERSION_2_0 " 
		                  )) {
        fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
        fflush(stderr);
        return false;
    }

    // default initialization
    glClearColor(0.1f, 0.2f, 0.3f, 1.0f);
    glEnable(GL_DEPTH_TEST);

    // good old-fashioned fixed function lighting
    float black[]    = { 0.0f, 0.0f, 0.0f, 1.0f };
    float white[]    = { 1.0f, 1.0f, 1.0f, 1.0f };
    float ambient[]  = { 0.1f, 0.1f, 0.1f, 1.0f };
    float diffuse[]  = { 0.9f, 0.9f, 0.9f, 1.0f };
    float lightPos[] = { 0.0f, 0.0f, 1.0f, 0.0f };

    glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, ambient);
    glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, diffuse);
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, black);

    glLightfv(GL_LIGHT0, GL_AMBIENT, white);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, white);
    glLightfv(GL_LIGHT0, GL_SPECULAR, white);
    glLightfv(GL_LIGHT0, GL_POSITION, lightPos);

    glLightModelfv(GL_LIGHT_MODEL_AMBIENT, black);

    glEnable(GL_LIGHT0);
    glEnable(GL_NORMALIZE);
    
    // load shader program
    gl_Shader = compileASMShader(GL_FRAGMENT_PROGRAM_ARB, shader_code);


	glutReportErrors();

    g_glInterop = true;

    return true;
}

////////////////////////////////////////////////////////////////////////////////
//! Create VBO
////////////////////////////////////////////////////////////////////////////////
void
createVBO(GLuint* vbo, unsigned int size, cl_mem &vbo_cl)
{
    // create buffer object
    glGenBuffers(1, vbo);
    glBindBuffer(GL_ARRAY_BUFFER, *vbo);

    // initialize buffer object
    glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glutReportErrors();

    vbo_cl = clCreateFromGLBuffer(cxGPUContext,CL_MEM_WRITE_ONLY, *vbo, &ciErrNum);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
}

////////////////////////////////////////////////////////////////////////////////
//! Delete VBO
////////////////////////////////////////////////////////////////////////////////
void
deleteVBO(GLuint* vbo, cl_mem vbo_cl)
{
    if( vbo_cl) clReleaseMemObject(vbo_cl);    

    if( *vbo ) {
        glBindBuffer(1, *vbo);
        glDeleteBuffers(1, vbo);
        
        *vbo = 0;
    }
}

////////////////////////////////////////////////////////////////////////////////
// Render isosurface geometry from the vertex buffers
////////////////////////////////////////////////////////////////////////////////
void renderIsosurface()
{
    glBindBuffer(GL_ARRAY_BUFFER, posVbo);
    glVertexPointer(4, GL_FLOAT, 0, 0);
    glEnableClientState(GL_VERTEX_ARRAY);

    glBindBufferARB(GL_ARRAY_BUFFER_ARB, normalVbo);
    glNormalPointer(GL_FLOAT, sizeof(float)*4, 0);
    glEnableClientState(GL_NORMAL_ARRAY);

    glColor3f(1.0, 0.0, 0.0);
    glDrawArrays(GL_TRIANGLES, 0, totalVerts);
    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_NORMAL_ARRAY);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

////////////////////////////////////////////////////////////////////////////////
//! Display callback
////////////////////////////////////////////////////////////////////////////////
void
display()
{
    shrDeltaT(0);

    // run CUDA kernel to generate geometry
    if (compute) {
        computeIsosurface();
    }


    // Common display code path
	{
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		// set view matrix
		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();
		glTranslatef(translate[0], translate[1], translate[2]);
		glRotatef(rotate[0], 1.0, 0.0, 0.0);
		glRotatef(rotate[1], 0.0, 1.0, 0.0);

		glPolygonMode(GL_FRONT_AND_BACK, wireframe? GL_LINE : GL_FILL);
		if (lighting) {
			glEnable(GL_LIGHTING);
		}

		// render
		if (render) {
			glPushMatrix();
			glRotatef(180.0, 0.0, 1.0, 0.0);
			glRotatef(90.0, 1.0, 0.0, 0.0);
			renderIsosurface();
			glPopMatrix();
		}

		glDisable(GL_LIGHTING);
	} 

    totalTime += shrDeltaT(0);

    glutSwapBuffers();
    glutReportErrors();

    computeFPS();
}


////////////////////////////////////////////////////////////////////////////////
//! Keyboard events handler
////////////////////////////////////////////////////////////////////////////////
void
keyboard(unsigned char key, int /*x*/, int /*y*/)
{
    switch(key) {
    case '\033':// Escape quits    
    case 'Q':   // Q quits
    case 'q':   // q quits
        g_bNoprompt = true;
		Cleanup(EXIT_SUCCESS);
		break;
    case '=':
        isoValue += 0.01f;
        break;
    case '-':
        isoValue -= 0.01f;
        break;
    case '+':
        isoValue += 0.1f;
        break;
    case '_':
        isoValue -= 0.1f;
        break;
    case 'w':
        wireframe = !wireframe;
        break;
    case ' ':
        animate = !animate;
        break;
    case 'l':
        lighting = !lighting;
        break;
    case 'r':
        render = !render;
        break;
    case 'c':
        compute = !compute;
        break;
    }

    printf("isoValue = %f\n", isoValue);
    printf("voxels = %d\n", activeVoxels);
    printf("verts = %d\n", totalVerts);
    printf("occupancy: %d / %d = %.2f%%\n", 
           activeVoxels, numVoxels, activeVoxels*100.0f / (float) numVoxels);

    if (!compute) {
        computeIsosurface();        
    }
}

////////////////////////////////////////////////////////////////////////////////
//! Mouse event handlers
////////////////////////////////////////////////////////////////////////////////
void
mouse(int button, int state, int x, int y)
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
    float dx = (float)(x - mouse_old_x);
    float dy = (float)(y - mouse_old_y);

    if (mouse_buttons==1) {
        rotate[0] += dy * 0.2f;
        rotate[1] += dx * 0.2f;
    } else if (mouse_buttons==2) {
        translate[0] += dx * 0.01f;
        translate[1] -= dy * 0.01f;
    } else if (mouse_buttons==3) {
        translate[2] += dy * 0.01f;
    }

    mouse_old_x = x;
    mouse_old_y = y;
}

void idle()
{
    animation();
    glutPostRedisplay();
}

void reshape(int w, int h)
{
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (float) w / (float) h, 0.1, 10.0);

    glMatrixMode(GL_MODELVIEW);
    glViewport(0, 0, w, h);
}

void mainMenu(int i)
{
    keyboard((unsigned char) i, 0, 0);
}

template <class T>
void dumpBuffer(cl_mem d_buffer, T *h_buffer, int nelements) {
    CECL_READ_BUFFER(cqCommandQueue,d_buffer,CL_FALSE, 0,nelements * sizeof(T), h_buffer, 0, 0, 0);
}

// Run a test sequence without any GL 
//*****************************************************************************
void TestNoGL()
{
    d_normal = CECL_BUFFER(cxGPUContext, CL_MEM_WRITE_ONLY,  maxVerts*sizeof(float)*4, NULL, &ciErrNum);
    d_pos = CECL_BUFFER(cxGPUContext, CL_MEM_WRITE_ONLY,  maxVerts*sizeof(float)*4, NULL, &ciErrNum);
        
    
    // Warmup
    computeIsosurface();
    clFinish(cqCommandQueue);
    
    // Start timer 0 and process n loops on the GPU 
    shrDeltaT(0); 
    int nIter = 100;

    for (int i = 0; i < nIter; i++)
    {
        computeIsosurface();
    }
    clFinish(cqCommandQueue);
    
    // Get elapsed time and throughput, then log to sample and master logs
    double dAvgTime = shrDeltaT(0)/nIter;
    shrLogEx(LOGBOTH | MASTER, 0, "oclMarchingCubes, Throughput = %.4f MVoxels/s, Time = %.5f s, Size = %u Voxels, NumDevsUsed = %u, Workgroup = %u\n", 
           (1.0e-6 * numVoxels)/dAvgTime, dAvgTime, numVoxels, 1, NTHREADS); 
}
