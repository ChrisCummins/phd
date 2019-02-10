#include <libcecl.h>
/**********************************************************************
Copyright ©2015 Advanced Micro Devices, Inc. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

•   Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
•   Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or
 other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
 DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
********************************************************************/


#include "NBody.hpp"
#include <GL/glut.h>
#include <cmath>
#include <malloc.h>

cl_uint numBodies;      /**< No. of particles*/
cl_float* pos;      /**< Output position */
void* me;           /**< Pointing to NBody class */


float
NBody::random(float randMax, float randMin)
{
    float result;
    result =(float)rand() / (float)RAND_MAX;

    return ((1.0f - result) * randMin + result *randMax);
}

int
NBody::setupNBody()
{
    // make sure numParticles is multiple of group size
    numParticles = (cl_uint)(((size_t)numParticles < groupSize) ? groupSize :
                            numParticles);
    numParticles = (cl_uint)((numParticles / groupSize) * groupSize);

    numBodies = numParticles;

    initPos = (cl_float*)malloc(numBodies * sizeof(cl_float4));
    CHECK_ALLOCATION(initPos, "Failed to allocate host memory. (initPos)");

    // initialization of inputs
    for(cl_uint i = 0; i < numBodies; ++i)
    {
        int index = 4 * i;

        // First 3 values are position in x,y and z direction
        for(int j = 0; j < 3; ++j)
        {
            initPos[index + j] = random(3, 50);
        }

        // Mass value
        initPos[index + 3] = random(1, 1000);
    }
    return SDK_SUCCESS;
}

int
NBody::genBinaryImage()
{
    bifData binaryData;
    binaryData.kernelName = std::string("NBody_Kernels.cl");
    binaryData.flagsStr = std::string("");
    if(sampleArgs->isComplierFlagsSpecified())
    {
        binaryData.flagsFileName = std::string(sampleArgs->flags.c_str());
    }

    binaryData.binaryName = std::string(sampleArgs->dumpBinary.c_str());
    int status = generateBinaryImage(binaryData);
    return status;
}


int
NBody::setupCL()
{
    cl_int status = CL_SUCCESS;

    cl_device_type dType;

    if(sampleArgs->deviceType.compare("cpu") == 0)
    {
        dType = CL_DEVICE_TYPE_GPU;
    }
    else //deviceType = "gpu"
    {
        dType = CL_DEVICE_TYPE_GPU;
        if(sampleArgs->isThereGPU() == false)
        {
            std::cout << "GPU not found. Falling back to CPU device" << std::endl;
            dType = CL_DEVICE_TYPE_GPU;
        }
    }

    /*
     * Have a look at the available platforms and pick either
     * the AMD one if available or a reasonable default.
     */
    cl_platform_id platform = NULL;
    int retValue = getPlatform(platform, sampleArgs->platformId,
                               sampleArgs->isPlatformEnabled());
    CHECK_ERROR(retValue, SDK_SUCCESS, "getPlatform() failed");

    // Display available devices.
    retValue = displayDevices(platform, dType);
    CHECK_ERROR(retValue, SDK_SUCCESS, "displayDevices() failed");


    /*
     * If we could find our platform, use it. Otherwise use just available platform.
     */
    cl_context_properties cps[3] =
    {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)platform,
        0
    };

    context = CECL_CREATE_CONTEXT_FROM_TYPE(
                  cps,
                  dType,
                  NULL,
                  NULL,
                  &status);
    CHECK_OPENCL_ERROR( status, "CECL_CREATE_CONTEXT_FROM_TYPE failed.");

    // getting device on which to run the sample
    status = getDevices(context, &devices, sampleArgs->deviceId,
                        sampleArgs->isDeviceIdEnabled());
    CHECK_ERROR(status, SDK_SUCCESS, "getDevices() failed");

    {
        // The block is to move the declaration of prop closer to its use
        cl_command_queue_properties prop = 0;
        commandQueue = CECL_CREATE_COMMAND_QUEUE(
                           context,
                           devices[sampleArgs->deviceId],
                           prop,
                           &status);
        CHECK_OPENCL_ERROR( status, "CECL_CREATE_COMMAND_QUEUE failed.");
    }

    //Set device info of given cl_device_id
    retValue = deviceInfo.setDeviceInfo(devices[sampleArgs->deviceId]);
    CHECK_ERROR(retValue, SDK_SUCCESS, "SDKDeviceInfo::setDeviceInfo() failed");

    /*
    * Create and initialize memory objects
    */
    size_t bufferSize = numBodies * sizeof(cl_float4);
    for (int i = 0; i < 2; i++)
    {
        particlePos[i] = CECL_BUFFER(context, CL_MEM_READ_WRITE, bufferSize, 0,
                                        &status);
        CHECK_OPENCL_ERROR(status, "CECL_BUFFER failed. (particlePos)");
        particleVel[i] = CECL_BUFFER(context, CL_MEM_READ_WRITE, bufferSize, 0,
                                        &status);
        CHECK_OPENCL_ERROR(status, "CECL_BUFFER failed. (particleVel)");
    }

    // Initialize position buffer
    status = CECL_WRITE_BUFFER(commandQueue,particlePos[0],CL_TRUE,0,bufferSize,
                                  initPos,0,0,NULL);
    CHECK_OPENCL_ERROR(status, "CECL_WRITE_BUFFER failed. ");

    // Initialize the velocity buffer to zero
    float* p = (float*) CECL_MAP_BUFFER(commandQueue, particleVel[0], CL_TRUE,
                                           CL_MAP_WRITE
                                           , 0, bufferSize, 0, NULL, NULL, &status);
    CHECK_OPENCL_ERROR(status, "CECL_MAP_BUFFER failed. ");
    memset(p, 0, bufferSize);
    status = clEnqueueUnmapMemObject(commandQueue, particleVel[0], p, 0, NULL,
                                     NULL);
    CHECK_OPENCL_ERROR(status, "clEnqueueUnmapMemObject failed. ");

    status = clFlush(commandQueue);
    CHECK_OPENCL_ERROR(status, "clFlush failed. ");

    // create a CL program using the kernel source
    buildProgramData buildData;
    buildData.kernelName = std::string("NBody_Kernels.cl");
    buildData.devices = devices;
    buildData.deviceId = sampleArgs->deviceId;
    buildData.flagsStr = std::string("");
    if(sampleArgs->isLoadBinaryEnabled())
    {
        buildData.binaryName = std::string(sampleArgs->loadBinary.c_str());
    }

    if(sampleArgs->isComplierFlagsSpecified())
    {
        buildData.flagsFileName = std::string(sampleArgs->flags.c_str());
    }

    retValue = buildOpenCLProgram(program, context, buildData);
    CHECK_ERROR(retValue, SDK_SUCCESS, "buildOpenCLProgram() failed");

    // get a kernel object handle for a kernel with the given name
    kernel = CECL_KERNEL(program,"nbody_sim",&status);
    CHECK_OPENCL_ERROR(status, "CECL_KERNEL failed.");

    return SDK_SUCCESS;
}


int
NBody::setupCLKernels()
{
    cl_int status;

    // Set appropriate arguments to the kernel

    // numBodies
    status = CECL_SET_KERNEL_ARG(
                 kernel,
                 2,
                 sizeof(cl_uint),
                 (void *)&numBodies);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG failed. (numBodies)");

    // time step
    status = CECL_SET_KERNEL_ARG(
                 kernel,
                 3,
                 sizeof(cl_float),
                 (void *)&delT);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG failed. (delT)");

    // upward Pseudoprobability
    status = CECL_SET_KERNEL_ARG(
                 kernel,
                 4,
                 sizeof(cl_float),
                 (void *)&espSqr);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG failed. (espSqr)");


    return SDK_SUCCESS;
}


int NBody::runCLKernels()
{
    cl_int status;

    int currentBuffer = currentPosBufferIndex;
    int nextBuffer = (currentPosBufferIndex+1)%2;

    /*
    * Enqueue a kernel run call.
    */
    size_t globalThreads[] = {numBodies};
    size_t localThreads[] = {groupSize};

    // Particle positions
    status = CECL_SET_KERNEL_ARG(kernel,0,sizeof(cl_mem),
                            (void*) (particlePos+currentBuffer));
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG failed. (updatedPos)");

    // Particle velocity
    status = CECL_SET_KERNEL_ARG(kernel,1,sizeof(cl_mem),
                            (void *) (particleVel+currentBuffer));
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG failed. (updatedVel)");

    // Particle positions
    status = CECL_SET_KERNEL_ARG(kernel,5,sizeof(cl_mem),
                            (void*) (particlePos+nextBuffer));
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG failed. (unewPos)");

    // Particle velocity
    status = CECL_SET_KERNEL_ARG(kernel,6,sizeof(cl_mem),
                            (void*) (particleVel+nextBuffer));
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG failed. (newVel)");

    status = CECL_ND_RANGE_KERNEL(commandQueue,kernel,1,NULL,globalThreads,
                                    localThreads,0,NULL,NULL);
    CHECK_OPENCL_ERROR(status, "CECL_ND_RANGE_KERNEL failed.");

    status = clFlush(commandQueue);
    CHECK_OPENCL_ERROR(status, "clFlush failed.");

    currentPosBufferIndex = nextBuffer;
    timerNumFrames++;
    return SDK_SUCCESS;
}

float* NBody::getMappedParticlePositions()
{
    cl_int status;
    mappedPosBufferIndex = currentPosBufferIndex;
    mappedPosBuffer = (float*) CECL_MAP_BUFFER(commandQueue,
                      particlePos[mappedPosBufferIndex], CL_TRUE, CL_MAP_READ
                      , 0, numBodies*4*sizeof(float), 0, NULL, NULL, &status);
    return mappedPosBuffer;
}

void NBody::releaseMappedParticlePositions()
{
    if (mappedPosBuffer)
    {
        cl_int status = clEnqueueUnmapMemObject(commandQueue,
                                                particlePos[mappedPosBufferIndex], mappedPosBuffer, 0, NULL, NULL);
        mappedPosBuffer = NULL;
        clFlush(commandQueue);
    }
}

/*
* n-body simulation on cpu
*/
void
NBody::nBodyCPUReference(float* currentPos, float* currentVel, float* newPos,
                         float* newVel)
{
    //Iterate for all samples
    for(cl_uint i = 0; i < numBodies; ++i)
    {
        int myIndex = 4 * i;
        float acc[3] = {0.0f, 0.0f, 0.0f};
        for(cl_uint j = 0; j < numBodies; ++j)
        {
            float r[3];
            int index = 4 * j;

            float distSqr = 0.0f;
            for(int k = 0; k < 3; ++k)
            {
                r[k] = currentPos[index + k] - currentPos[myIndex + k];

                distSqr += r[k] * r[k];
            }

            float invDist = 1.0f / sqrt(distSqr + espSqr);
            float invDistCube =  invDist * invDist * invDist;
            float s = currentPos[index + 3] * invDistCube;

            for(int k = 0; k < 3; ++k)
            {
                acc[k] += s * r[k];
            }
        }

        for(int k = 0; k < 3; ++k)
        {
            newPos[myIndex + k] = currentPos[myIndex + k] + currentVel[myIndex + k] * delT +
                                  0.5f * acc[k] * delT * delT;
            newVel[myIndex + k] = currentVel[myIndex + k] + acc[k] * delT;
        }
        newPos[myIndex+3] = currentPos[myIndex + 3];
    }
}

int
NBody::initialize()
{
    // Call base class Initialize to get default configuration
    int status = 0;
    if (sampleArgs->initialize() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    Option *num_particles = new Option;
    CHECK_ALLOCATION(num_particles,
                     "error. Failed to allocate memory (num_particles)\n");

    num_particles->_sVersion = "x";
    num_particles->_lVersion = "particles";
    num_particles->_description = "Number of particles";
    num_particles->_type = CA_ARG_INT;
    num_particles->_value = &numParticles;

    sampleArgs->AddOption(num_particles);
    delete num_particles;

    Option *num_iterations = new Option;
    CHECK_ALLOCATION(num_iterations,
                     "error. Failed to allocate memory (num_iterations)\n");

    num_iterations->_sVersion = "i";
    num_iterations->_lVersion = "iterations";
    num_iterations->_description = "Number of iterations";
    num_iterations->_type = CA_ARG_INT;
    num_iterations->_value = &iterations;

    sampleArgs->AddOption(num_iterations);
    delete num_iterations;

	Option *display_option = new Option;
	CHECK_ALLOCATION(display_option,
		"error. Failed to allocate memory (num_iterations)\n");

	display_option->_sVersion = "g";
	display_option->_lVersion = "gui";
	display_option->_description = "Enable graphical display";
	display_option->_type = CA_NO_ARGUMENT;
	display_option->_value = &display;

	sampleArgs->AddOption(display_option);
	delete display_option;

    return SDK_SUCCESS;
}

int
NBody::setup()
{
    int status = 0;
    if(setupNBody() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    int timer = sampleTimer->createTimer();
    sampleTimer->resetTimer(timer);
    sampleTimer->startTimer(timer);

    status = setupCL();
    if(status != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    sampleTimer->stopTimer(timer);
    // Compute setup time
    setupTime = (double)(sampleTimer->readTimer(timer));

    return SDK_SUCCESS;
}

/**
* @brief Initialize GL
*/
void
GLInit()
{
    glClearColor(0.0 ,0.0, 0.0, 0.0);
    glClear(GL_COLOR_BUFFER_BIT);
    glClear(GL_DEPTH_BUFFER_BIT);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
}

/**
* @brief Glut Idle function
*/
void
idle()
{
    glutPostRedisplay();
}

/**
* @brief Glut reshape func
*
* @param w numParticles of OpenGL window
* @param h height of OpenGL window
*/
void
reShape(int w,int h)
{
    glViewport(0, 0, w, h);

    glViewport(0, 0, w, h);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    gluPerspective(45.0f, w/h, 1.0f, 1000.0f);
    gluLookAt (0.0, 0.0, -2.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0);
}

/**
* @brief OpenGL display function
*/
void displayfunc()
{
    static int numFrames = 0;

    glClearColor(0.0 ,0.0, 0.0, 0.0);
    glClear(GL_COLOR_BUFFER_BIT);
    glClear(GL_DEPTH_BUFFER_BIT);

    glPointSize(1.0);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);
    glEnable(GL_BLEND);
    glDepthMask(GL_FALSE);

    glColor3f(1.0f, 0.5f, 0.5f);

    NBody *nb = (NBody *)me;
    if (nb->isFirstLuanch)
    {
        //Calling kernel for calculatig subsequent positions
        nb->runCLKernels();
        nb->isFirstLuanch = false;
        return;
    }


    cl_uint numBodies = nb->numParticles;
    float* pos = nb->getMappedParticlePositions();
    nb->runCLKernels();
    glBegin(GL_POINTS);
    for(cl_uint i = 0; i < numBodies; ++i,pos+=4)
    {
        //divided by 300 just for scaling
        glVertex4f(*pos,*(pos+1),*(pos+2),300.0f);
    }
    glEnd();
    nb->releaseMappedParticlePositions();

    //Calling kernel for calculating subsequent positions
    glFlush();
    glutSwapBuffers();

    numFrames++;
    // update window title with FPS
    if (numFrames >= 100)
    {
        char buf[256];
        sprintf(buf, "N-body simulation - %d Particles, %.02f FPS"
                , nb->numParticles, (float)nb->getFPS());
        glutSetWindowTitle(buf);
        numFrames = 0;
    }
}

// keyboard function
void
keyboardFunc(unsigned char key, int mouseX, int mouseY)
{
    switch(key)
    {
        // If the user hits escape or Q, then exit

        // ESCAPE_KEY = 27
    case 27:
    case 'q':
    case 'Q':
    {
        if(((NBody*)me)->cleanup() != SDK_SUCCESS)
        {
            exit(1);
        }
        else
        {
            exit(0);
        }
    }
    default:
        break;
    }
}


int
NBody::run()
{
    int status = 0;
    // Arguments are set and execution call is enqueued on command buffer
    if(setupCLKernels() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    if(sampleArgs->verify || sampleArgs->timing)
    {
        int timer = sampleTimer->createTimer();
        sampleTimer->resetTimer(timer);
        sampleTimer->startTimer(timer);

        for(int i = 0; i < iterations; ++i)
        {
            if(runCLKernels() != SDK_SUCCESS)
            {
                return SDK_FAILURE;
            }
        }

        status = clFinish(this->commandQueue);
        sampleTimer->stopTimer(timer);
        // Compute kernel time
        kernelTime = (double)(sampleTimer->readTimer(timer)) / iterations;

    }
    return SDK_SUCCESS;
}

int
NBody::verifyResults()
{
    int ret = SDK_SUCCESS;
    if(sampleArgs->verify)
    {
        float* posBuffers[2];
        float* velBuffers[2];
        for (int i = 0; i < 2; i++)
        {
            posBuffers[i] = (float*)malloc(numBodies * 4 * sizeof(float));
            CHECK_ALLOCATION(posBuffers[i], "Failed to allocate host memory. posBuffers");
            velBuffers[i] = (float*)malloc(numBodies * 4 * sizeof(float));
            CHECK_ALLOCATION(velBuffers[i], "Failed to allocate host memory. velBuffers");
        }
        memcpy(posBuffers[0], initPos, 4 * numBodies * sizeof(float));
        memset(velBuffers[0], 0, numBodies * 4 * sizeof(float));
        for(int i = 0; i < iterations; ++i)
        {
            int current = i%2;
            int next = (i+1)%2;
            nBodyCPUReference(posBuffers[current], velBuffers[current]
                              , posBuffers[next], velBuffers[next]);
        }

        // compare the results and see if they match
        float* pos = getMappedParticlePositions();
        if(compare(pos, posBuffers[(iterations)%2], 4 * numBodies, 0.00001))
        {
            std::cout << "Passed!\n" << std::endl;
            ret = SDK_SUCCESS;
        }
        else
        {
            std::cout << "Failed!\n" << std::endl;
            ret = SDK_FAILURE;
        }
        releaseMappedParticlePositions();

        for (int i = 0; i < 2; i++)
        {
            free(posBuffers[i]);
            free(velBuffers[i]);
        }

    }
    return ret;
}

void
NBody::printStats()
{
    if(sampleArgs->timing)
    {
        std::string strArray[4] =
        {
            "Particles",
            "Iterations",
            "Kernel Time(sec)",
			"GFLOPS"
        };

        std::string stats[4];
        
		double GFLOPs = ((double)((double)(KERNEL_FLOPS * numBodies))*numBodies*powf(10, -9)) / kernelTime;
        
		stats[0] = toString(numParticles, std::dec);
        stats[1] = toString(iterations, std::dec);
        stats[2] = toString(kernelTime, std::dec);
		stats[3] = toString(GFLOPs, std::dec);

        printStatistics(strArray, stats, 4);
    }
}

int
NBody::cleanup()
{
    // Releases OpenCL resources (Context, Memory etc.)
    cl_int status;

    status = clReleaseKernel(kernel);
    CHECK_OPENCL_ERROR(status, "clReleaseKernel failed.(kernel)");

    status = clReleaseProgram(program);
    CHECK_OPENCL_ERROR(status, "clReleaseProgram failed.(program)");

    for (int i = 0; i < 2; i++)
    {
        status = clReleaseMemObject(particlePos[i]);
        CHECK_OPENCL_ERROR(status, "clReleaseMemObject failed.(particlePos)");
        status = clReleaseMemObject(particleVel[i]);
        CHECK_OPENCL_ERROR(status, "clReleaseMemObject failed.(particleVel)");
    }

    status = clReleaseCommandQueue(commandQueue);
    CHECK_OPENCL_ERROR(status, "clReleaseCommandQueue failed.(commandQueue)");

    status = clReleaseContext(context);
    CHECK_OPENCL_ERROR(status, "clReleaseContext failed.(context)");

    return SDK_SUCCESS;
}

NBody::~NBody()
{
    if (this->glEvent)
    {
        clReleaseEvent(this->glEvent);
    }
    // release program resources
    FREE(initPos);

    FREE(initVel);

#if defined (_WIN32)
    ALIGNED_FREE(pos);
#else
    FREE(pos);
#endif

#if defined (_WIN32)
    ALIGNED_FREE(vel);
#else
    FREE(vel);
#endif

    FREE(devices);
}


int
main(int argc, char * argv[])
{
    int status = 0;
    NBody clNBody;
    me = &clNBody;

    if(clNBody.initialize() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    if (clNBody.sampleArgs->parseCommandLine(argc, argv) != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    if(clNBody.sampleArgs->isDumpBinaryEnabled())
    {
        return clNBody.genBinaryImage();
    }

    status = clNBody.setup();
    if(status != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    status = clNBody.run();
    CHECK_ERROR(status, SDK_SUCCESS, "Sample Run Program Failed");

    status = clNBody.verifyResults();
    CHECK_ERROR(status, SDK_SUCCESS, "Sample Verify Results Failed");

    clNBody.printStats();

	if (clNBody.display)
    {
        // Run in  graphical window if requested
        glutInit(&argc, argv);
        glutInitWindowPosition(100,10);
        glutInitWindowSize(600,600);
        glutInitDisplayMode( GLUT_RGB | GLUT_DOUBLE );
        glutCreateWindow("N-body simulation");
        GLInit();
        glutDisplayFunc(displayfunc);
        glutReshapeFunc(reShape);
        glutIdleFunc(idle);
        glutKeyboardFunc(keyboardFunc);
        glutMainLoop();
    }

    status = clNBody.cleanup();
    CHECK_ERROR(status, SDK_SUCCESS, "Sample CleanUP Failed");

    return SDK_SUCCESS;
}
