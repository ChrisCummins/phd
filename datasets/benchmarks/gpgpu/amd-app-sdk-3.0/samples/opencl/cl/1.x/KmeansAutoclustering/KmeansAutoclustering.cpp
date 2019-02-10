#include <libcecl.h>
/**********************************************************************
Copyright ©2015 Advanced Micro Devices, Inc. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

.   Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
.   Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or
 other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
 DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
********************************************************************/


#include "KmeansAutoclustering.hpp"

void* me;           /**< Pointing to KMeans class */
bool display;


void KMeans::initializeCentroidPos(void *pos, int K)
{
    memcpy(pos, backupCentroidPos, K * sizeof(cl_float2));
}

void KMeans::initializeCentroidPos(cl_mem posBuffer, int K)
{
    cl_float2* pointPos = NULL;
 
    mapBuffer<cl_float2>(clCentroidPos, pointPos, K * sizeof(cl_float2), CL_MAP_WRITE_INVALIDATE_REGION);
    memcpy(pointPos, backupCentroidPos, K * sizeof(cl_float2));
    unmapBuffer(clCentroidPos, pointPos);
}

float KMeans::getSilhouetteMapValue(int index)
{
    return silhouettesMap[index];
}

bool KMeans::getIsNumClustersSpecified()
{
    return isNumClustersSpecified;
}
    
bool KMeans::getIsSaturated()
{
    return isSaturated;
}

cl_mem KMeans::getclCentroidPos()
{
    return clCentroidPos;
}

void KMeans::setIsSaturated(bool val)
{
    isSaturated = val;
}

/**
 * This function returns numClusters if the user has specified a fixed K value to clustering
 * Otherwise it returns the Best possible K, computed after silhouette value comparisons.
 * The function must not be called before KMeans computation
 */
int KMeans::getNumClusters()
{
    if(isNumClustersSpecified)
        return numClusters;
    else
        return bestClusterNums;
}

int KMeans::getNumPoints()
{
    return numPoints;
}

/**
 * mapUnmapForRead
 * map a buffer, reads its contents and unmaps it
 * @param deviceBuffer cl_mem
 * @param hostPointer T*
 * @param sizeInBytes size_t
 * @param flags cl_map_flags
 * @return 0 if success else nonzero
 */
template<typename T>
int KMeans::mapBuffer(cl_mem deviceBuffer, T* &hostPointer,
                         size_t sizeInBytes, cl_map_flags flags)
{
    cl_int status;
    hostPointer = (T*) CECL_MAP_BUFFER(commandQueue,
                                          deviceBuffer,
                                          CL_TRUE,
                                          flags,
                                          0,
                                          sizeInBytes,
                                          0,
                                          NULL,
                                          NULL,
                                          &status);
    CHECK_OPENCL_ERROR(status, "CECL_MAP_BUFFER failed");

    return SDK_SUCCESS;
}

int KMeans::unmapBuffer(cl_mem deviceBuffer, void* hostPointer)
{
    cl_int status;
    status = clEnqueueUnmapMemObject(commandQueue,
                                     deviceBuffer,
                                     hostPointer,
                                     0,
                                     NULL,
                                     NULL);
    CHECK_OPENCL_ERROR(status, "clEnqueueUnmapMemObject failed");

    return SDK_SUCCESS;
}

//This function tries to guess the best possible initial centroid positions to run kmeans
// Here we divide the complete 2D plane in a number of bins(64), and compute an histogram
// of the input points. The initial centroid points are randomly generated from the regions 
// which corrospond to the most dense bins. This ensures that centroids are initially put
// close to many input points, instead of placing it at some random place.
// Effectively 2D plane is divided into a 8X8 mesh here. Different mesh sizes can be 
// experimented upon.
void KMeans::setInitialCentroidPos(int K)
{
    if(K == 0)
    {
        return;
    }
    
    //Doing histogram on input data, and selecting most-dense areas as input centroids
    int numBins = 64; //hard for now
    int binDim = sqrt( static_cast< float >( numBins ) );
    float binSize = (2.0 * MAX_COORD) / binDim;
    int offset = binDim / 2; //To make negative co-ordinate values to positive binID
    int* hist = new int[numBins];
    memset(hist, 0, numBins * sizeof(int));
    for(int i=0; i<numPoints; i++)
    {
        int binWidth = pointPos[i].s[0] / binSize + offset;
        int binHeight = pointPos[i].s[1] / binSize + offset;
        int binId = binHeight * binDim + binWidth;
        hist[binId]++;
    }
    

    //Creating sorted BinIds to choose initial centroid points
    int* sortedBinIDs = new int[K];
    int* bin = new int[numBins];
    memcpy(bin, hist, numBins * sizeof(int));
    int copyIndex = 0;
    for(int i=0; (i<numBins) && (i<K); i++)
    {
        int mostDenseBinId = 0;
        int mostDenseBinValue = 0;
        for(int j=0; j<numBins; j++)
        {
            if(bin[j] > mostDenseBinValue)
            {
                mostDenseBinId = j;
                mostDenseBinValue = bin[j];
            }
        }
        //if density is better than average density Choose centroid from this region
        if(bin[mostDenseBinId] >= (numPoints / (numBins))) 
        {
            sortedBinIDs[i] = mostDenseBinId;
        }
        else //Choose next centroid from already selected regions
        {
            sortedBinIDs[i] = sortedBinIDs[(copyIndex++) % i];
        }

        backupCentroidPos[i].s[0] = -MAX_COORD + (sortedBinIDs[i] % binDim) * binSize + ((float)rand() / RAND_MAX) / 2 + (binSize / 2);
        backupCentroidPos[i].s[1] = -MAX_COORD + (sortedBinIDs[i] / binDim) * binSize + ((float)rand() / RAND_MAX) / 2 + (binSize / 2);

        bin[mostDenseBinId] = 0; //make most dense area as zero, so that next most dense is selected next time
    }
    
    delete hist;
    delete bin;
    delete sortedBinIDs;
    if(!sampleArgs->quiet)
    {
        printArray<cl_float2>("Initial Centroids", backupCentroidPos, K, 1, 2);
    }
    return;
}


float
KMeans::random(int randMax, int randMin)
{
    int range = randMax - randMin;
    float result = randMin + ((float)rand() / RAND_MAX) * (range);
    return result;
}

int KMeans::sanityCheck()
{
    if(numClusters < 2 && isNumClustersSpecified)
    {
        std::cout << "Invalid Value for numClusters specified";
        return SDK_EXPECTED_FAILURE;
    }
    
    if(numClusters > MAX_CLUSTERS || lowerBoundForClustering > MAX_CLUSTERS || 
        upperBoundForClustering > MAX_CLUSTERS || (lowerBoundForClustering - upperBoundForClustering) > 0)
    {
        std::cout << "Invalid input provided for clustering";
        return SDK_EXPECTED_FAILURE;
    }
    return SDK_SUCCESS;
}

int
KMeans::setupKMeans()
{
    if(numClusters != 0) //Default behavior
    {
        isNumClustersSpecified = true;
    }
    
    int status = sanityCheck();
    if(status != SDK_SUCCESS)
    {
        return SDK_EXPECTED_FAILURE; //Invalid input params specified
    }

    refPointPos = (cl_float2*)malloc(numPoints * sizeof(cl_float2));
    refKMeansCluster = (cl_uint*)malloc(numPoints * sizeof(cl_uint));
    refCentroidPos = (cl_float2*)malloc(MAX_CLUSTERS * sizeof(cl_float2));
    backupCentroidPos = (cl_float2*)malloc(MAX_CLUSTERS * sizeof(cl_float2));
    refNewCentroidPos = (cl_float2*)malloc(MAX_CLUSTERS * sizeof(cl_float2));
    refCentroidPtsCount = (cl_uint*)malloc(MAX_CLUSTERS * sizeof(cl_uint));
    KMeansCluster = (cl_uint*)malloc(numPoints * sizeof(cl_uint)); //Needed for display function
    
    CHECK_ALLOCATION(refPointPos, "Failed to allocate host memory. (refPointPos)");
    CHECK_ALLOCATION(refKMeansCluster, "Failed to allocate host memory. (refKMeansCluster)");
    CHECK_ALLOCATION(refCentroidPos, "Failed to allocate host memory. (refCentroidPos)");
    CHECK_ALLOCATION(backupCentroidPos, "Failed to allocate host memory. (backupCentroidPos)");
    CHECK_ALLOCATION(refNewCentroidPos, "Failed to allocate host memory. (refNewCentroidPos)");
    CHECK_ALLOCATION(refCentroidPtsCount, "Failed to allocate host memory. (refCentroidCount)");
    CHECK_ALLOCATION(KMeansCluster, "Failed to allocate host memory. (KMeansCluster)");
    
    //Map the pointers to cl_mem objects
    mapBuffer<cl_float2>(clPointPos, pointPos, numPoints * sizeof(cl_float2), CL_MAP_WRITE_INVALIDATE_REGION);
    mapBuffer<cl_float2>(clCentroidPos, centroidPos, MAX_CLUSTERS * sizeof(cl_float2), CL_MAP_WRITE_INVALIDATE_REGION);
    mapBuffer<cl_float2>(clNewCentroidPos, newCentroidPos, MAX_CLUSTERS * sizeof(cl_float2), CL_MAP_WRITE_INVALIDATE_REGION);
    mapBuffer<cl_uint>(clCentroidPtsCount, centroidPtsCount, MAX_CLUSTERS * sizeof(cl_uint), CL_MAP_WRITE_INVALIDATE_REGION);


    //Creating Customized input points based on user inputs
    cl_float2* randCentroidPos = new cl_float2[randClusterNums];
    //Avoiding creation of centroids in corners, as the points are not distributed properly then
    for(int i = 0; i < randClusterNums; ++i)
    {
        randCentroidPos[i].s[0] = random((MAX_COORD - 3), -(MAX_COORD - 3));
        randCentroidPos[i].s[1] = random((MAX_COORD - 3), -(MAX_COORD - 3));
        if(!sampleArgs->quiet)
        {
            std::cout << "Custom Centroid Postions:" << randCentroidPos[i].s[0] << ", " 
                    << randCentroidPos[i].s[1] << std::endl;
        }
    }

    int pointPosVariationfactor = randClusterNums; //Just to create regions of appropriate size

    if(randClusterNums == 0) //Generate random data
    {
        for(int i=0; i<numPoints; i++)
        {
                pointPos[i].s[0] = random(MAX_COORD, -MAX_COORD);
                pointPos[i].s[1] = random(MAX_COORD, -MAX_COORD);
        }
    }
    else // Generate data based on randNumClusters
    {
        for(int i=0; i<numPoints; i++)
        {
            {
                float radius = ((float)rand() / RAND_MAX);
                float theta = ((float)rand() / RAND_MAX) * (2 * PI);
                int clusterIndex = rand() % randClusterNums;
                pointPos[i].s[0] = randCentroidPos[clusterIndex].s[0] + radius * cos(theta);
                pointPos[i].s[1] = randCentroidPos[clusterIndex].s[1] + radius * sin(theta);
            }
        }
    }

    delete[] randCentroidPos;

    
    for(int i=0; i<MAX_CLUSTERS; i++)
    {
        centroidPtsCount[i] = 0;
        centroidPos[i].s[0] = 0.f;
        centroidPos[i].s[1] = 0.f;
        newCentroidPos[i].s[0] = 0.f;
        newCentroidPos[i].s[1] = 0.f;
    }

    memcpy(refPointPos, pointPos, numPoints * sizeof(cl_float2));
    memcpy(refCentroidPos, centroidPos, MAX_CLUSTERS * sizeof(cl_float2));
    memcpy(refNewCentroidPos, newCentroidPos, MAX_CLUSTERS * sizeof(cl_float2));
    memcpy(refCentroidPtsCount, centroidPtsCount, MAX_CLUSTERS * sizeof(cl_uint));
    
    //Fixing Initial Centroid location
    setInitialCentroidPos(MAX_CLUSTERS);

    //Unmapping the mapped Buffers
    unmapBuffer(clPointPos, pointPos);
    unmapBuffer(clCentroidPos, centroidPos);
    unmapBuffer(clNewCentroidPos, newCentroidPos);
    unmapBuffer(clCentroidPtsCount, centroidPtsCount);

    return SDK_SUCCESS;
}

int 
KMeans::genBinaryImage()
{
    bifData binaryData;
    binaryData.kernelName = std::string("KmeansAutoclustering_Kernels.cl");
    std::ostringstream buildOptions;
    buildOptions << "-D MAX_CLUSTERS=" << MAX_CLUSTERS ;
    binaryData.flagsStr = buildOptions.str();
    if(sampleArgs->isComplierFlagsSpecified())
    {
        binaryData.flagsFileName = std::string(sampleArgs->flags.c_str());
    }

    binaryData.binaryName = std::string(sampleArgs->dumpBinary.c_str());
    int status = generateBinaryImage(binaryData);
    return status;
}


int KMeans::setupCL()
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
    int retValue = getPlatform(platform, sampleArgs->platformId, sampleArgs->isPlatformEnabled());
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
    status = getDevices(context, &devices, sampleArgs->deviceId, sampleArgs->isDeviceIdEnabled());
    CHECK_ERROR(status, SDK_SUCCESS, "sampleCommon::getDevices() failed");

    {
        // The block is to move the declaration of prop closer to its use
#ifdef CL_VERSION_2_0
		cl_queue_properties *props = NULL;
		commandQueue = CECL_CREATE_COMMAND_QUEUEWithProperties(
						   context,
						   devices[sampleArgs->deviceId],
						   props,
						   &status);
		CHECK_OPENCL_ERROR( status, "CECL_CREATE_COMMAND_QUEUEWithProperties failed.");
#else     
		cl_command_queue_properties prop = 0;
        commandQueue = CECL_CREATE_COMMAND_QUEUE(
                context, 
                devices[sampleArgs->deviceId], 
                prop, 
                &status);
        CHECK_OPENCL_ERROR( status, "CECL_CREATE_COMMAND_QUEUE failed.");
#endif
    }

    //Set device info of given cl_device_id
    retValue = deviceInfo.setDeviceInfo(devices[sampleArgs->deviceId]);
    CHECK_ERROR(retValue, SDK_SUCCESS, "SDKDeviceInfo::setDeviceInfo() failed");

    // Create memory objects for points position
    clPointPos = CECL_BUFFER(context,
                                CL_MEM_READ_ONLY,
                                numPoints * sizeof(cl_float2),
                                0,
                                &status);
    CHECK_OPENCL_ERROR(status, "CECL_BUFFER failed. (oldPos)");

    // Create memory objects for storing cluster for a point
    clKMeansCluster = CECL_BUFFER(context,
                                     CL_MEM_WRITE_ONLY,
                                     numPoints * sizeof(cl_uint),
                                     0,
                                     &status);
    CHECK_OPENCL_ERROR(status, "CECL_BUFFER failed. (newPos)");

    // Create memory objects for storing point's position for centroids
    clCentroidPos = CECL_BUFFER(context,
                                    CL_MEM_READ_WRITE,
                                    MAX_CLUSTERS * sizeof(cl_float2),
                                    0,
                                    &status);
    CHECK_OPENCL_ERROR(status, "CECL_BUFFER failed. (oldVel)");

    // Create memory object for storing new computed centroids
    clNewCentroidPos = CECL_BUFFER(context,
                                CL_MEM_READ_ONLY,
                                MAX_CLUSTERS * sizeof(cl_float2),
                                0,
                                &status);
    CHECK_OPENCL_ERROR(status, "CECL_BUFFER failed. (oldPos)");

    // Create memory object to store number of points in a cluster
    clCentroidPtsCount = CECL_BUFFER(context,
                                CL_MEM_READ_ONLY,
                                MAX_CLUSTERS * sizeof(cl_uint),
                                0,
                                &status);
    CHECK_OPENCL_ERROR(status, "CECL_BUFFER failed. (clCentroidPtsCount)");

    // Create memory object to stote to silhouette value of a clustering
    clSilhoutteValue = CECL_BUFFER(context,
                                CL_MEM_READ_WRITE,
                                sizeof(cl_float),
                                0,
                                &status);
    CHECK_OPENCL_ERROR(status, "CECL_BUFFER failed. (clSilhoutteValue)");

    status = clFlush(commandQueue);
    CHECK_OPENCL_ERROR(status, "clFlush failed. ");

    // create a CL program using the kernel source 
    buildProgramData buildData;
    buildData.kernelName = std::string("KmeansAutoclustering_Kernels.cl");
    buildData.devices = devices;
    buildData.deviceId = sampleArgs->deviceId;
    std::ostringstream buildOptions;
    buildOptions << "-D MAX_CLUSTERS=" << MAX_CLUSTERS;
    buildData.flagsStr = buildOptions.str();
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
    kernelAssignCentroid = CECL_KERNEL(
        program,
        "assignCentroid",
        &status);
    CHECK_OPENCL_ERROR(status, "CECL_KERNEL failed(kernelAssignCentroid).");

    kernelComputeSilhouette = CECL_KERNEL(
        program,
        "computeSilhouettes",
        &status);
    CHECK_OPENCL_ERROR(status, "CECL_KERNEL failed.(kernelComputeSilhouette)");

    return SDK_SUCCESS;
}


int 
KMeans::setupCLKernels()
{
    cl_int status;

    // Set appropriate arguments to the kernelAssignCentroid

    // Point positions
    status = CECL_SET_KERNEL_ARG(
        kernelAssignCentroid,
        0,
        sizeof(cl_mem),
        (void*)&clPointPos);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG failed. (clPointPos)");

    // Point to cluster map
    status = CECL_SET_KERNEL_ARG(
        kernelAssignCentroid,
        1,
        sizeof(cl_mem),
        (void *)&clKMeansCluster);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG failed. (clKMeansCluster)");

    // centroid positions
    status = CECL_SET_KERNEL_ARG(
        kernelAssignCentroid,
        2,
        sizeof(cl_mem),
        (void *)&clCentroidPos);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG failed. (clCentroidPos)");

     // New centroid positions
    status = CECL_SET_KERNEL_ARG(
        kernelAssignCentroid,
        3,
        sizeof(cl_mem),
        (void *)&clNewCentroidPos);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG failed. (clCentroidPos)");

     // centroid count
    status = CECL_SET_KERNEL_ARG(
        kernelAssignCentroid,
        4,
        sizeof(cl_mem),
        (void *)&clCentroidPtsCount);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG failed. (clCentroidPtsPos)");

    // LDS buffer for centroid bin
    status = CECL_SET_KERNEL_ARG(
        kernelAssignCentroid,
        5,
        MAX_CLUSTERS * sizeof(cl_float2),
        NULL);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG failed. (LDS Centroid Bin)");

     // LDS buffer for centroid count
    status = CECL_SET_KERNEL_ARG(
        kernelAssignCentroid,
        6,
        MAX_CLUSTERS * sizeof(cl_uint),
        NULL);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG failed. (clCentroidPos)");


    // Number of clusters. 
    //Argument 7 changes with every call to ComputeKMeans, so it is set there itself.

    // number of Points
    status = CECL_SET_KERNEL_ARG(
        kernelAssignCentroid,
        8,
        sizeof(cl_uint),
        (void *)&numPoints);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG failed. (numPoints)");


    //Setting arguments for kernelComputeSilhouette
    status = CECL_SET_KERNEL_ARG(
        kernelComputeSilhouette,
        0,
        sizeof(cl_mem),
        (void *)&clPointPos);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG failed. (clPointPos)");

    status = CECL_SET_KERNEL_ARG(
        kernelComputeSilhouette,
        1,
        sizeof(cl_mem),
        (void *)&clCentroidPos);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG failed. (clCentroidPos)");

    status = CECL_SET_KERNEL_ARG(
        kernelComputeSilhouette,
        2,
        sizeof(cl_mem),
        (void *)&clKMeansCluster);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG failed. (clKMeansCluster)");

    status = CECL_SET_KERNEL_ARG(
        kernelComputeSilhouette,
        3,
        sizeof(cl_mem),
        (void *)&clCentroidPtsCount);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG failed. (clCentroidPtsCount)");

    status = CECL_SET_KERNEL_ARG(
        kernelComputeSilhouette,
        4,
        sizeof(cl_int) * MAX_CLUSTERS,
        NULL);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG failed. (lCluster Count (LDS))");

    // Number of clusters. 
    //Argument 5 changes with every call to ComputeSilhouette, so it is set there itself.

    status = CECL_SET_KERNEL_ARG(
        kernelComputeSilhouette,
        6,
        sizeof(cl_int),
        (void *)&numPoints);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG failed. (numPoints)");

    // Space to store Dissimilarities in LDS, This would require
    // MAX_CLUSTERS * groupSize sizeof(float) (16 X 256 X 4) = 16KB
    //Currently VGPRs are being used for storing the dissimilarities
    
    status = CECL_SET_KERNEL_ARG(
        kernelComputeSilhouette,
        7,
        sizeof(cl_float),
        (void *)NULL);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG failed. localSilhouette Value (LDS)");

    status = CECL_SET_KERNEL_ARG(
        kernelComputeSilhouette,
        8,
        sizeof(cl_mem),
        (void *)&clSilhoutteValue);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG failed. global Silhouette Value");


    status = kernelInfo.setKernelWorkGroupInfo(kernelAssignCentroid, devices[sampleArgs->deviceId]);
    CHECK_ERROR(status, SDK_SUCCESS, "kernelInfo.setKernelWorkGroupInfo() failed");

    if(kernelInfo.localMemoryUsed > deviceInfo.localMemSize)
    {
        std::cout << "Unsupported: Insufficient local memory on device" <<
            std::endl;
        return SDK_EXPECTED_FAILURE;
    }

    if(groupSize > kernelInfo.kernelWorkGroupSize)
    {
        if(!sampleArgs->quiet)
        {
            std::cout << "Out of Resources!" << std::endl;
            std::cout << "Group Size specified : " << groupSize << std::endl;
            std::cout << "Max Group Size supported on the kernel : "
                      << kernelInfo.kernelWorkGroupSize << std::endl;
            std::cout << "Falling back to " << kernelInfo.kernelWorkGroupSize << std::endl;
        }
        groupSize = kernelInfo.kernelWorkGroupSize;
    }

    return SDK_SUCCESS;
}

int KMeans::computeKMeans(int K)
{
    cl_int status;

    if(isSaturated)
    {
        return SDK_SUCCESS; // solution already found
    }
    else
    {
        // Needs to be set everytime
        status = CECL_SET_KERNEL_ARG(
            kernelAssignCentroid,
            7,
            sizeof(cl_uint),
            (void *)&K);
        CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG failed. (numClusters)");
        
        //Enqueue a kernel run call.
        size_t globalThreads[] = {numPoints};
    
    
        status = CECL_ND_RANGE_KERNEL(
            commandQueue,
            kernelAssignCentroid,
            1,
            NULL,
            globalThreads,
            NULL,
            0,
            NULL,
            NULL);
        CHECK_OPENCL_ERROR(status, "CECL_ND_RANGE_KERNEL failed.");
    
        status = clFlush(commandQueue);
        CHECK_OPENCL_ERROR(status, "clFlush failed. ");
            
        status = checkCentroidSaturation(K);
        CHECK_ERROR(status, SDK_SUCCESS, "checkCentroidSaturation(K) Failed");
    }

    return SDK_SUCCESS;
}

int KMeans::computeSilhouette(int K, float& val)
{
    cl_int status;
    val = 0.f;
    size_t globalThreads[] = {numPoints};

    // Needs to be set everytime
    status = CECL_SET_KERNEL_ARG(
        kernelComputeSilhouette,
        5,
        sizeof(cl_int),
        (void *)&K);
    CHECK_OPENCL_ERROR(status, "CECL_SET_KERNEL_ARG failed. (numClusters)");
    
    // Initialize Silhouette Value to zero
    float* silVal = NULL;
    mapBuffer<cl_float>(clSilhoutteValue, silVal, 
        sizeof(cl_float), CL_MAP_WRITE);
    *silVal = 0.0f;
    unmapBuffer(clSilhoutteValue, silVal);

    /* 
    * Enqueue a kernel run call.
    */
    status = CECL_ND_RANGE_KERNEL(
            commandQueue,
            kernelComputeSilhouette,
            1,
            NULL,
            globalThreads,
            NULL,
            0,
            NULL,
            NULL);
    CHECK_OPENCL_ERROR(status, "CECL_ND_RANGE_KERNEL failed.");

    status = clFlush(commandQueue);
    CHECK_OPENCL_ERROR(status, "clFlush failed. ");
	    
    // Read data for verification or display
    mapBuffer<cl_float>(clSilhoutteValue, silVal, 
        sizeof(cl_float), CL_MAP_READ);
    val = *silVal;
    unmapBuffer(clSilhoutteValue, silVal);
    val = val / numPoints;

    //Set clCentroidPtsCount to 0, to make sure next computeKMeans run is good. 
    // Needed when Autoclustering is enabled
    if(!isNumClustersSpecified)
    {
        mapBuffer<cl_uint>(clCentroidPtsCount, centroidPtsCount,
            sizeof(cl_uint) * MAX_CLUSTERS, CL_MAP_WRITE_INVALIDATE_REGION);
        memset(centroidPtsCount, 0, sizeof(cl_uint) * MAX_CLUSTERS);
        unmapBuffer(clCentroidPtsCount, centroidPtsCount);
    }
    return CL_SUCCESS;
}

int 
KMeans::runCLKernels()
{
    cl_int status = 0;
    if(isNumClustersSpecified)
    {
        initializeCentroidPos(clCentroidPos, numClusters);
        isSaturated = false;
        while(!isSaturated)  
        {
            status = computeKMeans(numClusters);
            CHECK_ERROR(status, SDK_SUCCESS, "ComputeKmeans failed");
        }
        float val;
        status = computeSilhouette(numClusters, val);
        CHECK_ERROR(status, SDK_SUCCESS, "ComputeSilhouette failed");
        silhouettesMap[numClusters] = val;
        
        if(!sampleArgs->quiet)
        {
            printArray<cl_float2>("New Centroid Points", centroidPos, numClusters, 1, 2);
        }
    }
    else
    {
        for(int K=lowerBoundForClustering; K<=upperBoundForClustering; K++)
        {
            initializeCentroidPos(clCentroidPos, K);
            isSaturated = false;
            while(!isSaturated)  
            {
                status = computeKMeans(K);
                CHECK_ERROR(status, SDK_SUCCESS, "ComputeKmeans failed");
            }

            float val;
            status = computeSilhouette(K, val);
            CHECK_ERROR(status, SDK_SUCCESS, "ComputeSilhouette failed");
            if(val > bestSilhouetteValue)
            {
                bestClusterNums = K;
                bestSilhouetteValue = val;
            }
            silhouettesMap[K] = val;
        }
    }
    return CL_SUCCESS;
}

/**
* Check whether the centroid position had converged
* true is saturated, false otherwise
*/
int KMeans::checkCentroidSaturation(int K)
{
    if(isSaturated)
        return SDK_SUCCESS;
    
    // This is not algorithmically required. We only intend to do it, when display is enabled.
    // This read should give the effect that centroids are correcting itself after every iteration.
    // Also Using a map unmap here would not help, as KMeansCluster pointer is required to contain
    // valid data, after this function has been executed in displayfunc function. 
    // So sticking with old CECL_READ_BUFFER.
    if(display)
    {
        int status = CECL_READ_BUFFER(
                commandQueue,
                clKMeansCluster, 
                CL_TRUE,
                0,
                numPoints * sizeof(cl_int),
                KMeansCluster,
                0,
                NULL,
                NULL);
        CHECK_OPENCL_ERROR(status, "CECL_READ_BUFFER(clKMeansCluster) failed.");
    }
    
    mapBuffer<cl_float2>(clCentroidPos, centroidPos,
        K * sizeof(cl_float2), CL_MAP_READ | CL_MAP_WRITE);
    mapBuffer<cl_float2>(clNewCentroidPos, newCentroidPos,
        K * sizeof(cl_float2), CL_MAP_READ | CL_MAP_WRITE);
    mapBuffer<cl_uint>(clCentroidPtsCount, centroidPtsCount, 
        K * sizeof(cl_uint), CL_MAP_READ | CL_MAP_WRITE);

    for(int i=0; i<K; i++)
    {
        if(centroidPtsCount[i] == 0)
            continue;

        newCentroidPos[i].s[0] = newCentroidPos[i].s[0] / centroidPtsCount[i];
        newCentroidPos[i].s[1] = newCentroidPos[i].s[1] / centroidPtsCount[i];
    }
    
    int i;
    isSaturated = true;
    for(i=0; i<K; i++)
    {
        if((fabsf(centroidPos[i].s[0] - newCentroidPos[i].s[0]) > erfc) || (fabsf(centroidPos[i].s[1] - newCentroidPos[i].s[1]) > erfc)) 
        {
            isSaturated = false;
            break;
        }
    }
    
    if(!isSaturated)// Another kmeans iteration is needed
    {
        //copy newCentroidPos to centroidPos
        memcpy(centroidPos, newCentroidPos, K * sizeof(cl_float2));
        //Reset new centroid Pos to zero
        memset(newCentroidPos, 0, K * sizeof(cl_float2));
        //Reset centroid points count to zero
        memset(centroidPtsCount, 0, K * sizeof(cl_int));
    }

    unmapBuffer(clCentroidPos, centroidPos);
    unmapBuffer(clNewCentroidPos, newCentroidPos);
    unmapBuffer(clCentroidPtsCount, centroidPtsCount);
    return SDK_SUCCESS;
}


/*
 * KMeans clustering simulation on cpu
 */
void 
KMeans::KMeansCPUReference()
{
    if(isNumClustersSpecified)
    {
        initializeCentroidPos(refCentroidPos, numClusters);
        computeRefKMeans(numClusters); //Saturation is checked within this function
        float val;
        computeRefSilhouette(numClusters, val);
        refSilhouettesMap[numClusters] = val;
    }
    else
    {
        for(int K=lowerBoundForClustering; K<=upperBoundForClustering; K++)
        {
            initializeCentroidPos(refCentroidPos, K);
            computeRefKMeans(K); //Saturation is checked within this function
            float val;
            computeRefSilhouette(K, val);
            if(val > bestSilhouetteValue)
            {
                bestClusterNums = K;
                bestSilhouetteValue = val;
            }

            refSilhouettesMap[K] = val;
        }
    }
}

int
KMeans::computeRefKMeans(int K)
{
    bool isSaturated = false;
    int iter=0;
    
    for(; (!isSaturated || iter>maxIter); iter++)
    {
        memset(refNewCentroidPos, 0, MAX_CLUSTERS * sizeof(cl_float2));
        memset(refCentroidPtsCount, 0, MAX_CLUSTERS * sizeof(cl_uint));
        
        for(int i=0; i<numPoints; ++i)
        {
            float leastDistSqr = MAX_FLOAT;
            float distSqr = 0;
            for(int j=0; j<K; j++)
            {
                distSqr = pow((refPointPos[i].s[0] - refCentroidPos[j].s[0]), 2.0f) + 
                          pow((refPointPos[i].s[1] - refCentroidPos[j].s[1]), 2.0f);
   
                if(distSqr <= leastDistSqr)
                {
                    leastDistSqr = distSqr;
                    refKMeansCluster[i] = j;
                }
            }
        }

        //Computing Centroid
        for(int i=0; i<numPoints; i++)
        {
            refNewCentroidPos[refKMeansCluster[i]].s[0] += refPointPos[i].s[0];
            refNewCentroidPos[refKMeansCluster[i]].s[1] += refPointPos[i].s[1];
            refCentroidPtsCount[refKMeansCluster[i]]++;
        }

        isSaturated = true;
        for(int i=0; i<K; i++)
        {
            if(refCentroidPtsCount[i] != 0) 
            {
                float newx, newy;
                
                newx = (refNewCentroidPos[i].s[0] / refCentroidPtsCount[i]);
                newy = (refNewCentroidPos[i].s[1] / refCentroidPtsCount[i]);  

                if((fabsf(refCentroidPos[i].s[0] - newx) > erfc)  ||
                        (fabsf(refCentroidPos[i].s[1] - newy) > erfc))
                {
                    isSaturated = false;
                }

                refCentroidPos[i].s[0] = newx;
                refCentroidPos[i].s[1] = newy;
            }
        }
    }
    return CL_SUCCESS;
}

int KMeans::computeRefSilhouette(int K, float& val)
{
    float silhouetteValue = 0.f;
    float* dissimilarities = new float[K];
    
    for(int point=0; point<numPoints; point++)
    {
        for(int i=0; i<K; i++)
        {
            dissimilarities[i] = 0.0;
        }
        
        for(int i=0; i<numPoints; i++)
        {
            if(point == i)
                continue;
            
            dissimilarities[refKMeansCluster[i]] += (sqrt(pow(refPointPos[i].s[0] - refPointPos[point].s[0], 2.0f)
                                                 + pow(refPointPos[i].s[1] - refPointPos[point].s[1], 2.0f)));
        }
        
        float a = dissimilarities[refKMeansCluster[point]] / refCentroidPtsCount[refKMeansCluster[point]];
        float b = FLT_MAX;
        for(int i=0; i<K; i++)
        {
            if((i != refKMeansCluster[point]) && (refCentroidPtsCount[i] != 0))
            {
                b = min(b, dissimilarities[i] / refCentroidPtsCount[i]);
            }
        }
        
        silhouetteValue += ((b - a) / max(a, b));
    }
    delete dissimilarities;
    val =  silhouetteValue / numPoints;
    return CL_SUCCESS;
}

int
KMeans::initialize()
{
    // Call base class Initialize to get default configuration
    int status = 0;
    if (sampleArgs->initialize() != SDK_SUCCESS)
        return SDK_FAILURE;

    Option *num_points = new Option;
    CHECK_ALLOCATION(num_points, "error. Failed to allocate memory (num_Points)\n");

    num_points->_sVersion = "x";
    num_points->_lVersion = "points";
    num_points->_description = "Number of Points";
    num_points->_type = CA_ARG_INT;
    num_points->_value = &numPoints;

    sampleArgs->AddOption(num_points);
    delete num_points;

    Option *num_clusters = new Option;
    CHECK_ALLOCATION(num_clusters, "error. Failed to allocate memory (num_Clusters)\n");

    num_clusters->_sVersion = "k";
    num_clusters->_lVersion = "clusters";
    num_clusters->_description = "Number of clusters( must be less than MAX_CLUSTERS)";
    num_clusters->_type = CA_ARG_INT;
    num_clusters->_value = &numClusters;

    sampleArgs->AddOption(num_clusters);
    delete num_clusters;

    Option *lBound_num_clusters = new Option;
    CHECK_ALLOCATION(lBound_num_clusters, "error. Failed to allocate memory (lBound_num_Clusters)\n");

    lBound_num_clusters->_sVersion = "lk";
    lBound_num_clusters->_lVersion = "lboundnumclusters";
    lBound_num_clusters->_description = "Lower bound for number of clusters for autoclustering";
    lBound_num_clusters->_type = CA_ARG_INT;
    lBound_num_clusters->_value = &lowerBoundForClustering;

    sampleArgs->AddOption(lBound_num_clusters);
    delete lBound_num_clusters;

    Option *ubound_num_clusters = new Option;
    CHECK_ALLOCATION(ubound_num_clusters, "error. Failed to allocate memory (uBound_num_Clusters)\n");

    ubound_num_clusters->_sVersion = "uk";
    ubound_num_clusters->_lVersion = "uboundnumclusters";
    ubound_num_clusters->_description = "Upper bound for number of clusters for autoclustering";
    ubound_num_clusters->_type = CA_ARG_INT;
    ubound_num_clusters->_value = &upperBoundForClustering;

    sampleArgs->AddOption(ubound_num_clusters);
    delete ubound_num_clusters;

    Option *rand_clusters_nums = new Option;
    CHECK_ALLOCATION(rand_clusters_nums, "error. Failed to allocate memory (uBound_num_Clusters)\n");

    rand_clusters_nums->_sVersion = "ck";
    rand_clusters_nums->_lVersion = "randinputcentroids";
    rand_clusters_nums->_description = "Number of random clusters created in input data(0-random)";
    rand_clusters_nums->_type = CA_ARG_INT;
    rand_clusters_nums->_value = &randClusterNums;

    sampleArgs->AddOption(rand_clusters_nums);
    delete rand_clusters_nums;

    Option *num_iterations = new Option;
    CHECK_ALLOCATION(num_iterations, "error. Failed to allocate memory (num_iterations)\n");

    num_iterations->_sVersion = "i";
    num_iterations->_lVersion = "iterations";
    num_iterations->_description = "Number of iterations";
    num_iterations->_type = CA_ARG_INT;
    num_iterations->_value = &iterations;

    sampleArgs->AddOption(num_iterations);
    delete num_iterations;

    return SDK_SUCCESS;
}

int
KMeans::setup()
{
    int status = 0;

    int timer = sampleTimer->createTimer();
    sampleTimer->resetTimer(timer);
    sampleTimer->startTimer(timer);
    status = setupCL();
    if(status != SDK_SUCCESS)
        return status;
    
    status = setupKMeans();
    if(status != SDK_SUCCESS)
        return status;
    
    sampleTimer->stopTimer(timer);
    // Compute setup time
    setupTime = (double)(sampleTimer->readTimer(timer));

    display = (!sampleArgs->quiet) && (!sampleArgs->verify) && (!sampleArgs->timing);

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
    if(h == 0) h=1;
    gluPerspective(45.0f, w/h, 1.0f, 1000.0f);
    gluLookAt (0.0, 0.0, -2.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0);
}

/** 
* @brief OpenGL display function
*/
void displayfunc()
{
    glClearColor(0.0 ,0.0, 0.0, 0.0);
    glClear(GL_COLOR_BUFFER_BIT);
    glClear(GL_DEPTH_BUFFER_BIT);

    glPointSize(1.0);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);
    glEnable(GL_BLEND);
    glDepthMask(GL_FALSE);

    KMeans *km = (KMeans *)me;
    int clusters;

    km->initializeCentroidPos(km->getclCentroidPos(), km->getNumClusters());
    if(!km->getIsSaturated()) // Should be useful in displaying gradual formation of clusters
    {
        km->computeKMeans(km->getNumClusters());
    }
    float val;
    km->computeSilhouette(km->getNumClusters(), val);
    clusters = km->getNumClusters();
    
    glBegin(GL_POINTS);
    //Creating AXIS
    glColor3f(1.0,1.0,1.0);
    for(double i=-1.0; i<1.0; i+=0.01)
    {
        glVertex3d(i,0.0,0.0);
        glVertex3d(0.0,i,0.0);
    }
    // Generating distinct colors for different clusters
    for(int i = 0; i < km->getNumPoints(); i++)
    {
        uint kmNumber = (km->KMeansCluster[i]+1) * 1729;
        uint RComp = (kmNumber % 203) + 32;
        uint GComp = (kmNumber % 201) + 32;
        uint BComp = (kmNumber % 207) + 32;
        glColor3f((RComp / 256.0), (GComp / 256.0), (BComp / 256.0));
        glVertex3d(((double)(km->pointPos[i].s[0])/MAX_COORD), ((double)(km->pointPos[i].s[1])/MAX_COORD), 0.0);
    }
    glEnd();

    //Printing Centroid points
    glPointSize(8.0);
    glBegin(GL_POINTS);
    glColor3f(1.0,1.0,1.0);
    for(int i=0; i<clusters; i++)
    {
        glVertex3d(((double)km->centroidPos[i].s[0]) / MAX_COORD, ((double)km->centroidPos[i].s[1]) / MAX_COORD, 0.0);
    }

    glEnd();

    glFlush();
    glutSwapBuffers();
    
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
            if(((KMeans*)me)->cleanup() != SDK_SUCCESS)
                exit(1);
            else
                exit(0);
        }

        default:
            break;
    }
}


int 
KMeans::run()
{
    int status = 0;
    // Arguments are set and execution call is enqueued on command buffer
    if((status=setupCLKernels()) != SDK_SUCCESS)
        return status;

    int timer = sampleTimer->createTimer();
    sampleTimer->resetTimer(timer);
    sampleTimer->startTimer(timer);

    for(int i = 0; i < iterations; ++i)
    {
        if(runCLKernels() != SDK_SUCCESS)
                return SDK_FAILURE;
    }

    status = clFinish(this->commandQueue);
    sampleTimer->stopTimer(timer);
    // Compute kernel time
    kernelTime = (double)(sampleTimer->readTimer(timer)) / iterations;

    if(!sampleArgs->quiet)
    {
        if(isNumClustersSpecified)
        {
            std::cout << "Silhouette Output:" << silhouettesMap[numClusters] << std::endl;
        }
        else
        {
            for(int i=lowerBoundForClustering; i<=upperBoundForClustering; i++)
            {
                std::cout << "For K:" << i << ", Silhouette Output:" << silhouettesMap[i] << std::endl;
            }
        }
    }

    return SDK_SUCCESS;
}

int
KMeans::verifyResults()
{
    if(sampleArgs->verify)
    {
        /* reference implementation
         * it overwrites the input array with the output
         */

        int timer = sampleTimer->createTimer();
        sampleTimer->resetTimer(timer);
        sampleTimer->startTimer(timer);
        
        KMeansCPUReference();
        
        sampleTimer->stopTimer(timer);
        // Compute kernel time
        refImplTime = (double)(sampleTimer->readTimer(timer));

        if(isNumClustersSpecified)
        {
            if(fabsf(silhouettesMap[numClusters] - refSilhouettesMap[numClusters]) > erfc)
            {
                std::cout << "Failed!\n" << std::endl;
                return SDK_FAILURE;
            }
        }
        else
        {
            // compare the results and see if they match
            for(int i=lowerBoundForClustering; i<=upperBoundForClustering; i++)
            {
                if(fabsf(silhouettesMap[i] - refSilhouettesMap[i]) > erfc)
                {
                    std::cout << "Failed!\n" << std::endl;
                    return SDK_FAILURE;
                }
            }
        }
        std::cout << "Passed!\n" << std::endl;
    }

    if(sampleArgs->timing)
    {
        printStats();
    }
    return SDK_SUCCESS;
}

void 
KMeans::printStats()
{
    std::string strArray[6] = 
    {
        "Points", 
        "Clusters",
        "Iterations", 
        "Setup time",
        "Avg kernelTime(sec)",
        "Ref Impl time(sec)"
    };

    std::string stats[6];
    
    stats[0] = toString(numPoints, std::dec);
    stats[1] = toString(getNumClusters(), std::dec);
    stats[2] = toString(iterations, std::dec);
    stats[3] = toString(setupTime, std::dec);
    stats[4] = toString(kernelTime, std::dec);
    stats[5] = toString(refImplTime, std::dec);

    if(sampleArgs->verify)
    {
        printStatistics(strArray, stats, 6);
    }
    else
    {
        printStatistics(strArray, stats, 5);
    }
}

int
KMeans::cleanup()
{
    // Releases OpenCL resources (Context, Memory etc.)
    cl_int status;

    status = clReleaseKernel(kernelAssignCentroid);
    CHECK_OPENCL_ERROR(status, "clReleaseKernel failed.(kernelAssignCentroid)");
    
    status = clReleaseKernel(kernelComputeSilhouette);
    CHECK_OPENCL_ERROR(status, "clReleaseKernel failed.(kernelComputeSilhouette)");

    status = clReleaseMemObject(clPointPos);
    CHECK_OPENCL_ERROR(status, "clReleaseMemObject failed.(clPointPos)");

    status = clReleaseMemObject(clKMeansCluster);
    CHECK_OPENCL_ERROR(status, "clReleaseMemObject failed.(clKMeansCluster)");

    status = clReleaseMemObject(clCentroidPos);
    CHECK_OPENCL_ERROR(status, "clReleaseMemObject failed.(clCentroidPos)");

    status = clReleaseMemObject(clNewCentroidPos);
    CHECK_OPENCL_ERROR(status, "clReleaseMemObject failed.(clNewCentroidPos)");
    
    status = clReleaseMemObject(clCentroidPtsCount);
    CHECK_OPENCL_ERROR(status, "clReleaseMemObject failed.(clCentroidPtsCount)");

	status = clReleaseMemObject(clSilhoutteValue);
    CHECK_OPENCL_ERROR(status, "clReleaseMemObject failed.(clSilhoutteValue)");

	status = clReleaseProgram(program);
    CHECK_OPENCL_ERROR(status, "clReleaseProgram failed.(program)");

    status = clReleaseCommandQueue(commandQueue);
    CHECK_OPENCL_ERROR(status, "clReleaseCommandQueue failed.(commandQueue)");

    status = clReleaseContext(context);
    CHECK_OPENCL_ERROR(status, "clReleaseContext failed.(context)");
    // release program resources 
    FREE(refPointPos);
    FREE(refKMeansCluster);
    FREE(refCentroidPos);
    FREE(refNewCentroidPos);
    FREE(refCentroidPtsCount);
    FREE(backupCentroidPos);
    FREE(KMeansCluster);

    FREE(devices);

    return SDK_SUCCESS;
}

KMeans::~KMeans()
{

}


int 
main(int argc, char * argv[])
{
    int status = 0;
    KMeans clKMeans;
    me = &clKMeans;

    if(clKMeans.initialize() != SDK_SUCCESS)
        return SDK_FAILURE;

    if (clKMeans.sampleArgs->parseCommandLine(argc, argv) != SDK_SUCCESS)
        return SDK_FAILURE;

    if(clKMeans.sampleArgs->isDumpBinaryEnabled())
    {
        return clKMeans.genBinaryImage();
    }

    status = clKMeans.setup();
    if(status != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    status = clKMeans.run();
    if(status != SDK_SUCCESS)
    {
        return status;
    }

    status = clKMeans.verifyResults();
    if(status != SDK_SUCCESS)
    {
        return status;
    }
    
    if(display)
    {
        std::ostringstream label;
        label << "KMeans simulation, With K=" << clKMeans.getNumClusters()
                << " and Silhouette=" << clKMeans.getSilhouetteMapValue(clKMeans.getNumClusters());
        
        ((KMeans*)me)->setIsSaturated(false);
        // Run in  graphical window if requested 
        glutInit(&argc, argv);
        glutInitWindowPosition(10,10);
        glutInitWindowSize(900,600); 
        glutInitDisplayMode( GLUT_RGB | GLUT_DOUBLE );
        glutCreateWindow(label.str().c_str()); 
        GLInit(); 
        glutDisplayFunc(displayfunc); 
        glutReshapeFunc(reShape);
        glutIdleFunc(idle); 
        glutKeyboardFunc(keyboardFunc);
        glutMainLoop();
    }

    status = clKMeans.cleanup();
    CHECK_ERROR(status, SDK_SUCCESS, "Sample CleanUP Failed");

    return SDK_SUCCESS;
}

