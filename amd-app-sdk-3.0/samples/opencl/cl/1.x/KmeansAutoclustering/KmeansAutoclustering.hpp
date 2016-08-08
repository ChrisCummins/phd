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


#ifndef KMEANS_H_
#define KMEANS_H_
#include <GL/glut.h>
#include "CLUtil.hpp"
#include "float.h"
#include <map>

using namespace appsdk;

#define SAMPLE_VERSION "AMD-APP-SDK-v3.0.130.3"
#define GROUP_SIZE      64
#define MAX_FLOAT       FLT_MAX
#define PI 3.14159265
#define ALIGNMENT 4096
#define uint unsigned int
#define MAX_COORD 10
#define MAX_PERCENT_TOLERENCE 1.0
#define MAX_CLUSTERS 16
#ifndef min
#define min(a,b) (((a) < (b)) ? (a) : (b))
#endif

#ifndef max
#define max(a,b) (((a) > (b)) ? (a) : (b))
#endif

/**
* KMeans 
* Class implements OpenCL  KMeans sample
* Derived from SDKSample base class
*/

class KMeans
{
private:
    cl_double setupTime;                /**< time taken to setup OpenCL resources and building kernel */
    cl_double kernelTime;               /**< time taken to run kernel and read result back */
    cl_double refImplTime;               /**< time taken to run Host Implementation */

    cl_float2 *refPointPos;             /**< Position of points for host side */
    cl_uint *refKMeansCluster; /**< To store cluster to which a point belongs to in reference implementation */ 
    cl_float2 *refCentroidPos;         /**< Current position of the centroids */
    cl_float2 *newCentroidPos;          /**< To store latest centrois position. This is a device-only buffer. */
    cl_float2 *refNewCentroidPos;       /**< To store latest centrois position for host implementation  */
    cl_uint *centroidPtsCount;          /**< To store count of points in a cluster */
    cl_uint *refCentroidPtsCount;          /**< To store count of points in a cluster */
    cl_float *silhouetteValue;          /**< To store output from kernel */
    std::map<int,float> silhouettesMap, refSilhouettesMap;
    int lowerBoundForClustering;   /**< Lower bound to start computing silhoutte value for*/
    int upperBoundForClustering;   /**< Upper bound to stop computing silhoutte value for*/
    
    bool isSaturated;               /**< To check whether solution has saturated*/
    int bestClusterNums;            /**< Best number of clusters, detected after silhoutte computation*/
    int numPoints;      /**< Number of input Points */
    float bestSilhouetteValue;
    bool isNumClustersSpecified;    /**<To display user specified cluster value*/
    int numClusters;    /**< Number of clusters in which input points are classified */
    static const unsigned int vecLen = 4;

    cl_context context;                 /**< CL context */
    cl_device_id *devices;              /**< CL device list */
    cl_mem clPointPos;                  /**< OpenCL buffer for point's position */
    cl_mem clKMeansCluster;             /**< OpenCL buffer for KMeansCluster */
    cl_mem clCentroidPos;           /**< OpenCL buffer for curr Centroid Position */
    cl_mem clNewCentroidPos;            /**< OpenCL buffer to store new Centroid position. */
    cl_mem clCentroidPtsCount;             /**< OpenCL buffer to store number of elements in a centroiod*/
    cl_mem clSilhoutteValue;            /**< OpenCL buffer to store output of silhouette values*/
    cl_command_queue commandQueue;      /**< CL command queue */
    cl_program program;                 /**< CL program */
    cl_kernel kernelAssignCentroid, kernelComputeSilhouette;         /**< CL kernel */
    size_t groupSize;                   /**< Work-Group size */

    int iterations;
    float erfc;
    int maxIter;                   /**< max iterations allowed when verifying */
    
    
    int randClusterNums;        /**< This value is used to create customized input data. We initially
                                        create customNumClusters numbers of clusters, and then try to 
                                        cluster them into numClusters number of clusters using K-Means*/
    
    SDKDeviceInfo         deviceInfo;            /**< Structure to store device information*/
    KernelWorkGroupInfo        kernelInfo;      /**< Structure to store kernel related info */
    SDKTimer    *sampleTimer;                   /**< SDKTimer object */

public:
    CLCommandArgs   *sampleArgs;   /**< CLCommand argument class */
    float random(int randMax, int randMin);
    int sanityCheck();
    int computeKMeans(int);
    int computeSilhouette(int, float&);
    int computeRefKMeans(int);
    int computeRefSilhouette(int, float&);
    void setInitialCentroidPos(int);
    void initializeCentroidPos(void *, int);
    void initializeCentroidPos(cl_mem, int);
    float getSilhouetteMapValue(int);
    bool getIsNumClustersSpecified();
    bool getIsSaturated();
    void setIsSaturated(bool val);
    int getNumClusters();
    int getNumPoints();
    float getBestSilhouetteValue();
    int getBestClusterNums();
    cl_mem getclCentroidPos();
    template<typename T> int mapBuffer(cl_mem, T* &, size_t, cl_map_flags);
    int unmapBuffer(cl_mem, void*);

    cl_float2 *pointPos;                /**< Position of point to be used in OpenCL implementation*/
    cl_uint *KMeansCluster;             /**< Cluster to which the point belongs to*/
    cl_float2 *centroidPos;         /**< Current position of the centroids */
    cl_float2 *backupCentroidPos;

    
    
    

    /** 
    * Constructor 
    * Initialize member variables
    */
    explicit KMeans():
        setupTime(0),
        kernelTime(0),
        refImplTime(0),
        pointPos(NULL),
        refPointPos(NULL),
        KMeansCluster(NULL),
        refKMeansCluster(NULL),
        centroidPos(NULL),
        refCentroidPos(NULL),
        newCentroidPos(NULL),
        refNewCentroidPos(NULL),
        centroidPtsCount(NULL),
        refCentroidPtsCount(NULL),
        clPointPos(NULL),
        clKMeansCluster(NULL),
        clCentroidPos(NULL),
        clNewCentroidPos(NULL),
        clCentroidPtsCount(NULL),
        devices(NULL),
        groupSize(GROUP_SIZE),
        iterations(1),
        isSaturated(false)
    {
        erfc = 1e-2;
        maxIter = 50;
        numPoints = 1024;
        numClusters = 0;
        lowerBoundForClustering = 2;
        upperBoundForClustering = 10;
        bestClusterNums = 0;
        bestSilhouetteValue = -1;
        isNumClustersSpecified = false;

        // Customized creation of inputs, based on numClusters value
        // creating numCluster number of random points
        randClusterNums = 0;// 0 means random input creation
        sampleArgs = new CLCommandArgs();
        sampleTimer = new SDKTimer();
		sampleArgs->sampleVerStr = SAMPLE_VERSION;
    }

    ~KMeans();

    /**
    * Check whether the centroid position had converged
    * true is saturated, false otherwise
    */
    int checkCentroidSaturation(int);

    /**
    * Allocate and initialize host memory array with random values
    * @return SDK_SUCCESS on success and SDK_FAILURE on failure
    */
    int setupKMeans();

    /**
     * Override from SDKSample, Generate binary image of given kernel 
     * and exit application
    * @return SDK_SUCCESS on success and SDK_FAILURE on failure
     */
    int genBinaryImage();

    /**
    * OpenCL related initialisations. 
    * Set up Context, Device list, Command Queue, Memory buffers
    * Build CL kernel program executable
    * @return SDK_SUCCESS on success and SDK_FAILURE on failure
    */
    int setupCL();

    /**
    * Set values for kernels' arguments
    * @return SDK_SUCCESS on success and SDK_FAILURE on failure
    */
    int setupCLKernels();

    /**
    * Enqueue calls to the kernels
    * on to the command queue, wait till end of kernel execution.
    * Get kernel start and end time if timing is enabled
    * @return SDK_SUCCESS on success and SDK_FAILURE on failure
    */
    int runCLKernels();

    /**
    * Reference CPU implementation of Binomial Option
    * for performance comparison
    */
    void KMeansCPUReference();

    /**
    * Override from SDKSample. Print sample stats.
    */
    void printStats();

    /**
    * Override from SDKSample. Initialize 
    * command line parser, add custom options
    * @return SDK_SUCCESS on success and SDK_FAILURE on failure
    */
    int initialize();

    /**
    * Override from SDKSample, adjust width and height 
    * of execution domain, perform all sample setup
    * @return SDK_SUCCESS on success and SDK_FAILURE on failure
    */
    int setup();

    /**
    * Override from SDKSample
    * Run OpenCL KMeans
    * @return SDK_SUCCESS on success and SDK_FAILURE on failure
    */
    int run();

    /**
    * Override from SDKSample
    * Cleanup memory allocations
    * @return SDK_SUCCESS on success and SDK_FAILURE on failure
    */
    int cleanup();

    /**
    * Override from SDKSample
    * Verify against reference implementation
    * @return SDK_SUCCESS on success and SDK_FAILURE on failure
    */
    int verifyResults();
};

#endif // KMEANS_H_

