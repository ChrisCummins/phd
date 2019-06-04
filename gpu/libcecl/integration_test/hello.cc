//------------------------------------------------------------------------------
//
// Name:       vadd.c
//
// Purpose:    Elementwise addition of two vectors (c = a + b)
//
// HISTORY:    Written by Tim Mattson, December 2009
//             Updated by Tom Deakin and Simon McIntosh-Smith, October 2012
//             Updated by Tom Deakin, July 2013
//             Updated by Tom Deakin, October 2014
//
//------------------------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

const char *err_code(cl_int err_in) {
  switch (err_in) {
    case CL_SUCCESS:
      return (char *)"CL_SUCCESS";
    case CL_DEVICE_NOT_FOUND:
      return (char *)"CL_DEVICE_NOT_FOUND";
    case CL_DEVICE_NOT_AVAILABLE:
      return (char *)"CL_DEVICE_NOT_AVAILABLE";
    case CL_COMPILER_NOT_AVAILABLE:
      return (char *)"CL_COMPILER_NOT_AVAILABLE";
    case CL_MEM_OBJECT_ALLOCATION_FAILURE:
      return (char *)"CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case CL_OUT_OF_RESOURCES:
      return (char *)"CL_OUT_OF_RESOURCES";
    case CL_OUT_OF_HOST_MEMORY:
      return (char *)"CL_OUT_OF_HOST_MEMORY";
    case CL_PROFILING_INFO_NOT_AVAILABLE:
      return (char *)"CL_PROFILING_INFO_NOT_AVAILABLE";
    case CL_MEM_COPY_OVERLAP:
      return (char *)"CL_MEM_COPY_OVERLAP";
    case CL_IMAGE_FORMAT_MISMATCH:
      return (char *)"CL_IMAGE_FORMAT_MISMATCH";
    case CL_IMAGE_FORMAT_NOT_SUPPORTED:
      return (char *)"CL_IMAGE_FORMAT_NOT_SUPPORTED";
    case CL_BUILD_PROGRAM_FAILURE:
      return (char *)"CL_BUILD_PROGRAM_FAILURE";
    case CL_MAP_FAILURE:
      return (char *)"CL_MAP_FAILURE";
    case CL_MISALIGNED_SUB_BUFFER_OFFSET:
      return (char *)"CL_MISALIGNED_SUB_BUFFER_OFFSET";
    case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:
      return (char *)"CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
    case CL_INVALID_VALUE:
      return (char *)"CL_INVALID_VALUE";
    case CL_INVALID_DEVICE_TYPE:
      return (char *)"CL_INVALID_DEVICE_TYPE";
    case CL_INVALID_PLATFORM:
      return (char *)"CL_INVALID_PLATFORM";
    case CL_INVALID_DEVICE:
      return (char *)"CL_INVALID_DEVICE";
    case CL_INVALID_CONTEXT:
      return (char *)"CL_INVALID_CONTEXT";
    case CL_INVALID_QUEUE_PROPERTIES:
      return (char *)"CL_INVALID_QUEUE_PROPERTIES";
    case CL_INVALID_COMMAND_QUEUE:
      return (char *)"CL_INVALID_COMMAND_QUEUE";
    case CL_INVALID_HOST_PTR:
      return (char *)"CL_INVALID_HOST_PTR";
    case CL_INVALID_MEM_OBJECT:
      return (char *)"CL_INVALID_MEM_OBJECT";
    case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
      return (char *)"CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
    case CL_INVALID_IMAGE_SIZE:
      return (char *)"CL_INVALID_IMAGE_SIZE";
    case CL_INVALID_SAMPLER:
      return (char *)"CL_INVALID_SAMPLER";
    case CL_INVALID_BINARY:
      return (char *)"CL_INVALID_BINARY";
    case CL_INVALID_BUILD_OPTIONS:
      return (char *)"CL_INVALID_BUILD_OPTIONS";
    case CL_INVALID_PROGRAM:
      return (char *)"CL_INVALID_PROGRAM";
    case CL_INVALID_PROGRAM_EXECUTABLE:
      return (char *)"CL_INVALID_PROGRAM_EXECUTABLE";
    case CL_INVALID_KERNEL_NAME:
      return (char *)"CL_INVALID_KERNEL_NAME";
    case CL_INVALID_KERNEL_DEFINITION:
      return (char *)"CL_INVALID_KERNEL_DEFINITION";
    case CL_INVALID_KERNEL:
      return (char *)"CL_INVALID_KERNEL";
    case CL_INVALID_ARG_INDEX:
      return (char *)"CL_INVALID_ARG_INDEX";
    case CL_INVALID_ARG_VALUE:
      return (char *)"CL_INVALID_ARG_VALUE";
    case CL_INVALID_ARG_SIZE:
      return (char *)"CL_INVALID_ARG_SIZE";
    case CL_INVALID_KERNEL_ARGS:
      return (char *)"CL_INVALID_KERNEL_ARGS";
    case CL_INVALID_WORK_DIMENSION:
      return (char *)"CL_INVALID_WORK_DIMENSION";
    case CL_INVALID_WORK_GROUP_SIZE:
      return (char *)"CL_INVALID_WORK_GROUP_SIZE";
    case CL_INVALID_WORK_ITEM_SIZE:
      return (char *)"CL_INVALID_WORK_ITEM_SIZE";
    case CL_INVALID_GLOBAL_OFFSET:
      return (char *)"CL_INVALID_GLOBAL_OFFSET";
    case CL_INVALID_EVENT_WAIT_LIST:
      return (char *)"CL_INVALID_EVENT_WAIT_LIST";
    case CL_INVALID_EVENT:
      return (char *)"CL_INVALID_EVENT";
    case CL_INVALID_OPERATION:
      return (char *)"CL_INVALID_OPERATION";
    case CL_INVALID_GL_OBJECT:
      return (char *)"CL_INVALID_GL_OBJECT";
    case CL_INVALID_BUFFER_SIZE:
      return (char *)"CL_INVALID_BUFFER_SIZE";
    case CL_INVALID_MIP_LEVEL:
      return (char *)"CL_INVALID_MIP_LEVEL";
    case CL_INVALID_GLOBAL_WORK_SIZE:
      return (char *)"CL_INVALID_GLOBAL_WORK_SIZE";
    case CL_INVALID_PROPERTY:
      return (char *)"CL_INVALID_PROPERTY";
    default:
      return (char *)"UNKNOWN ERROR";
  }
}

void check_error(cl_int err, const char *operation, const char *filename,
                 const int line) {
  if (err != CL_SUCCESS) {
    fprintf(stderr, "Error during operation '%s', ", operation);
    fprintf(stderr, "in '%s' on line %d\n", filename, line);
    fprintf(stderr, "Error code was \"%s\" (%d)\n", err_code(err), err);
    exit(EXIT_FAILURE);
  }
}

#define checkError(E, S) check_error(E, S, __FILE__, __LINE__)

#ifndef DEVICE
#define DEVICE CL_DEVICE_TYPE_DEFAULT
#endif

#define TOL (0.001f)
#define LENGTH ((size_t)1024)

const char *KernelSource =
    "\n"
    "__kernel void vadd(\n"
    "   __global float* a,\n"
    "   __global float* b,\n"
    "   __global float* c,\n"
    "   const unsigned int count)\n"
    "{\n"
    "   int i = get_global_id(0);\n"
    "   if(i < count)\n"
    "       c[i] = a[i] + b[i];\n"
    "}\n"
    "\n";

int main(int argc, char **argv) {
  int err;

  float *h_a = (float *)calloc(LENGTH, sizeof(float));  // a vector
  float *h_b = (float *)calloc(LENGTH, sizeof(float));  // b vector
  float *h_c = (float *)calloc(LENGTH, sizeof(float));  // c vector (a+b)

  unsigned int correct;  // number of correct results
  size_t global;         // global domain size

  cl_context context;         // compute context
  cl_command_queue commands;  // compute command queue
  cl_program program;         // compute program
  cl_kernel ko_vadd;          // compute kernel

  cl_mem d_a;  // device memory used for the input  a vector
  cl_mem d_b;  // device memory used for the input  b vector
  cl_mem d_c;  // device memory used for the output c vector

  // Fill vectors a and b with random float values:
  size_t i = 0;
  for (i = 0; i < LENGTH; i++) {
    h_a[i] = rand() / (float)RAND_MAX;
    h_b[i] = rand() / (float)RAND_MAX;
  }

  // Set up platform and GPU device

  cl_uint numPlatforms;

  // Find number of platforms
  err = clGetPlatformIDs(0, NULL, &numPlatforms);
  checkError(err, "Finding platforms");
  if (numPlatforms == 0) {
    printf("Found 0 platforms!\n");
    return EXIT_FAILURE;
  }

  // Get all platforms
  cl_platform_id *Platform =
      (cl_platform_id *)malloc(sizeof(Platform) * numPlatforms);
  err = clGetPlatformIDs(numPlatforms, Platform, NULL);
  checkError(err, "Getting platforms");

  // Secure a GPU
  cl_device_id device_id = NULL;
  for (i = 0; i < numPlatforms; i++) {
    err = clGetDeviceIDs(Platform[i], DEVICE, 1, &device_id, NULL);
    if (err == CL_SUCCESS) break;
  }

  free(Platform);

  if (device_id == NULL) checkError(err, "Finding a device");

  // Create a compute context
  context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
  checkError(err, "Creating context");

  // Create a command queue
  commands = clCreateCommandQueue(context, device_id, 0, &err);
  checkError(err, "Creating command queue");

  // Create the compute program from the source buffer
  program = clCreateProgramWithSource(context, 1, (const char **)&KernelSource,
                                      NULL, &err);
  checkError(err, "Creating program");

  // Build the program
  err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
  if (err != CL_SUCCESS) {
    size_t len;
    char buffer[2048];

    printf("Error: Failed to build program executable!\n%s\n", err_code(err));
    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG,
                          sizeof(buffer), buffer, &len);
    printf("%s\n", buffer);
    return EXIT_FAILURE;
  }

  // Create the compute kernel from the program
  ko_vadd = clCreateKernel(program, "vadd", &err);
  checkError(err, "Creating kernel");

  // Create the input (a, b) and output (c) arrays in device memory
  d_a = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * LENGTH, NULL,
                       &err);
  checkError(err, "Creating buffer d_a");

  d_b = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * LENGTH, NULL,
                       &err);
  checkError(err, "Creating buffer d_b");

  d_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * LENGTH, NULL,
                       &err);
  checkError(err, "Creating buffer d_c");

  // Write a and b vectors into compute device memory
  err = clEnqueueWriteBuffer(commands, d_a, CL_TRUE, 0, sizeof(float) * LENGTH,
                             h_a, 0, NULL, NULL);
  checkError(err, "Copying h_a to device at d_a");

  err = clEnqueueWriteBuffer(commands, d_b, CL_TRUE, 0, sizeof(float) * LENGTH,
                             h_b, 0, NULL, NULL);
  checkError(err, "Copying h_b to device at d_b");

  // Set the arguments to our compute kernel
  size_t count = LENGTH;
  err = clSetKernelArg(ko_vadd, 0, sizeof(cl_mem), &d_a);
  err |= clSetKernelArg(ko_vadd, 1, sizeof(cl_mem), &d_b);
  err |= clSetKernelArg(ko_vadd, 2, sizeof(cl_mem), &d_c);
  err |= clSetKernelArg(ko_vadd, 3, sizeof(unsigned int), &count);
  checkError(err, "Setting kernel arguments");

  // Execute the kernel over the entire range of our 1d input data set
  // letting the OpenCL runtime choose the work-group size
  global = LENGTH;
  err = clEnqueueNDRangeKernel(commands, ko_vadd, 1, NULL, &global, NULL, 0,
                               NULL, NULL);
  checkError(err, "Enqueueing kernel");

  // Wait for the commands to complete before stopping the timer
  err = clFinish(commands);
  checkError(err, "Waiting for kernel to finish");

  // Read back the results from the compute device
  err = clEnqueueReadBuffer(commands, d_c, CL_TRUE, 0, sizeof(float) * LENGTH,
                            h_c, 0, NULL, NULL);
  if (err != CL_SUCCESS) {
    printf("Error: Failed to read output array!\n%s\n", err_code(err));
    exit(1);
  }

  // Test the results
  correct = 0;
  float tmp;

  for (i = 0; i < LENGTH; i++) {
    tmp = h_a[i] + h_b[i];  // assign element i of a+b to tmp
    tmp -= h_c[i];          // compute deviation of expected and output result
    if (tmp * tmp < TOL * TOL)
      ++correct;
    else {
      printf(" tmp %f h_a %f h_b %f h_c %f \n", tmp, h_a[i], h_b[i], h_c[i]);
    }
  }

  // summarise results
  printf("C = A+B:  %d out of %zu results were correct.\n", correct, LENGTH);

  // cleanup then shutdown
  clReleaseMemObject(d_a);
  clReleaseMemObject(d_b);
  clReleaseMemObject(d_c);
  clReleaseProgram(program);
  clReleaseKernel(ko_vadd);
  clReleaseCommandQueue(commands);
  clReleaseContext(context);

  free(h_a);
  free(h_b);
  free(h_c);

  return 0;
}
