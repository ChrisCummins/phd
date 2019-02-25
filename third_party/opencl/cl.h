/*
 * OpenCL include wrapper.
 */

#ifdef __APPLE__
# include <OpenCL/opencl.h>
# include <unistd.h>
#else
# include <CL/cl.h>
#endif
