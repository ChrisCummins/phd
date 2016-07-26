#ifndef CEC_PROFILE
#define CEC_PROFILE

#include <CL/cl.h>

// Get OpenCL event pointer.
cl_event* cec_event();

// Print profiling info for event.
void cec_profile(const char* name);
