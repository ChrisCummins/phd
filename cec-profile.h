#ifndef CEC_PROFILE
#define CEC_PROFILE

#include <CL/cl.h>

#define CEC_COMMAND_QUEUE(context, device, properties, errorcode) \
	clCreateCommandQueue(context, device, properties | CL_QUEUE_PROFILING_ENABLE, errorcode)

#define CEC_ND_KERNEL(command_queue, \
                      kernel, \
                      work_dim, \
                      global_work_offset, \
                      global_work_size, \
                      local_work_size, \
                      num_events_in_wait_list, \
                      event_wait_list) \
        clEnqueueNDRangeKernel(command_queue, kernel, work_dim, \
                               global_work_offset, global_work_size, \
                               local_work_size, num_events_in_wait_list, \
                               event_wait_list, cec_event()); \
        { \
            size_t cec_lws = 1; \
            for (cl_uint cec_i = 0; cec_i < work_dim; ++cec_i) \
                cec_lws *= local_work_size[cec_i]; \
            cec_profile("clEnqueueNDRangeKernel " #kernel, cec_lws); \
        }

// Get OpenCL event pointer.
cl_event* cec_event();

// Print profiling info for event.
void cec_profile(const char* name, const size_t wgsize);

#endif  // CEC_PROFILE
