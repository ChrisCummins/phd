#ifndef CEC_PROFILE
#define CEC_PROFILE

#include <stdio.h>
#include <stdlib.h>

#include <CL/cl.h>

#define CEC_COMMAND_QUEUE(context, device, properties, errorcode)       \
    clCreateCommandQueue(context,                                       \
                         device,                                        \
                         properties | CL_QUEUE_PROFILING_ENABLE,        \
                         errorcode)

#define CEC_ND_KERNEL(command_queue,                                    \
                      kernel,                                           \
                      work_dim,                                         \
                      global_work_offset,                               \
                      global_work_size,                                 \
                      local_work_size,                                  \
                      num_events_in_wait_list,                          \
                      event_wait_list)                                  \
    cec_profile_kernel(command_queue, kernel, work_dim,                 \
                       global_work_offset, global_work_size,            \
                       local_work_size, num_events_in_wait_list,        \
                       event_wait_list, #kernel);


#define CEC_TASK(command_queue,                                         \
                 kernel,                                                \
                 num_events_in_wait_list,                               \
                 event_wait_list,                                       \
                 __UNUSED__event)                                       \
    clEnqueueTask(command_queue, kernel, num_events_in_wait_list,       \
                  event_wait_list, cec_event());                        \
    cec_profile_kernel("clEnqueueNDRangeKernel " #kernel, 1);


#define CEC_READ_BUFFER(command_queue,                                  \
                        buffer,                                         \
                        blocking_read,                                  \
                        offset,                                         \
                        size,                                           \
                        ptr,                                            \
                        num_events_in_wait_list,                        \
                        event_wait_list,                                \
                        __UNUSED__event)                                \
    clEnqueueReadBuffer(command_queue,                                  \
                        buffer,                                         \
                        blocking_read,                                  \
                        offset,                                         \
                        size,                                           \
                        ptr,                                            \
                        num_events_in_wait_list,                        \
                        event_wait_list,                                \
                        cec_event());                                   \
    cec_profile_read("clEnqueueReadBuffer " #buffer, size);

#define CEC_WRITE_BUFFER(command_queue,                                 \
                         buffer,                                        \
                         blocking_write,                                \
                         offset,                                        \
                         size,                                          \
                         ptr,                                           \
                         num_events_in_wait_list,                       \
                         event_wait_list,                               \
                         __UNUSED__event)                               \
    clEnqueueWriteBuffer(command_queue,                                 \
                         buffer,                                        \
                         blocking_write,                                \
                         offset,                                        \
                         size,                                          \
                         ptr,                                           \
                         num_events_in_wait_list,                       \
                         event_wait_list,                               \
                         cec_event());                                  \
    cec_profile_write("clEnqueueWriteBuffer " #buffer, size);


#define CEC_CREATE_KERNEL(program,                                      \
                          kernel_name,                                  \
                          errcode)                                      \
    clCreateKernel(program, kernel_name, errcode);                      \
    fprintf(stderr, "\n[CEC] clCreateKernel %s\n", kernel_name);


// Get OpenCL event pointer.
cl_event* cec_event();

// Print profiling info for event.
cl_int cec_profile_kernel(cl_command_queue command_queue,
                          cl_kernel kernel,
                          cl_uint work_dim,
                          const size_t *global_work_offset,
                          const size_t *global_work_size,
                          const size_t *local_work_size,
                          cl_uint num_events_in_wait_list,
                          const cl_event *event_wait_list,
                          const char* name);
void cec_profile_read(const char* name, const size_t size);
void cec_profile_write(const char* name, const size_t size);


#endif  // CEC_PROFILE
