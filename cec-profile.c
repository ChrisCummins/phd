#include "./cec-profile.h"

cl_event cec_event_v;


cl_event* cec_event() {
    return &cec_event_v;
}

cl_int cec_get_event_profiling_info(cl_event event,
                                    cl_profiling_info param_name,
                                    size_t param_value_size,
                                    void *param_value,
                                    size_t *param_value_size_ret) {
    cl_int err = clGetEventProfilingInfo(event, param_name, param_value_size,
                                         param_value, param_value_size_ret);
    if (err == CL_SUCCESS)
        return err;
    // error: fatal
    fprintf(stderr, "[CEC] ERROR: clGetEventProfilingInfo() failed! Cause: ");
    if (err == CL_PROFILING_INFO_NOT_AVAILABLE)
        fprintf(stderr, "profiling info not available!\n");
    else if (err == CL_INVALID_VALUE)
        fprintf(stderr, "invalid value!\n");
    else if (err == CL_INVALID_EVENT)
        fprintf(stderr, "bad event!\n");
    else
        fprintf(stderr, "unknown error!\n");
    exit(104);
}


cl_int cec_enqueue_nd_range_kernel(cl_command_queue command_queue,
                                   cl_kernel kernel,
                                   cl_uint work_dim,
                                   const size_t *global_work_offset,
                                   const size_t *global_work_size,
                                   const size_t *local_work_size,
                                   cl_uint num_events_in_wait_list,
                                   const cl_event *event_wait_list,
                                   cl_event *event) {
    cl_int err = clEnqueueNDRangeKernel(command_queue, kernel, work_dim,
                                        global_work_offset, global_work_size,
                                        local_work_size,
                                        num_events_in_wait_list,
                                        event_wait_list, event);

    if (err == CL_SUCCESS)
        return err;
    // error: fatal
    fprintf(stderr, "[CEC] ERROR: clEnqueueNDRangeKernel() failed! Cause: ");
    if (CL_INVALID_PROGRAM_EXECUTABLE)
        fprintf(stderr, "there is no successfully built program "
                "executable available for device associated with "
                "command_queue.\n");
    else if (CL_INVALID_COMMAND_QUEUE)
        fprintf(stderr, "command_queue is not a valid "
                "command-queue.\n");
    else if (CL_INVALID_KERNEL)
        fprintf(stderr, "kernel is not a valid kernel object.\n");
    else if (CL_INVALID_CONTEXT)
        fprintf(stderr, "context associated with command_queue "
                "and kernel is not the same or if the context associated with "
                "command_queue and events in event_wait_list are not the "
                "same.\n");
    else if (CL_INVALID_KERNEL_ARGS)
        fprintf(stderr, "the kernel argument values have not been "
                "specified.\n");
    else if (CL_INVALID_WORK_DIMENSION)
        fprintf(stderr, "work_dim is not a valid value "
                "(i.e. a value between 1 and 3).\n");
    else if (CL_INVALID_WORK_GROUP_SIZE)
        fprintf(stderr, "local_work_size is specified and number "
                "of work-items specified by global_work_size is not evenly "
                "divisable by size of work-group given by local_work_size or "
                "does not match the work-group size specified for kernel using "
                "the __attribute__((reqd_work_group_size(X, Y, Z))) qualifier "
                "in program source.\n");
    else if (CL_INVALID_WORK_GROUP_SIZE)
        fprintf(stderr, "local_work_size is specified and the "
                "total number of work-items in the work-group computed as "
                "local_work_size[0] *... local_work_size[work_dim - 1] is "
                "greater than the value specified by "
                "CL_DEVICE_MAX_WORK_GROUP_SIZE in the table of OpenCL "
                "Device Queries for clGetDeviceInfo.\n");
    else if (CL_INVALID_WORK_GROUP_SIZE)
        fprintf(stderr, "local_work_size is NULL and the "
                "__attribute__((reqd_work_group_size(X, Y, Z))) qualifier is "
                "used to declare the work-group size for kernel in the "
                "program source.\n");
    else if (CL_INVALID_WORK_ITEM_SIZE)
        fprintf(stderr, "the number of work-items specified in "
                "any of local_work_size[0], ... local_work_size[work_dim - 1] "
                "is greater than the corresponding values specified by "
                "CL_DEVICE_MAX_WORK_ITEM_SIZES[0], .... "
                "CL_DEVICE_MAX_WORK_ITEM_SIZES[work_dim - 1].\n");
    else if (CL_INVALID_GLOBAL_OFFSET)
        fprintf(stderr, "global_work_offset is not NULL.\n");
    else if (CL_OUT_OF_RESOURCES)
        fprintf(stderr, "there is a failure to queue the "
                "execution instance of kernel on the command-queue because "
                "of insufficient resources needed to execute the kernel. "
                "For example, the explicitly specified local_work_size "
                "causes a failure to execute the kernel because of "
                "insufficient resources such as registers or local memory. "
                "Another example would be the number of read-only image args "
                "used in kernel exceed the CL_DEVICE_MAX_READ_IMAGE_ARGS "
                "value for device or the number of write-only image args "
                "used in kernel exceed the CL_DEVICE_MAX_WRITE_IMAGE_ARGS "
                "value for device or the number of samplers used in kernel "
                "exceed CL_DEVICE_MAX_SAMPLERS for device.\n");
    else if (CL_MEM_OBJECT_ALLOCATION_FAILURE)
        fprintf(stderr, "there is a failure to allocate memory "
                "for data store associated with image or buffer objects "
                "specified as arguments to kernel.\n");
    else if (CL_INVALID_EVENT_WAIT_LIST)
        fprintf(stderr, "event_wait_list is NULL and "
                "num_events_in_wait_list > 0, or event_wait_list is not NULL "
                "and num_events_in_wait_list is 0, or if event objects in "
                "event_wait_list are not valid events.\n");
    else if (CL_OUT_OF_HOST_MEMORY)
        fprintf(stderr, "there is a failure to allocate "
                "resources required by the OpenCL implementation on the "
                "host.\n");
    else
        fprintf(stderr, "unknown error code!");
    exit(104);
}


cl_int cec_profile_kernel(cl_command_queue command_queue,
                          cl_kernel kernel,
                          cl_uint work_dim,
                          const size_t *global_work_offset,
                          const size_t *global_work_size,
                          const size_t *local_work_size,
                          cl_uint num_events_in_wait_list,
                          const cl_event *event_wait_list,
                          const char* name) {
    cl_int kern_err;
    cl_event event;
    cl_int profile_err;
    cl_ulong start_time, end_time;
    size_t wgsize = 1;
    cl_uint i;

    cec_enqueue_nd_range_kernel(command_queue, kernel, work_dim,
                                global_work_offset, global_work_size,
                                local_work_size, num_events_in_wait_list,
                                event_wait_list, &event);
    clWaitForEvents(1, &event);

    // Get elapsed time:
    cec_get_event_profiling_info(event, CL_PROFILING_COMMAND_QUEUED,
                                 sizeof(start_time), &start_time, NULL);
    cec_get_event_profiling_info(event, CL_PROFILING_COMMAND_END,
                                 sizeof(end_time), &end_time, NULL);

    double elapsed_ms = (double)(end_time - start_time) / 1000000.0;

    // Get number of work-items:
    for (i = 0; i < work_dim; ++i)
        wgsize *= (local_work_size)[i];

    fprintf(stderr, "\n[CEC] clEnqueueNDRangeKernel %s %zu %.3f\n",
            name, wgsize, elapsed_ms);

    return kern_err;
}


void cec_profile_read(const char* name, const size_t size) {
    clWaitForEvents(1, cec_event());

    cl_int err;
    cl_ulong start_time, end_time;

    cec_get_event_profiling_info(*cec_event(), CL_PROFILING_COMMAND_QUEUED,
                                 sizeof(start_time), &start_time, NULL);
    cec_get_event_profiling_info(*cec_event(), CL_PROFILING_COMMAND_END,
                                 sizeof(end_time), &end_time, NULL);

    double elapsed_ms = (double)(end_time - start_time) / 1000000.0;
    fprintf(stderr, "\n[CEC] %s %zu %.3f\n", name, size, elapsed_ms);
}


void cec_profile_write(const char* name, const size_t size) {
    clWaitForEvents(1, cec_event());

    cl_int err;
    cl_ulong start_time, end_time;

    cec_get_event_profiling_info(*cec_event(), CL_PROFILING_COMMAND_QUEUED,
                                 sizeof(start_time), &start_time, NULL);
    cec_get_event_profiling_info(*cec_event(), CL_PROFILING_COMMAND_END,
                                 sizeof(end_time), &end_time, NULL);

    double elapsed_ms = (double)(end_time - start_time) / 1000000.0;
    fprintf(stderr, "\n[CEC] %s %zu %.3f\n", name, size, elapsed_ms);
}
