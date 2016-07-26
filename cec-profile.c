#include <CL/cl.h>

#include <stdio.h>
#include <stdlib.h>

static cl_event cec_event_v;

cl_event* cec_event() {
    return &cec_event_v;
}

void cec_profile(cl_event event, const char* name) {
    clWaitForEvents(1, &event);
    cl_int err;
    cl_ulong start_time, end_time;

    err = clGetEventProfilingInfo(event,
                                  CL_PROFILING_COMMAND_QUEUED,
                                  sizeof(start_time), &start_time,
                                  NULL);
    if (err != CL_SUCCESS) {
        printf("[CEC] fatal: Kernel timer 1!");
        exit(104);
    }

    err = clGetEventProfilingInfo(event,
                                  CL_PROFILING_COMMAND_END,
                                  sizeof(end_time), &end_time,
                                  NULL);
    if (err != CL_SUCCESS) {
        printf("[CEC] fatal: Kernel timer 2!");
        exit(105);
    }

    double elapsed_ms = (double)(end_time - start_time) / 1000000.0;
    printf("\n[CEC] %s %.3f\n", name, elapsed_ms);
}
