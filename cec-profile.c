#include "./cec-profile.h"

#include <stdio.h>
#include <stdlib.h>

cl_event cec_event_v;

cl_event* cec_event() {
    return &cec_event_v;
}

void cec_profile(const char* name) {
    clWaitForEvents(1, cec_event());
    cl_int err;
    cl_ulong start_time, end_time;

    err = clGetEventProfilingInfo(*cec_event(),
                                  CL_PROFILING_COMMAND_QUEUED,
                                  sizeof(start_time), &start_time,
                                  NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "\n[CEC] fatal: Kernel timer 1! Did you use CL_QUEUE_PROFILING_ENABLE ?\n");
        exit(104);
    }

    err = clGetEventProfilingInfo(*cec_event(),
                                  CL_PROFILING_COMMAND_END,
                                  sizeof(end_time), &end_time,
                                  NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "\n[CEC] fatal: Kernel timer 2!\n");
        exit(105);
    }

    double elapsed_ms = (double)(end_time - start_time) / 1000000.0;
    fprintf(stderr, "\n[CEC] %s %.3f\n", name, elapsed_ms);
}
