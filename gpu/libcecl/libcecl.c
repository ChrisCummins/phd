/*
 * Libcecl implementation.
 *
 * Provides a blocking, verbose wrappers around a subset of the OpenCL
 * API.
 *
 * libcecl is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * libcecl is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with libcecl.  If not, see <http://www.gnu.org/licenses/>.
 */
#include "gpu/libcecl/libcecl.h"

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define E_CL_FAILURE 101

/*
 * Wraps clGetEventProfilingInfo.
 *
 * No return if fails.
 */
static cl_int cecl_get_event_profiling_info(cl_event event,
                                            cl_profiling_info param_name,
                                            size_t param_value_size,
                                            void* param_value,
                                            size_t* param_value_size_ret) {
    cl_int err = clGetEventProfilingInfo(event, param_name, param_value_size,
                                         param_value, param_value_size_ret);
    if (err == CL_SUCCESS)
        return err;
    /* error: fatal */
    fprintf(stderr, "\n[CECL] ERROR: clGetEventProfilingInfo() failed! "
            "Cause: ");
    if (err == CL_PROFILING_INFO_NOT_AVAILABLE)
        fprintf(stderr, "profiling info not available!\n");
    else if (err == CL_INVALID_VALUE)
        fprintf(stderr, "invalid value!\n");
    else if (err == CL_INVALID_EVENT)
        fprintf(stderr, "bad event!\n");
    else
        fprintf(stderr, "unknown error!\n");
    exit(E_CL_FAILURE);
}


/*
 * Wraps clGetDeviceInfo.
 *
 * No return if fails.
 */
static cl_int cecl_get_device_info(cl_device_id device,
                                   cl_device_info param_name,
                                   size_t param_value_size,
                                   void* param_value,
                                   size_t* param_value_size_ret) {
    cl_int err = clGetDeviceInfo(device, param_name, param_value_size,
                                 param_value, param_value_size_ret);


    if (err == CL_SUCCESS)
        return err;
    /* error: fatal */
    if (err == CL_INVALID_DEVICE)
        fprintf(stderr, "if device is not valid.\n");
    else if (err == CL_INVALID_VALUE)
        fprintf(stderr, "param_name is not one of the supported values or "
                "size in bytes specified by param_value_size is less than "
                "size of return type as shown in the table above and "
                "param_value is not a NULL value.\n");
    else
        fprintf(stderr, "unknown error!\n");
    exit(E_CL_FAILURE);
}


/*
 * Wraps clGetKernelInfo.
 *
 * No return if fails.
 */
static cl_int cecl_get_kernel_info(cl_kernel kernel,
                                   cl_kernel_info param_name,
                                   size_t param_value_size,
                                   void* param_value,
                                   size_t* param_value_size_ret) {
    cl_int err = clGetKernelInfo(kernel, param_name, param_value_size,
                                 param_value, param_value_size_ret);

    if (err == CL_SUCCESS)
        return err;
    /* error: fatal */
    if (err == CL_INVALID_VALUE)
        fprintf(stderr, "param_name is not valid, or size in bytes specified "
                "by param_value_size is less than the size of return type "
                "and param_value is not NULL\n");
    else if (err == CL_INVALID_KERNEL)
        fprintf(stderr, "kernel is not a valid kernel object.\n");
    else if (err == CL_OUT_OF_RESOURCES)
        fprintf(stderr, "there is a failure to allocate resources required by "
                "the OpenCL implementation on the device.\n");
    else if (err == CL_OUT_OF_HOST_MEMORY)
        fprintf(stderr, "there is a failure to allocate resources required by "
                "the OpenCL implementation on the host.\n");
    else
        fprintf(stderr, "unknown error!\n");
    exit(E_CL_FAILURE);
}


/*
 * Wraps clWaitForEvents.
 *
 * No return if fails.
 */
static void cec_wait_for_event(cl_event* event) {
    cl_int err = clWaitForEvents(1, event);

    if (err == CL_SUCCESS) {
        return;
    }
    /* error: fatal */
    fprintf(stderr, "\n[CECL] ERROR: clWaitForEvents() failed! Cause: ");
    if (err == CL_INVALID_VALUE)  {
        fprintf(stderr, "num_events is zero.\n");
    } else if (err == CL_INVALID_CONTEXT) {
        fprintf(stderr, "events specified in event_list do not belong to the "
                "same context.\n");
    } else if (err == CL_INVALID_EVENT) {
        fprintf(stderr, "event objects specified in event_list are not valid "
                "event objects.\n");
    }
    exit(E_CL_FAILURE);
}


/*
 * Block until OpenCL event completes, then return the elapsed time in
 * milliseconds.
 *
 * No return if fails.
 */
static double cecl_get_elapsed_ms(cl_event event) {
    cl_ulong start_time, end_time;

    cec_wait_for_event(&event);

    cecl_get_event_profiling_info(event, CL_PROFILING_COMMAND_QUEUED,
                                  sizeof(start_time), &start_time, NULL);
    cecl_get_event_profiling_info(event, CL_PROFILING_COMMAND_END,
                                  sizeof(end_time), &end_time, NULL);

    return (double)(end_time - start_time) / 1000000.0;
}


/* External: see cecl.h for documentation. */


cl_int cecl_nd_range_kernel(cl_command_queue command_queue,
                            cl_kernel kernel,
                            cl_uint work_dim,
                            const size_t* global_work_offset,
                            const size_t* global_work_size,
                            const size_t* local_work_size,
                            cl_uint num_events_in_wait_list,
                            const cl_event* event_wait_list,
                            cl_event* event) {
    cl_event local_event;

    if (!event) {  /* in case event points to NULL */
        event = &local_event;
    }

    cl_int err = clEnqueueNDRangeKernel(command_queue, kernel, work_dim,
                                        global_work_offset, global_work_size,
                                        local_work_size,
                                        num_events_in_wait_list,
                                        event_wait_list, event);

    if (err == CL_SUCCESS) {
        char kernel_name[255];
        cl_uint i;
        size_t gsize = 1;
        size_t wgsize = 1;

        for (i = 0; i < work_dim; ++i)
            gsize *= (global_work_size)[i];

        /* Get number of work-items: */
        if (local_work_size) {
            for (i = 0; i < work_dim; ++i)
                wgsize *= (local_work_size)[i];
        } else {
            wgsize = 0;
        }

        cecl_get_kernel_info(kernel, CL_KERNEL_FUNCTION_NAME,
                             sizeof(kernel_name), kernel_name, NULL);
        double elapsed_ms = cecl_get_elapsed_ms(*event);

        fprintf(stderr,
                "\n[CECL] clEnqueueNDRangeKernel ; %s ; %lu ; %lu ; %.6f\n",
                kernel_name, (unsigned long)gsize, (unsigned long)wgsize,
                elapsed_ms);
        return err;
    }
    /* error: fatal */
    fprintf(stderr, "\n[CECL] ERROR: clEnqueueNDRangeKernel() failed! Cause: ");
    if (err == CL_INVALID_PROGRAM_EXECUTABLE)
        fprintf(stderr, "there is no successfully built program "
                "executable available for device associated with "
                "command_queue.\n");
    else if (err == CL_INVALID_COMMAND_QUEUE)
        fprintf(stderr, "command_queue is not a valid "
                "command-queue.\n");
    else if (err == CL_INVALID_KERNEL)
        fprintf(stderr, "kernel is not a valid kernel object.\n");
    else if (err == CL_INVALID_CONTEXT)
        fprintf(stderr, "context associated with command_queue "
                "and kernel is not the same or if the context associated with "
                "command_queue and events in event_wait_list are not the "
                "same.\n");
    else if (err == CL_INVALID_KERNEL_ARGS)
        fprintf(stderr, "the kernel argument values have not been "
                "specified.\n");
    else if (err == CL_INVALID_WORK_DIMENSION)
        fprintf(stderr, "work_dim is not a valid value "
                "(i.e. a value between 1 and 3).\n");
    else if (err == CL_INVALID_WORK_GROUP_SIZE)
        fprintf(stderr, "local_work_size is specified and number "
                "of work-items specified by global_work_size is not evenly "
                "divisable by size of work-group given by local_work_size or "
                "does not match the work-group size specified for kernel using "
                "the __attribute__((reqd_work_group_size(X, Y, Z))) qualifier "
                "in program source.\n");
    else if (err == CL_INVALID_WORK_GROUP_SIZE)
        fprintf(stderr, "local_work_size is specified and the "
                "total number of work-items in the work-group computed as "
                "local_work_size[0] *... local_work_size[work_dim - 1] is "
                "greater than the value specified by "
                "CL_DEVICE_MAX_WORK_GROUP_SIZE in the table of OpenCL "
                "Device Queries for clGetDeviceInfo.\n");
    else if (err == CL_INVALID_WORK_GROUP_SIZE)
        fprintf(stderr, "local_work_size is NULL and the "
                "__attribute__((reqd_work_group_size(X, Y, Z))) qualifier is "
                "used to declare the work-group size for kernel in the "
                "program source.\n");
    else if (err == CL_INVALID_WORK_ITEM_SIZE)
        fprintf(stderr, "the number of work-items specified in "
                "any of local_work_size[0], ... local_work_size[work_dim - 1] "
                "is greater than the corresponding values specified by "
                "CL_DEVICE_MAX_WORK_ITEM_SIZES[0], .... "
                "CL_DEVICE_MAX_WORK_ITEM_SIZES[work_dim - 1].\n");
    else if (err == CL_INVALID_GLOBAL_OFFSET)
        fprintf(stderr, "global_work_offset is not NULL.\n");
    else if (err == CL_OUT_OF_RESOURCES)
        fprintf(stderr, "there is a failure to queue the "
                "execution instance of kernel on the command-queue because "
                "of insufficient resources needed to execute the kernel.\n");
    else if (err == CL_MEM_OBJECT_ALLOCATION_FAILURE)
        fprintf(stderr, "there is a failure to allocate memory "
                "for data store associated with image or buffer objects "
                "specified as arguments to kernel.\n");
    else if (err == CL_INVALID_EVENT_WAIT_LIST)
        fprintf(stderr, "event_wait_list is NULL and "
                "num_events_in_wait_list > 0, or event_wait_list is not NULL "
                "and num_events_in_wait_list is 0, or if event objects in "
                "event_wait_list are not valid events.\n");
    else if (err == CL_OUT_OF_HOST_MEMORY)
        fprintf(stderr, "there is a failure to allocate "
                "resources required by the OpenCL implementation on the "
                "host.\n");
    else
        fprintf(stderr, "unknown error code!\n");
    exit(E_CL_FAILURE);
}


cl_int cecl_task(cl_command_queue command_queue,
                 cl_kernel kernel,
                 cl_uint num_events_in_wait_list,
                 const cl_event* event_wait_list,
                 cl_event* event) {
    cl_event local_event;

    if (!event) {  /* in case event points to NULL */
        event = &local_event;
    }

    cl_int err = clEnqueueTask(command_queue, kernel, num_events_in_wait_list,
                               event_wait_list, event);

    if (err == CL_SUCCESS) {
        char kernel_name[255];

        cecl_get_kernel_info(kernel, CL_KERNEL_FUNCTION_NAME,
                             sizeof(kernel_name), kernel_name, NULL);
        double elapsed_ms = cecl_get_elapsed_ms(*event);

        fprintf(stderr, "\n[CECL] clEnqueueTask ; %s ; %.6f\n",
                kernel_name, elapsed_ms);
        return err;
    }
    /* error: fatal */
    fprintf(stderr, "\n[CECL] ERROR: clEnqueueTask() failed! Cause: ");
    if (err == CL_INVALID_PROGRAM_EXECUTABLE)
        fprintf(stderr, "there is no successfully built program executable "
                "available for device associated with command_queue.\n");
    else if (err == CL_INVALID_COMMAND_QUEUE)
        fprintf(stderr, "command_queue is not a valid command-queue.\n");
    else if (err == CL_INVALID_KERNEL)
        fprintf(stderr, "kernel is not a valid kernel object.\n");
    else if (err == CL_INVALID_CONTEXT)
        fprintf(stderr, "context associated with command_queue and kernel is "
                "not the same or if the context associated with command_queue "
                "and events in event_wait_list are not the same.\n");
    else if (err == CL_INVALID_KERNEL_ARGS)
        fprintf(stderr, "the kernel argument values have not been "
                "specified.\n");
    else if (err == CL_INVALID_WORK_GROUP_SIZE)
        fprintf(stderr, "a work-group size is specified for kernel using the "
                "__attribute__((reqd_work_group_size(X, Y, Z))) qualifier in "
                "program source and is not (1, 1, 1).\n");
    else if (err == CL_OUT_OF_RESOURCES)
        fprintf(stderr, "there is a failure to queue the execution instance "
                "of kernel on the command-queue because of insufficient "
                "resources needed to execute the kernel.\n");
    else if (err == CL_MEM_OBJECT_ALLOCATION_FAILURE)
        fprintf(stderr, "there is a failure to allocate memory for data "
                "store associated with image or buffer objects specified as "
                "arguments to kernel.\n");
    else if (err == CL_INVALID_EVENT_WAIT_LIST)
        fprintf(stderr, "event_wait_list is NULL and num_events_in_wait_list "
                "is greater than 0, or event_wait_list is not NULL and "
                "num_events_in_wait_list is 0, or if event objects in "
                "event_wait_list are not valid events.\n");
    else if (err == CL_OUT_OF_HOST_MEMORY)
        fprintf(stderr, "there is a failure to allocate resources required by "
                "the OpenCL implementation on the host.\n");
    else
        fprintf(stderr, "unknown error code!\n");
    exit(E_CL_FAILURE);
}


static cl_platform_id cecGetForcedPlatformIdOrDie() {
  const char* target_platform = getenv("LIBCECL_PLATFORM");
  if (target_platform == NULL) {
    fprintf(stderr, "[CECL] Required environment variable "
                    "LIBCECL_PLATFORM not set!\n");
    exit(E_CL_FAILURE);
  }

  cl_uint platform_count;
  cl_int local_err = clGetPlatformIDs(0, NULL, &platform_count);
  if (local_err != CL_SUCCESS) {
    fprintf(stderr, "Cannot get platform IDs!\n");
    exit(E_CL_FAILURE);
  }

  cl_platform_id platforms[platform_count];
  local_err = clGetPlatformIDs(platform_count, platforms, NULL);
  if (local_err != CL_SUCCESS) {
    fprintf(stderr, "Cannot get the list of OpenCL platforms\n");
    exit(E_CL_FAILURE);
  }

  for (size_t i = 0; i < platform_count; ++i) {
    size_t buffer_size;
    local_err = clGetPlatformInfo(
        platforms[i], CL_PLATFORM_NAME, 0, NULL, &buffer_size);
    if (local_err != CL_SUCCESS) {
      fprintf(stderr, "Cannot get the size of the CL_PLATFORM_NAME "
                      "parameter\n");
      exit(E_CL_FAILURE);
    }

    char* buffer = malloc(buffer_size);
    local_err = clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, buffer_size,
                                  buffer, NULL);
    if (local_err != CL_SUCCESS) {
      fprintf(stderr, "Cannot get the CL_PLATFORM_NAME parameter\n");
      exit(E_CL_FAILURE);
    }

    if (!strcmp(buffer, target_platform)) {
      free(buffer);
      return platforms[i];
    }

    free(buffer);
  }

  fprintf(stderr, "Failed to get a platform with matching name %s\n",
          target_platform);
  exit(E_CL_FAILURE);
}


static cl_device_id cecGetForcedDeviceIdOrDie() {
  cl_platform_id platform = cecGetForcedPlatformIdOrDie();

  cl_uint device_count;
  const char* target_device = getenv("LIBCECL_DEVICE");
  if (target_device == NULL) {
      fprintf(stderr, "[CECL] Required environment variable LIBCECL_DEVICE "
                      "not set!\n");
      exit(E_CL_FAILURE);
  }

  cl_int local_err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &device_count);
  if (local_err != CL_SUCCESS) {
    fprintf(stderr, "Cannot get the number of devices\n");
    exit(E_CL_FAILURE);
  }

  cl_device_id devices[device_count];
  local_err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, device_count, devices, NULL);
  if (local_err != CL_SUCCESS) {
    fprintf(stderr, "Cannot get device IDs\n");
    exit(E_CL_FAILURE);
  }

  for (size_t i = 0; i < device_count; ++i) {
    size_t buffer_size;
    cecl_get_device_info(devices[i], CL_DEVICE_NAME, 0, NULL, &buffer_size);

    char* buffer = malloc(buffer_size);
    cecl_get_device_info(devices[i], CL_DEVICE_NAME, buffer_size, buffer, NULL);

    if (!strcmp(buffer, target_device)) {
      free(buffer);
      return devices[i];
    }

    free(buffer);
  }

  fprintf(stderr, "Failed to get a device with matching name %s\n",
          target_device);
  exit(E_CL_FAILURE);
}

cl_context CECL_CREATE_CONTEXT(cl_context_properties *properties,
                               cl_uint num_devices,
                               const cl_device_id *unused_devices,
                               void *unused_pfn_notify(const char *errinfo,
                                                       const void *private_info,
                                                       size_t cb,
                                                       void *user_data),
                               void *user_data,
                               cl_int *errcode_ret) {
  if (num_devices != 1) {
    fprintf(stderr, "[CECL] libcecl only supports OpenCL contexts for 1 "
            "device!");
    exit(E_CL_FAILURE);
  }

  const cl_device_id device_id = cecGetForcedDeviceIdOrDie();

  return clCreateContext(
      properties, 1, &device_id, NULL, user_data, errcode_ret);
}


cl_context CECL_CREATE_CONTEXT_FROM_TYPE(cl_context_properties *properties,
                                         cl_device_type device_type_unused,
                                         void *pfn_notify (const char *errinfo,
                                                           const void *private_info,
                                                           size_t cb,
                                                           void *user_data),
                                         void *user_data,
                                         cl_int *errcode_ret) {
    return CECL_CREATE_CONTEXT(properties, 1, NULL, NULL, NULL, errcode_ret);
}


static cl_context cecGetForcedContextOrDie() {
  cl_int err;
  const cl_device_id device_id = cecGetForcedDeviceIdOrDie();
  cl_context ctx = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
  if (err != CL_SUCCESS) {
    fprintf(stderr, "[CEC] Failed to create forced context\n");
    exit(E_CL_FAILURE);
  }

  return ctx;
}


// Create a command queue. The device argument is ignored, instead, the
// LICECL_PLATFORM and LIBCECL_DEVICE arguments are used to identify the
// device to create the command queue for.
cl_command_queue CECL_CREATE_COMMAND_QUEUE(cl_context context,
                                           cl_device_id device_unused,
                                           cl_command_queue_properties props,
                                           cl_int* err) {
    cl_int local_err;

    // cl_context context = cecGetForcedContextOrDie();

    cl_device_id device_id = cecGetForcedDeviceIdOrDie();
    cl_command_queue q = clCreateCommandQueue(
        context, device_id, props | CL_QUEUE_PROFILING_ENABLE, &local_err);
    if (local_err == CL_SUCCESS) {
      cl_device_type devtype;
      char devname[100];

      cecl_get_device_info(device_id, CL_DEVICE_TYPE, sizeof(devtype),
                           &devtype, NULL);
      cecl_get_device_info(device_id, CL_DEVICE_NAME, sizeof(devname),
                           devname, NULL);

      fprintf(stderr, "\n[CECL] clCreateCommandQueue ; ");
      if (devtype == CL_DEVICE_TYPE_CPU) {
          fprintf(stderr, "CPU");
      } else if (devtype == CL_DEVICE_TYPE_GPU) {
          fprintf(stderr, "GPU");
      } else {
          fprintf(stderr, "UNKNOWN");
      }
      fprintf(stderr, " ; %s\n", devname);

      if (err) {
          *err = local_err;
      }

      return q;
    }
    /* error: fatal */
    fprintf(stderr, "\n[CECL] ERROR: clCreateCommandQueue() failed! Cause: ");
    if (local_err == CL_INVALID_CONTEXT)
        fprintf(stderr, "context is not a valid context.\n");
    else if (local_err == CL_INVALID_DEVICE)
        fprintf(stderr, "device is not a valid device or is not associated "
                "with context.\n");
    else if (local_err == CL_INVALID_VALUE)
        fprintf(stderr, "values specified in properties are not valid.\n");
    else if (local_err == CL_INVALID_QUEUE_PROPERTIES)
        fprintf(stderr, "values specified in properties are valid but are not "
                "supported by the device.\n");
    else if (local_err == CL_OUT_OF_HOST_MEMORY)
        fprintf(stderr, "there is a failure to allocate resources required by "
                "the OpenCL implementation on the host.\n");
    else
        fprintf(stderr, "unknown error code!");
    exit(E_CL_FAILURE);
}


cl_int CECL_GET_KERNEL_WORK_GROUP_INFO(cl_kernel kernel,
                                       cl_device_id unused_device,
                                       cl_kernel_work_group_info param_name,
                                       size_t param_value_size,
                                       void *param_value,
                                       size_t *param_value_size_ret) {
  cl_device_id device_id = cecGetForcedDeviceIdOrDie();
  return clGetKernelWorkGroupInfo(kernel, device_id, param_name,
                                  param_value_size, param_value,
                                  param_value_size_ret);
}


cl_program CECL_PROGRAM_WITH_SOURCE(cl_context context,
                                    cl_uint count,
                                    const char** strings,
                                    const size_t* lengths,
                                    cl_int* err) {
    // cl_context context = cecGetForcedContextOrDie();

    cl_int local_err;
    cl_uint i;
    cl_program p = clCreateProgramWithSource(context,
                                             count,
                                             strings,
                                             lengths,
                                             &local_err);

    fprintf(stderr, "\n[CECL] clCreateProgramWithSource\n"
            "[CECL] BEGIN PROGRAM SOURCE\n");
    for (i = 0; i < count; ++i) {
        const char *c = strings[i];
        fprintf(stderr, "[CECL] ");
        while (*c != '\0') {
            if (*c == '\n')
                fprintf(stderr, "\n[CECL] ");
            else
                fprintf(stderr, "%c", *c);
            ++c;
        }
    }
    fprintf(stderr, "\n[CECL] END PROGRAM SOURCE\n");

    if (local_err == CL_SUCCESS) {
        if (err) {
            *err = local_err;
        }

        return p;
    }
    /* error: fatal */
    fprintf(stderr, "\n[CECL] ERROR: clCreateProgramWithSource() failed! "
            "Cause: ");
    if (local_err == CL_INVALID_CONTEXT)
        fprintf(stderr, "context is not a valid context.\n");
    else if (local_err == CL_INVALID_VALUE)
        fprintf(stderr, "count is zero or if strings or any entry in strings "
                "is NULL.\n");
    else if (local_err == CL_OUT_OF_HOST_MEMORY)
        fprintf(stderr, "there is a failure to allocate resources required "
                "by the OpenCL implementation on the host.\n");
    else
        fprintf(stderr, "unknown error code!\n");
    exit(E_CL_FAILURE);
}


cl_int cecl_program(cl_program program,
                    cl_uint num_devices,
                    const cl_device_id* device_list,
                    const char* options,
                    void (*pfn_notify)(cl_program, void* user_data),
                    void* user_data,
                    const char* program_name) {
    cl_int err = clBuildProgram(program, num_devices, device_list, options,
                                pfn_notify, user_data);

    fprintf(stderr, "\n[CECL] clBuildProgram ; %s\n", program_name);

    if (err == CL_SUCCESS)
        return err;
    /* error: fatal */
    if (err == CL_INVALID_PROGRAM)
        fprintf(stderr, "program is not a valid program object.\n");
    else if (err == CL_INVALID_VALUE)
        fprintf(stderr, "device_list is NULL and num_devices is greater than "
                "zero, or if device_list is not NULL and num_devices is "
                "zero.\n");
    else if (err == CL_INVALID_VALUE)
        fprintf(stderr, "pfn_notify is NULL but user_data is not NULL.\n");
    else if (err == CL_INVALID_DEVICE)
        fprintf(stderr, "OpenCL devices listed in device_list are not in the "
                "list of devices associated with program.\n");
    else if (err == CL_INVALID_BINARY)
        fprintf(stderr, "program is created with clCreateWithProgramWithBinary "
                "and devices listed in device_list do not have a valid program "
                "binary loaded.\n");
    else if (err == CL_INVALID_BUILD_OPTIONS)
        fprintf(stderr, "the build options specified by options are "
                "invalid.\n");
    else if (err == CL_INVALID_OPERATION)
        fprintf(stderr, "the build of a program executable for any of the "
                "devices listed in device_list by a previous call to "
                "clBuildProgram for program has not completed.\n");
    else if (err == CL_COMPILER_NOT_AVAILABLE)
        fprintf(stderr, "program is created with clCreateProgramWithSource and "
                "a compiler is not available i.e. CL_DEVICE_COMPILER_AVAILABLE "
                "specified in the table of OpenCL Device Queries for "
                "clGetDeviceInfo is set to CL_FALSE.\n");
    else if (err == CL_BUILD_PROGRAM_FAILURE)
        fprintf(stderr, "there is a failure to build the program executable. "
                "This error will be returned if clBuildProgram does not "
                "return until the build has completed.\n");
    else if (err == CL_INVALID_OPERATION)
        fprintf(stderr, "there are kernel objects attached to program.\n");
    else if (err == CL_OUT_OF_HOST_MEMORY)
        fprintf(stderr, "there is a failure to allocate resources required by "
                "the OpenCL implementation on the host.\n");
    else
        fprintf(stderr, "unknown err code!\n");
    exit(E_CL_FAILURE);
}


cl_kernel cecl_kernel(cl_program  program,
                      const char* program_name,
                      const char* kernel_name,
                      cl_int* err) {
    cl_int local_err;
    cl_kernel k = clCreateKernel(program, kernel_name, &local_err);

    fprintf(stderr, "\n[CECL] clCreateKernel ; %s ; %s\n",
            program_name, kernel_name);

    if (local_err == CL_SUCCESS) {
        if (err) {
            *err = local_err;
        }

        return k;
    }
    /* error: fatal */
    fprintf(stderr, "\n[CECL] ERROR: clCreateKernel() failed! Cause: ");
    if (local_err == CL_INVALID_PROGRAM)
        fprintf(stderr, "program is not a valid program object.\n");
    else if (local_err == CL_INVALID_PROGRAM_EXECUTABLE)
        fprintf(stderr, "there is no successfully built executable for "
                "program.\n");
    else if (local_err == CL_INVALID_KERNEL_NAME)
        fprintf(stderr, "kernel_name is not found in program.\n");
    else if (local_err == CL_INVALID_KERNEL_DEFINITION)
        fprintf(stderr, "the function definition for __kernel function given "
                "by kernel_name such as the number of arguments, the argument "
                "types are not the same for all devices for which the program "
                "executable has been built.\n");
    else if (local_err == CL_INVALID_VALUE)
        fprintf(stderr, "kernel_name is NULL.\n");
    else if (local_err == CL_OUT_OF_HOST_MEMORY)
        fprintf(stderr, "there is a failure to allocate resources required by "
                "the OpenCL implementation on the host.\n");
    else
        fprintf(stderr, "unknown error!\n");
    exit(E_CL_FAILURE);
}



cl_int cecl_write_buffer(cl_command_queue command_queue,
                         cl_mem buffer,
                         const char* buffer_name,
                         size_t offset,
                         size_t cb,
                         const void* ptr,
                         cl_uint num_events_in_wait_list,
                         const cl_event* event_wait_list,
                         cl_event* event) {
    cl_event local_event;

    if (!event) {  /* in case event points to NULL */
        event = &local_event;
    }

    cl_int err = clEnqueueWriteBuffer(command_queue,
                                      buffer,
                                      true,  /* blocking write */
                                      offset,
                                      cb,
                                      ptr,
                                      num_events_in_wait_list,
                                      event_wait_list,
                                      event);
    if (err == CL_SUCCESS) {
        double elapsed_ms = cecl_get_elapsed_ms(*event);

        fprintf(stderr, "\n[CECL] clEnqueueWriteBuffer ; %s ; %lu ; %.6f\n",
                buffer_name, (unsigned long)cb, elapsed_ms);
        return err;
    }
    /* error: fatal */
    fprintf(stderr, "\n[CECL] ERROR: clEnqueueWriteBuffer() failed! Cause: ");
    if (err == CL_INVALID_COMMAND_QUEUE)
        fprintf(stderr, "command_queue is not a valid command-queue.\n");
    else if (err == CL_INVALID_CONTEXT)
        fprintf(stderr, "the context associated with command_queue and buffer "
                "are not the same or if the context associated with "
                "command_queue and events in event_wait_list are not the "
                "same.\n");
    else if (err == CL_INVALID_MEM_OBJECT)
        fprintf(stderr, "buffer is not a valid buffer object.\n");
    else if (err == CL_INVALID_VALUE)
        fprintf(stderr, "the region being read specified by (offset, cb) is "
                "out of bounds or if ptr is a NULL value.\n");
    else if (err == CL_INVALID_EVENT_WAIT_LIST)
        fprintf(stderr, "event_wait_list is NULL and num_events_in_wait_list "
                "greater than 0, or event_wait_list is not NULL and "
                "num_events_in_wait_list is 0, or if event objects in "
                "event_wait_list are not valid events.\n");
    else if (err == CL_MEM_OBJECT_ALLOCATION_FAILURE)
        fprintf(stderr, "there is a failure to allocate memory for data store "
                "associated with buffer.\n");
    else if (err == CL_OUT_OF_HOST_MEMORY)
        fprintf(stderr, "there is a failure to allocate resources required by "
                "the OpenCL implementation on the host.\n");
    else
        fprintf(stderr, "unknown error!\n");
    exit(E_CL_FAILURE);
}


cl_int cecl_set_kernel_arg(cl_kernel kernel,
                           cl_uint arg_index,
                           size_t arg_size,
                           const void* arg_value,
                           const char* arg_name) {
    cl_int err = clSetKernelArg(kernel, arg_index, arg_size, arg_value);

    if (err == CL_SUCCESS) {
        char kernel_name[255];

        cecl_get_kernel_info(kernel, CL_KERNEL_FUNCTION_NAME,
                             sizeof(kernel_name), kernel_name, NULL);

        fprintf(stderr, "\n[CECL] clSetKernelArg ; %s ; %u ; %lu ; %s\n",
                kernel_name, arg_index, (unsigned long)arg_size, arg_name);
        return err;
    }
    /* error: fatal */
    fprintf(stderr, "\n[CECL] ERROR: clSetKernelArg() failed! Cause: ");
    if (err == CL_INVALID_KERNEL)
        fprintf(stderr, "kernel is not a valid kernel object.\n");
    else if (err == CL_INVALID_ARG_INDEX)
        fprintf(stderr, "arg_index is not a valid argument index.\n");
    else if (err == CL_INVALID_ARG_VALUE)
        fprintf(stderr, "arg_value specified is NULL for an argument that is "
                "not declared with the __local qualifier or vice-versa.\n");
    else if (err == CL_INVALID_MEM_OBJECT)
        fprintf(stderr, "an argument declared to be a memory object when the "
                "specified arg_value is not a valid memory object.\n");
    else if (err == CL_INVALID_SAMPLER)
        fprintf(stderr, "an argument declared to be of type sampler_t when "
                "the specified arg_value is not a valid sampler object.\n");
    else if (err == CL_INVALID_ARG_SIZE)
        fprintf(stderr, "arg_size does not match the size of the data type "
                "for an argument that is not a memory object or if the "
                "argument is a memory object and arg_size != sizeof(cl_mem) "
                "or if arg_size is zero and the argument is declared with "
                "the __local qualifier or if the argument is a sampler and "
                "arg_size != sizeof(cl_sampler).\n");
    else
        fprintf(stderr, "unknown error!\n");
    exit(E_CL_FAILURE);
}


cl_mem cecl_buffer(cl_context context,
                   cl_mem_flags flags,
                   size_t size,
                   void* host_ptr,
                   cl_int* err,
                   const char* host_ptr_name,
                   const char* flags_name) {
    cl_int local_err;
    cl_mem b = clCreateBuffer(context, flags, size, host_ptr, &local_err);

    if (local_err == CL_SUCCESS) {
        fprintf(stderr, "\n[CECL] clCreateBuffer ; %lu ; %s ; %s\n",
                (unsigned long)size, host_ptr_name, flags_name);

        if (err) {
            *err = local_err;
        }

        return b;
    }
    /* error: fatal */
    fprintf(stderr, "\n[CECL] ERROR: clCreateBuffer() failed! Cause: ");
    if (local_err == CL_INVALID_CONTEXT)
        fprintf(stderr, "context is not a valid context.\n");
    else if (local_err == CL_INVALID_VALUE)
        fprintf(stderr, "values specified in flags are not valid as defined "
                "in the table above.\n");
    else if (local_err == CL_INVALID_BUFFER_SIZE)
        fprintf(stderr, "size is 0. Implementations may return "
                "CL_INVALID_BUFFER_SIZE if size is greater than the "
                "CL_DEVICE_MAX_MEM_ALLOC_SIZE value specified in the table "
                "of allowed values for param_name for clGetDeviceInfo for "
                "all devices in context.\n");
    else if (local_err == CL_INVALID_HOST_PTR)
        fprintf(stderr, "host_ptr is NULL and CL_MEM_USE_HOST_PTR or "
                "CL_MEM_COPY_HOST_PTR are set in flags or if host_ptr is "
                "not NULL but CL_MEM_COPY_HOST_PTR or CL_MEM_USE_HOST_PTR "
                "are not set in flags.\n");
    else if (local_err == CL_MEM_OBJECT_ALLOCATION_FAILURE)
        fprintf(stderr, "there is a failure to allocate memory for buffer "
                "object.\n");
    else if (local_err == CL_OUT_OF_RESOURCES)
        fprintf(stderr, "there is a failure to allocate resources required "
                "by the OpenCL implementation on the device.\n");
    else if (local_err == CL_OUT_OF_HOST_MEMORY)
        fprintf(stderr, "there is a failure to allocate resources required "
                "by the OpenCL implementation on the host.\n");
    else
        fprintf(stderr, "unknown error!\n");
    exit(E_CL_FAILURE);
}



cl_int cecl_read_buffer(cl_command_queue command_queue,
                        cl_mem buffer,
                        const char* buffer_name,
                        size_t offset,
                        size_t cb,
                        void *ptr,
                        cl_uint num_events_in_wait_list,
                        const cl_event *event_wait_list,
                        cl_event* event) {
    cl_event local_event;

    if (!event) {  /* in case event points to NULL */
        event = &local_event;
    }

    cl_int err = clEnqueueReadBuffer(command_queue,
                                     buffer,
                                     true,  /* blocking read */
                                     offset,
                                     cb,
                                     ptr,
                                     num_events_in_wait_list,
                                     event_wait_list,
                                     event);

    if (err == CL_SUCCESS) {
        double elapsed_ms = cecl_get_elapsed_ms(*event);

        fprintf(stderr, "\n[CECL] clEnqueueReadBuffer ; %s ; %lu ; %.6f\n",
                buffer_name, (unsigned long)cb, elapsed_ms);
        return err;
    }
    /* error: fatal */
    fprintf(stderr, "\n[CECL] ERROR: clEnqueueReadBuffer() failed! Cause: ");
    if (err == CL_INVALID_COMMAND_QUEUE)
        fprintf(stderr, "command_queue is not a valid command-queue.\n");
    else if (err == CL_INVALID_CONTEXT)
        fprintf(stderr, "the context associated with command_queue and buffer "
                "are not the same or if the context associated with "
                "command_queue and events in event_wait_list are not the "
                "same.\n");
    else if (err == CL_INVALID_MEM_OBJECT)
        fprintf(stderr, "buffer is not a valid buffer object.\n");
    else if (err == CL_INVALID_VALUE)
        fprintf(stderr, "the region being read specified by (offset, size) is "
                "out of bounds or if ptr is a NULL value or if size is 0.\n");
    else if (err == CL_INVALID_EVENT_WAIT_LIST)
        fprintf(stderr, "event_wait_list is NULL and num_events_in_wait_list "
                "greater than 0, or event_wait_list is not NULL and "
                "num_events_in_wait_list is 0, or if event objects in "
                "event_wait_list are not valid events.\n");
    else if (err == CL_MISALIGNED_SUB_BUFFER_OFFSET)
        fprintf(stderr, "buffer is a sub-buffer object and offset specified "
                "when the sub-buffer object is created is not aligned to "
                "CL_DEVICE_MEM_BASE_ADDR_ALIGN value for device associated "
                "with queue.\n");
    else if (err == CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST)
        fprintf(stderr, "the read and write operations are blocking and the "
                "execution status of any of the events in event_wait_list is "
                "a negative integer value.\n");
    else if (err == CL_MEM_OBJECT_ALLOCATION_FAILURE)
        fprintf(stderr, "there is a failure to allocate memory for data store "
                "associated with buffer.\n");
    else if (err == CL_INVALID_OPERATION)
        fprintf(stderr, "clEnqueueReadBuffer is called on buffer which has "
                "been created with CL_MEM_HOST_WRITE_ONLY or "
                "CL_MEM_HOST_NO_ACCESS.\n");
    else if (err == CL_OUT_OF_RESOURCES)
        fprintf(stderr, "there is a failure to allocate resources required by "
                "the OpenCL implementation on the device.\n");
    else if (err == CL_OUT_OF_HOST_MEMORY)
        fprintf(stderr, "there is a failure to allocate resources required by "
                "the OpenCL implementation on the host.\n");
    else
        fprintf(stderr, "unknown error!\n");
    exit(E_CL_FAILURE);
}


void* cecl_map_buffer(cl_command_queue command_queue,
                      cl_mem buffer,
                      const char* buffer_name,
                      cl_map_flags map_flags,
                      size_t offset,
                      size_t cb,
                      cl_uint num_events_in_wait_list,
                      const cl_event* event_wait_list,
                      cl_event* event,
                      cl_int* errcode_ret) {
    cl_int local_err;
    cl_event local_event;

    if (!event) {  /* in case event points to NULL */
        event = &local_event;
    }

    if (!errcode_ret) {  /* in case errcode_ret points to NULL */
        errcode_ret = &local_err;
    }

    void* ret = clEnqueueMapBuffer(command_queue,
                                   buffer,
                                   true,  /* blocking map */
                                   map_flags,
                                   offset,
                                   cb,
                                   num_events_in_wait_list,
                                   event_wait_list,
                                   event,
                                   errcode_ret);

    if (*errcode_ret == CL_SUCCESS) {
        double elapsed_ms = cecl_get_elapsed_ms(*event);

        fprintf(stderr, "\n[CECL] clEnqueueMapBuffer ; %s ; %lu ; %.6f\n",
                buffer_name, (unsigned long)cb, elapsed_ms);

        return ret;
    }
    /* error: fatal */
    fprintf(stderr, "\n[CECL] ERROR: clEnqueueMapBuffer() failed! Cause: ");
    if (*errcode_ret == CL_INVALID_COMMAND_QUEUE)
        fprintf(stderr, "command_queue is not a valid command-queue.\n");
    else if (*errcode_ret == CL_INVALID_CONTEXT)
        fprintf(stderr, "the context associated with command_queue, src_image "
                "and dst_buffer are not the same or the context associated "
                "with command_queue and events in event_wait_list are not the "
                "same.\n");
    else if (*errcode_ret == CL_INVALID_MEM_OBJECT)
        fprintf(stderr, "buffer is not a valid buffer object.\n");
    else if (*errcode_ret == CL_INVALID_VALUE)
        fprintf(stderr, "region being mapped given by (offset, cb) is out of "
                "bounds or if values specified in map_flags are not valid\n");
    else if (*errcode_ret == CL_INVALID_EVENT_WAIT_LIST)
        fprintf(stderr, "event_wait_list is NULL and num_events_in_wait_list "
                "greater than 0, or event_wait_list is not NULL and "
                "num_events_in_wait_list is 0, or if event objects in "
                "event_wait_list are not valid events.\n");
    else if (*errcode_ret == CL_MAP_FAILURE)
        fprintf(stderr, "there is a failure to map the requested region into "
                "the host address space. This error cannot occur for buffer "
                "objects created with CL_MEM_USE_HOST_PTR or "
                "CL_MEM_ALLOC_HOST_PTR.\n");
    else if (*errcode_ret == CL_MEM_OBJECT_ALLOCATION_FAILURE)
        fprintf(stderr, "there is a failure to allocate memory for data "
                "store associated with buffer.\n");
    else if (*errcode_ret == CL_OUT_OF_HOST_MEMORY)
        fprintf(stderr, "there is a failure to allocate resources required by "
                "the OpenCL implementation on the host.\n");
    else
        fprintf(stderr, "unknown error!\n");
    exit(E_CL_FAILURE);
}
