// Libcecl implementation.
//
// Provides a blocking, verbose wrappers around a subset of the OpenCL API.
//
// Copyright (c) 2016, 2017, 2018, 2019 Chris Cummins.
//
// libcecl is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// libcecl is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with libcecl.  If not, see <https://www.gnu.org/licenses/>.
#pragma once

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif  /* __cplusplus */


cl_context CECL_CREATE_CONTEXT(cl_context_properties *properties,
                               cl_uint num_devices,
                               const cl_device_id *devices,
                               void *pfn_notify (const char *errinfo,
                                                 const void *private_info,
                                                 size_t cb,
                                                 void *user_data),
                               void *user_data,
                               cl_int *errcode_ret);


cl_context CECL_CREATE_CONTEXT_FROM_TYPE(cl_context_properties *properties,
                                         cl_device_type device_type_unused,
                                         void *pfn_notify (const char *errinfo,
                                                           const void *private_info,
                                                           size_t cb,
                                                           void *user_data),
                                         void *user_data,
                                         cl_int *errcode_ret);


cl_command_queue CECL_CREATE_COMMAND_QUEUE(cl_context context,
                                           cl_device_id device,
                                           cl_command_queue_properties props,
                                           cl_int* err);


cl_program CECL_PROGRAM_WITH_SOURCE(cl_context context,
                                    cl_uint count,
                                    const char** strings,
                                    const size_t* lengths,
                                    cl_int* err);

cl_int CECL_GET_KERNEL_WORK_GROUP_INFO(cl_kernel kernel,
                                       cl_device_id unused_device,
                                       cl_kernel_work_group_info param_name,
                                       size_t param_value_size,
                                       void *param_value,
                                       size_t *param_value_size_ret);

#define CECL_PROGRAM(program, num_devices, device_list, options, \
                     pfn_notify, user_data) \
    cecl_program(program, num_devices, device_list, options, pfn_notify, \
                 user_data, #program)


#define CECL_KERNEL(program, kernel_name, err) \
    cecl_kernel(program, #program, kernel_name, err)


#define CECL_CREATE_KERNELS_IN_PROGRAM(program, num_kernels, kernels, num_kernels_ret) \
    cecl_create_kernels_in_program(program, #program, num_kernels, kernels, #kernels, num_kernels_ret)


#define CECL_MAP_BUFFER(command_queue, \
                        buffer, \
                        blocking_map__UNUSED__, \
                        map_flags, \
                        offset, \
                        cb, \
                        num_events_in_wait_list, \
                        event_wait_list, \
                        event, \
                        errcode_ret) \
    cecl_map_buffer(command_queue, buffer, #buffer, map_flags, offset, cb, \
                    num_events_in_wait_list, event_wait_list, event, \
                    errcode_ret)


#define CECL_WRITE_BUFFER(command_queue, \
                          buffer, \
                          blocking_write__UNUSED__, \
                          offset, \
                          cb, \
                          ptr, \
                          num_events_in_wait_list, \
                          event_wait_list, \
                          event) \
    cecl_write_buffer(command_queue, buffer, #buffer, offset, \
                      cb, ptr, num_events_in_wait_list, \
                      event_wait_list, event)


#define CECL_SET_KERNEL_ARG(kernel, \
                            arg_index, \
                            arg_size, \
                            arg_value) \
    cecl_set_kernel_arg(kernel, arg_index, arg_size, arg_value, #arg_value)


#define CECL_BUFFER(context, \
                    flags, \
                    size, \
                    host_ptr, \
                    err) \
    cecl_buffer(context, flags, size, host_ptr, err, #host_ptr, #flags)


#define CECL_ND_RANGE_KERNEL(command_queue, \
                             kernel, \
                             work_dim, \
                             global_work_offset, \
                             global_work_size, \
                             local_work_size, \
                             num_events_in_wait_list, \
                             event_wait_list, \
                             event) \
    cecl_nd_range_kernel(command_queue, kernel, work_dim, \
                         global_work_offset, global_work_size, \
                         local_work_size, num_events_in_wait_list, \
                         event_wait_list, event)


#define CECL_TASK(command_queue, \
                  kernel, \
                  num_events_in_wait_list, \
                  event_wait_list, \
                  event) \
    cecl_task(command_queue, kernel, #kernel, num_events_in_wait_list, \
              event_wait_list, event)


#define CECL_READ_BUFFER(command_queue, \
                         buffer, \
                         blocking_read__UNUSED__, \
                         offset, \
                         cb, \
                         ptr, \
                         num_events_in_wait_list, \
                         event_wait_list, \
                         event) \
    cecl_read_buffer(command_queue, buffer, #buffer, offset, cb, ptr, \
                     num_events_in_wait_list, event_wait_list, event)

/* Internal: */


cl_kernel cecl_kernel(cl_program  program,
                      const char* program_name,
                      const char* kernel_name,
                      cl_int* err);

cl_int cecl_create_kernels_in_program(cl_program  program,
                                      const char* program_name,
                                      cl_uint num_kernels,
                                      cl_kernel *kernels,
                                      const char* kernels_name,
                                      cl_uint *num_kernels_ret);

cl_int cecl_program(cl_program program,
                    cl_uint num_devices,
                    const cl_device_id* device_list,
                    const char* options,
                    void (*pfn_notify)(cl_program, void* user_data),
                    void* user_data,
                    const char* program_name);

cl_int cecl_write_buffer(cl_command_queue command_queue,
                         cl_mem buffer,
                         const char *buffer_name,
                         size_t offset,
                         size_t cb,
                         const void* ptr,
                         cl_uint num_events_in_wait_list,
                         const cl_event* event_wait_list,
                         cl_event* event);

cl_int cecl_set_kernel_arg(cl_kernel kernel,
                           cl_uint arg_index,
                           size_t arg_size,
                           const void* arg_value,
                           const char* arg_name);

cl_mem cecl_buffer(cl_context context,
                   cl_mem_flags flags,
                   size_t size,
                   void* host_ptr,
                   cl_int* err,
                   const char* host_ptr_name,
                   const char* flags_name);

cl_int cecl_nd_range_kernel(cl_command_queue command_queue,
                            cl_kernel kernel,
                            cl_uint work_dim,
                            const size_t* global_work_offset,
                            const size_t* global_work_size,
                            const size_t* local_work_size,
                            cl_uint num_events_in_wait_list,
                            const cl_event* event_wait_list,
                            cl_event *event);

cl_int cecl_task(cl_command_queue command_queue,
                 cl_kernel kernel,
                 cl_uint num_events_in_wait_list,
                 const cl_event* event_wait_list,
                 cl_event* event);

cl_int cecl_read_buffer(cl_command_queue command_queue,
                        cl_mem buffer,
                        const char* buffer_name,
                        size_t offset,
                        size_t cb,
                        void *ptr,
                        cl_uint num_events_in_wait_list,
                        const cl_event *event_wait_list,
                        cl_event* event);

void* cecl_map_buffer(cl_command_queue command_queue,
                      cl_mem buffer,
                      const char* buffer_name,
                      cl_map_flags map_flags,
                      size_t offset,
                      size_t cb,
                      cl_uint num_events_in_wait_list,
                      const cl_event* event_wait_list,
                      cl_event* event,
                      cl_int* errcode_ret);


#ifdef __cplusplus
}  /* extern "C" */
#endif  /* __cplusplus */
