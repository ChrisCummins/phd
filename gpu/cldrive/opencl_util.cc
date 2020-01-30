// Copyright (c) 2016-2020 Chris Cummins.
// This file is part of cldrive.
//
// cldrive is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// cldrive is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with cldrive.  If not, see <https://www.gnu.org/licenses/>.
#include "gpu/cldrive/opencl_util.h"

namespace gpu {
namespace cldrive {
namespace util {

void CopyHostToDevice(const cl::CommandQueue& queue, void* host_pointer,
                      const cl::Buffer& buffer, size_t buffer_size,
                      ProfilingData* profiling) {
  cl::Event event;
  queue.enqueueWriteBuffer(
      buffer, /*blocking=*/true, /*offset=*/0, /*size=*/buffer_size,
      /*ptr=*/host_pointer, /*events=*/nullptr, /*event=*/&event);

  // Set profiling data.
  profiling->transfer_nanoseconds += GetElapsedNanoseconds(event);
  profiling->transferred_bytes += buffer_size;
}

void CopyDeviceToHost(const cl::CommandQueue& queue, const cl::Buffer& buffer,
                      void* host_pointer, size_t buffer_size,
                      ProfilingData* profiling) {
  cl::Event event;
  queue.enqueueReadBuffer(
      buffer, /*blocking=*/true, /*offset=*/0, /*size=*/buffer_size,
      /*ptr=*/host_pointer, /*events=*/nullptr, /*event=*/&event);

  // Set profiling data.
  profiling->transfer_nanoseconds += GetElapsedNanoseconds(event);
  profiling->transferred_bytes += buffer_size;
}

string GetOpenClKernelName(const cl::Kernel& kernel) {
  // Rather than determine the size of the character array needed to store the
  // string, allocate a buffer that *should be* large enough. This is a
  // workaround for a bug in an OpenCL implementation.
  size_t buffer_size = 512;
  char* chars = new char[buffer_size];

  size_t actual_size;
  CHECK(clGetKernelInfo(kernel(), CL_KERNEL_FUNCTION_NAME, buffer_size, chars,
                        /*param_value_size_ret=*/&actual_size) == CL_SUCCESS);

  CHECK(actual_size <= buffer_size)
      << "OpenCL kernel name exceeds " << buffer_size << " characters";

  // Construct a string from the buffer.
  string name(chars);
  // name_size includes trailing '\0' character, name.size() does not.
  CHECK(name.size() == actual_size - 1);
  delete[] chars;

  return name;
}

string GetKernelArgTypeName(const cl::Kernel& kernel, size_t arg_index) {
  // Rather than determine the size of the character array needed to store the
  // string, allocate a buffer that *should be* large enough. This is a
  // workaround for a bug in an OpenCL implementation.
  size_t buffer_size = 512;
  char* chars = new char[buffer_size];

  size_t actual_size;
  CHECK(clGetKernelArgInfo(
            kernel(), arg_index, CL_KERNEL_ARG_TYPE_NAME, buffer_size, chars,
            /*param_value_size_ret=*/&actual_size) == CL_SUCCESS);

  CHECK(actual_size <= buffer_size)
      << "OpenCL kernel name exceeds " << buffer_size << " characters";

  // Construct a string from the buffer.
  string name(chars);
  // name_size includes trailing '\0' character, name.size() does not.
  CHECK(name.size() == actual_size - 1);
  delete[] chars;

  return name;
}

}  // namespace util
}  // namespace cldrive
}  // namespace gpu
