#pragma once

#include "gpu/cldrive/array_kernel_arg_value.h"
#include "gpu/cldrive/kernel_arg_value.h"
#include "gpu/cldrive/scalar_kernel_arg_value.h"
#include "gpu/cldrive/proto/cldrive.pb.h"

#include "third_party/opencl/include/cl.hpp"

namespace gpu {
namespace cldrive {
namespace test {

// Create an OpenCL kernel from the given string else abort.
//
// The string must contain the OpenCL source for a single kernel,
// e.g. cl::Kernel kernel = CreateClKernel("kernel void A() {}");
cl::Kernel CreateClKernel(const string& opencl_kernel);


// Downcast a KernelArgValue to the given type.
//
// If the cast cannot be made, test aborts.
template <typename T>
T* Downcast(KernelArgValue* t) {
  CHECK(t) << "KernelArgValue pointer is null";
  auto pointer = dynamic_cast<T*>(t);
  CHECK(pointer) << "Failed to cast KernelArgValue pointer";
  return pointer;
}

DynamicParams MakeParams(size_t global_size, size_t local_size = 1);

}  // namespace test
}  // namespace cldrive
}  // namespace gpu
