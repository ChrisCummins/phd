#pragma once

#include "gpu/cldrive/kernel_values.h"
#include "gpu/cldrive/proto/cldrive.pb.h"
#include "phd/status.h"
#include "third_party/opencl/include/cl.hpp"

namespace gpu {
namespace cldrive {

class KernelArg {
 public:
  KernelArg(const cl::Context &context, cl::Kernel *kernel, size_t arg_index);

  phd::Status Init();

  std::unique_ptr<KernelValue> CreateRandom(
      const DynamicParams &dynamic_params);

  std::unique_ptr<KernelValue> CreateOnes(const DynamicParams &dynamic_params);

  bool IsMutable() const;

  bool IsPointer() const;

 private:
  cl::Context context_;
  cl::Kernel *kernel_;
  size_t arg_index_;

  // One of:
  //   CL_KERNEL_ARG_ADDRESS_GLOBAL
  //   CL_KERNEL_ARG_ADDRESS_LOCAL
  //   CL_KERNEL_ARG_ADDRESS_CONSTANT
  //   CL_KERNEL_ARG_ADDRESS_PRIVATE
  cl_kernel_arg_address_qualifier address_;
  string type_name_;
  bool is_pointer_;
};

}  // namespace cldrive
}  // namespace gpu
