#pragma once

#include "gpu/cldrive/kernel_arg_value.h"
#include "gpu/cldrive/proto/cldrive.pb.h"
#include "phd/status.h"
#include "phd/statusor.h"
#include "third_party/opencl/include/cl.hpp"

namespace gpu {
namespace cldrive {

class KernelArg {
 public:
  KernelArg(cl::Kernel *kernel, size_t arg_index);

  phd::Status Init();

  // Create a random value for this argument. If the argument is not supported,
  // returns nullptr.
  std::unique_ptr<KernelArgValue> MaybeCreateRandomValue(
     const cl::Context& context, const DynamicParams &dynamic_params);

  // Create a "ones" value for this argument. If the argument is not supported,
  // returns nullptr.
  std::unique_ptr<KernelArgValue> MaybeCreateOnesValue(
     const cl::Context& context, const DynamicParams &dynamic_params);

  // Address qualifier accessors.

  bool IsGlobal() const;

  bool IsLocal() const;

  bool IsConstant() const;

  bool IsPrivate() const;

  bool IsPointer() const;

  const string& type_name() const;

 private:
  cl::Kernel *kernel_;
  size_t arg_index_;

  cl_kernel_arg_address_qualifier address_;
  string type_name_;
  bool is_pointer_;
};

}  // namespace cldrive
}  // namespace gpu
