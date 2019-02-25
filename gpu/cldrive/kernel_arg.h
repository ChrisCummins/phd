#pragma once

#include "gpu/cldrive/kernel_arg_value.h"
#include "gpu/cldrive/proto/cldrive.pb.h"
#include "phd/status.h"
#include "phd/statusor.h"
#include "third_party/opencl/include/cl.hpp"

namespace gpu {
namespace cldrive {

// See: https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/scalarDataTypes.html
enum OpenClArgType {
  DEFAULT_UNKNOWN,
  BOOL,
  CHAR,
  UCHAR,
  SHORT,
  USHORT,
  INT,
  UINT,
  LONG,
  ULONG,
  FLOAT,
  DOUBLE,
  HALF
};

phd::StatusOr<OpenClArgType> OpenClArgTypeFromString(const string& type_name);

class KernelArg {
 public:
  KernelArg(cl::Kernel *kernel, size_t arg_index);

  phd::Status Init();

  // Create a random value for this argument. If the argument is not supported,
  // returns nullptr.
  std::unique_ptr<KernelArgValue> TryToCreateRandomValue(
     const cl::Context& context, const DynamicParams &dynamic_params);

  // Create a "ones" value for this argument. If the argument is not supported,
  // returns nullptr.
  std::unique_ptr<KernelArgValue> TryToCreateOnesValue(
     const cl::Context& context, const DynamicParams &dynamic_params);

  // Address qualifier accessors.

  bool IsGlobal() const;

  bool IsLocal() const;

  bool IsConstant() const;

  bool IsPrivate() const;

  bool IsPointer() const;

 private:

  std::unique_ptr<KernelArgValue> TryToCreateKernelArgValue(
      const cl::Context& context, const DynamicParams &dynamic_params,
      bool rand_values);

  cl::Kernel *kernel_;
  size_t arg_index_;

  cl_kernel_arg_address_qualifier address_;
  OpenClArgType type_;
  bool is_pointer_;
};

}  // namespace cldrive
}  // namespace gpu
