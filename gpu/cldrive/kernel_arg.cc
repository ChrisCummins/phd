#include "gpu/cldrive/kernel_arg.h"

#include "gpu/cldrive/array_kernel_arg_value.h"
#include "gpu/cldrive/scalar_kernel_arg_value.h"

namespace gpu {
namespace cldrive {

KernelArg::KernelArg(cl::Kernel *kernel, size_t arg_index)
    : kernel_(kernel), arg_index_(arg_index),
      address_(kernel->getArgInfo<CL_KERNEL_ARG_ADDRESS_QUALIFIER>(arg_index)) {
  string full_type_name =
      kernel->getArgInfo<CL_KERNEL_ARG_TYPE_NAME>(arg_index);
  CHECK(full_type_name.size());

  // Strip the final '*' character form the type name, if present.
  is_pointer_ = full_type_name[full_type_name.size() - 2] == '*';
  if (is_pointer_) {
    type_name_ = full_type_name.substr(full_type_name.size() - 2);
  } else {
    type_name_ = full_type_name;
  }

  CHECK(IsGlobal() || IsLocal() || IsConstant() || IsPrivate());
}

phd::Status KernelArg::Init() {
  // Address qualifier is one of:
  //   CL_KERNEL_ARG_ACCESS_READ_ONLY
  //   CL_KERNEL_ARG_ACCESS_WRITE_ONLY
  //   CL_KERNEL_ARG_ACCESS_READ_WRITE
  //   CL_KERNEL_ARG_ACCESS_NONE
  //
  // If argument is not an image type, CL_KERNEL_ARG_ACCESS_NONE is returned.
  // If argument is an image type, the access qualifier specified or the
  // default access qualifier is returned.
  auto access_qualifier =
      kernel_->getArgInfo<CL_KERNEL_ARG_ACCESS_QUALIFIER>(arg_index_);
  if (access_qualifier != CL_KERNEL_ARG_ACCESS_NONE) {
    LOG(ERROR) << "Argument " << arg_index_ << " is an image type";
    return phd::Status::UNKNOWN;
  }

  return phd::Status::OK;
}

std::unique_ptr<KernelArgValue> KernelArg::MaybeCreateRandomValue(
const cl::Context& context, const DynamicParams &dynamic_params) {
  if (IsGlobal()) {
    auto arg_buffer = std::make_unique<ArrayKernelArgValueWithBuffer<int>>(
        context, dynamic_params.global_size_x());
    for (size_t i = 0; i < dynamic_params.global_size_x(); ++i) {
      arg_buffer->vector()[i] = rand();
    }
    // TODO(cec): Populate with random values.
    return std::move(arg_buffer);
  } else {
    return std::make_unique<ScalarKernelArgValue<int>>(
        dynamic_params.global_size_x());
  }
}

std::unique_ptr<KernelArgValue> KernelArg::MaybeCreateOnesValue(
  const cl::Context& context, const DynamicParams &dynamic_params) {
  if (IsGlobal()) {
    if (type_name_.compare("int")) {
      return std::make_unique<ArrayKernelArgValueWithBuffer<int>>(
          context, dynamic_params.global_size_x(), 1);
    } else if (type_name_.compare("float")) {
      return std::make_unique<ArrayKernelArgValueWithBuffer<float>>(
          context, dynamic_params.global_size_x(), 1);
    } else {
    }
  } else {
    return std::make_unique<ScalarKernelArgValue<int>>(
        dynamic_params.global_size_x());
  }
}

bool KernelArg::IsGlobal() const {
  return address_ == CL_KERNEL_ARG_ADDRESS_GLOBAL;
}

bool KernelArg::IsLocal() const {
  return address_ == CL_KERNEL_ARG_ADDRESS_LOCAL;
}

bool KernelArg::IsConstant() const {
  return address_ == CL_KERNEL_ARG_ADDRESS_CONSTANT;
}

bool KernelArg::IsPrivate() const {
  return address_ == CL_KERNEL_ARG_ADDRESS_PRIVATE;
}


const string& KernelArg::type_name() const {
  return type_name_;
}

bool KernelArg::IsPointer() const { return is_pointer_; }

}  // namespace cldrive
}  // namespace gpu
