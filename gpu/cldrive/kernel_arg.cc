#include "gpu/cldrive/kernel_arg.h"

namespace gpu {
namespace cldrive {

KernelArg::KernelArg(const cl::Context &context, cl::Kernel *kernel,
                     size_t arg_index)
    : context_(context),
      kernel_(kernel),
      arg_index_(arg_index),
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

std::unique_ptr<KernelValue> KernelArg::CreateRandom(
    const DynamicParams &dynamic_params) {
  if (IsMutable()) {
    auto arg_buffer = std::make_unique<ArrayValueWithBuffer<int>>(
        context_, dynamic_params.global_size_x());
    for (size_t i = 0; i < dynamic_params.global_size_x(); ++i) {
      arg_buffer->vector()[i] = rand();
    }
    // TODO(cec): Populate with random values.
    return std::move(arg_buffer);
  } else {
    return std::make_unique<ScalarKernelArg<int>>(
        dynamic_params.global_size_x());
  }
}

std::unique_ptr<KernelValue> KernelArg::CreateOnes(
    const DynamicParams &dynamic_params) {
  if (IsMutable()) {
    auto arg_buffer = std::make_unique<ArrayValueWithBuffer<int>>(
        context_, dynamic_params.global_size_x(), 1);
    return std::move(arg_buffer);
  } else {
    return std::make_unique<ScalarKernelArg<int>>(
        dynamic_params.global_size_x());
  }
}

bool KernelArg::IsMutable() const {
  return address_ == CL_KERNEL_ARG_ADDRESS_GLOBAL;
}

bool KernelArg::IsPointer() const { return is_pointer_; }

}  // namespace cldrive
}  // namespace gpu
