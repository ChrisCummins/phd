#include "gpu/cldrive/kernel_arg_set.h"

#include "phd/logging.h"
#include "phd/status_macros.h"

namespace gpu {
namespace cldrive {

KernelArgSet::KernelArgSet(const cl::Context& context, cl::Kernel* kernel)
    : context_(context), kernel_(kernel) {}

CldriveKernelInstance::KernelInstanceOutcome KernelArgSet::LogErrorOutcome(
    const CldriveKernelInstance::KernelInstanceOutcome& outcome) {
  LOG(ERROR) << "Kernel " << kernel_->getInfo<CL_KERNEL_FUNCTION_NAME>() << " "
             << CldriveKernelInstance::KernelInstanceOutcome_Name(outcome);
  return outcome;
}

CldriveKernelInstance::KernelInstanceOutcome KernelArgSet::Init() {
  // Early exit if the kernel has no arguments.
  size_t num_args = kernel_->getInfo<CL_KERNEL_NUM_ARGS>();
  if (!num_args) {
    return LogErrorOutcome(CldriveKernelInstance::NO_ARGUMENTS);
  }

  // Create args.
  int num_mutable_args = 0;
  for (size_t i = 0; i < num_args; ++i) {
    auto arg_driver = KernelArg(kernel_, i);
    if (!arg_driver.Init().ok()) {
      // Early exit if argument is not supported.
      return LogErrorOutcome(CldriveKernelInstance::UNSUPPORTED_ARGUMENTS);
    }
    if (arg_driver.IsGlobal()) {
      ++num_mutable_args;
    }
    args_.push_back(std::move(arg_driver));
  }

  // Early exit if the kernel has no mutable arguments.
  if (!num_mutable_args) {
    return LogErrorOutcome(CldriveKernelInstance::NO_MUTABLE_ARGUMENTS);
  }

  return CldriveKernelInstance::PASS;
}

phd::Status KernelArgSet::SetRandom(const DynamicParams& dynamic_params,
                                    KernelArgValuesSet* values) {
  values->Clear();
  for (auto& arg : args_) {
    auto value = arg.MaybeCreateRandomValue(context_, dynamic_params);
    if (value) {
      values->AddKernelArgValue(std::move(value));
    } else {
      // MaybeCreateRandomValue() returns nullptr if the argument is not supported.
      return phd::Status::UNKNOWN;
    }
  }
  return phd::Status::OK;
}

phd::Status KernelArgSet::SetOnes(const DynamicParams& dynamic_params,
                                  KernelArgValuesSet* values) {
  values->Clear();
  for (auto& arg : args_) {
    auto value = arg.MaybeCreateOnesValue(context_, dynamic_params);
    if (value) {
      values->AddKernelArgValue(std::move(value));
    } else {
      // MaybeCreateRandomValue() returns nullptr if the argument is not supported.
      return phd::Status::UNKNOWN;
    }
  }
  return phd::Status::OK;
}

}  // namespace cldrive
}  // namespace gpu
