#include "gpu/cldrive/kernel_arg_set.h"

#include "phd/logging.h"

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
    auto arg_driver = KernelArg(context_, kernel_, i);
    if (!arg_driver.Init().ok()) {
      // Early exit if argument is not supported.
      return LogErrorOutcome(CldriveKernelInstance::UNSUPPORTED_ARGUMENTS);
    }
    if (arg_driver.IsMutable()) {
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

void KernelArgSet::SetRandom(const DynamicParams& dynamic_params,
                             KernelValuesSet* values) {
  values->Clear();
  for (auto& arg : args_) {
    values->AddKernelValue(arg.CreateRandom(dynamic_params));
  }
}

void KernelArgSet::SetOnes(const DynamicParams& dynamic_params,
                           KernelValuesSet* values) {
  values->Clear();
  for (auto& arg : args_) {
    values->AddKernelValue(arg.CreateOnes(dynamic_params));
  }
}

}  // namespace cldrive
}  // namespace gpu
