#pragma once

#include "gpu/cldrive/kernel_arg.h"
#include "gpu/cldrive/kernel_values_set.h"
#include "gpu/cldrive/proto/cldrive.pb.h"
#include "third_party/opencl/include/cl.hpp"

namespace gpu {
namespace cldrive {

class KernelArgSet {
 public:
  KernelArgSet(const cl::Context& context, cl::Kernel* kernel);

  CldriveKernelInstance::KernelInstanceOutcome LogErrorOutcome(
      const CldriveKernelInstance::KernelInstanceOutcome& outcome);

  CldriveKernelInstance::KernelInstanceOutcome Init();

  void SetRandom(const DynamicParams& dynamic_params, KernelValuesSet* values);

  void SetOnes(const DynamicParams& dynamic_params, KernelValuesSet* values);

 private:
  cl::Context context_;
  cl::Kernel* kernel_;
  std::vector<KernelArg> args_;
};

}  // namespace cldrive
}  // namespace gpu
